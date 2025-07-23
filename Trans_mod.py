import os
import pickle
import time
import scipy.io as sio
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt

import datasets
import utils
from transformer import SparseSwinAutoEncoder
from plots import plot_abundance, plot_endmembers


class Train_test:
    def __init__(self, dataset, device, skip_train=False, save=False):
        self.skip_train = skip_train
        self.device = device
        self.dataset = dataset
        self.save = save
        self.save_dir = "trans_mod_" + dataset + "/"
        os.makedirs(self.save_dir, exist_ok=True)

        # Dynamically load parameters from dataset definition
        self.data = datasets.Data(dataset, device)
        self.P = self.data.M.shape[1]
        self.L = self.data.L
        self.col = self.data.col
        self.patch = self.data.patch
        self.dim = self.data.dim
        self.LR = self.data.LR
        self.EPOCH = self.data.EPOCH
        self.beta = self.data.beta
        self.gamma = self.data.gamma
        self.weight_decay_param = self.data.weight_decay_param
        self.order_abd = self.data.order_abd
        self.order_endmem = self.data.order_endmem

        self.loader = self.data.get_loader(batch_size=self.col**2)
        self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()

    def run(self, smry=False):
        net = SparseSwinAutoEncoder(
            P=self.P,
            L=self.L,
            size=self.col,
            patch=self.patch,
            dim=self.dim
        ).to(self.device)

        if smry:
            summary(net, (1, self.L, self.col, self.col), batch_dim=None)
            return

        net.apply(net.weights_init)
        model_dict = net.state_dict()
        model_dict['decoder.0.weight'] = self.init_weight
        net.load_state_dict(model_dict)

        loss_func = nn.MSELoss(reduction='mean')
        loss_func2 = utils.SAD(self.L)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

        if not self.skip_train:
            print("Training started...")
            time_start = time.time()
            net.train()
            epo_vs_los = []

            for epoch in range(self.EPOCH):
                for x, _ in self.loader:
                    x = x.transpose(1, 0).view(1, -1, self.col, self.col).to(self.device)
                    abu_est, re_result = net(x)

                    loss_re = self.beta * loss_func(re_result, x)
                    loss_sad = self.gamma * torch.sum(loss_func2(re_result.view(1, self.L, -1).transpose(1, 2),
                                                                 x.view(1, self.L, -1).transpose(1, 2)))
                    total_loss = loss_re + loss_sad

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                    optimizer.step()

                    if epoch % 10 == 0:
                        print(f'Epoch: {epoch}, Train loss: {total_loss:.4f}, RE loss: {loss_re:.4f}, SAD loss: {loss_sad:.4f}')
                    epo_vs_los.append(total_loss.item())

                scheduler.step()

            time_end = time.time()
            print(f'Total computational cost: {time_end - time_start:.2f}s')

            if self.save:
                with open(self.save_dir + 'weights_new.pickle', 'wb') as handle:
                    pickle.dump(net.state_dict(), handle)
                sio.savemat(self.save_dir + f"{self.dataset}_losses.mat", {"losses": epo_vs_los})
        else:
            with open(self.save_dir + 'weights.pickle', 'rb') as handle:
                net.load_state_dict(pickle.load(handle))

        # Inference
        Y = self.data.get("hs_img")
        H = W = int(Y.shape[0] ** 0.5)
        x = Y.view(H, W, self.L).permute(2, 0, 1).unsqueeze(0).to(self.device)
        abu_est, re_result = net(x)

        abu_est = abu_est / torch.sum(abu_est, dim=1, keepdim=True)
        abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        target = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()
        true_endmem = self.data.get("end_mem").numpy()
        est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy().reshape((self.L, self.P))

        abu_est = abu_est[:, :, self.order_abd]
        est_endmem = est_endmem[:, self.order_endmem]

        # Save visual comparisons
        plot_abundance(target, abu_est, self.P, self.save_dir)
        plot_endmembers(true_endmem, est_endmem, self.P, self.save_dir)

        # Save outputs
        sio.savemat(self.save_dir + f"{self.dataset}_abd_map.mat", {"A_est": abu_est})
        sio.savemat(self.save_dir + f"{self.dataset}_endmem.mat", {"E_est": est_endmem})

        # Compute metrics
        re = utils.compute_re(x.view(-1, self.col, self.col).permute(1, 2, 0).cpu().numpy(),
                              re_result.detach().view(-1, self.col, self.col).permute(1, 2, 0).cpu().numpy())
        print("RE:", re)

        rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
        print("Class-wise RMSE:")
        for i, val in enumerate(rmse_cls):
            print(f"Class {i + 1}: {val:.4f}")
        print("Mean RMSE:", mean_rmse)

        sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)
        print("Class-wise SAD:")
        for i, val in enumerate(sad_cls):
            print(f"Class {i + 1}: {val:.4f}")
        print("Mean SAD:", mean_sad)

        # Log results
        with open(self.save_dir + "log1.csv", 'a') as file:
            file.write(f"LR: {self.LR}, WD: {self.weight_decay_param}, RE: {re:.4f}, SAD: {mean_sad:.4f}, RMSE: {mean_rmse:.4f}\n")

        # Save all metrics to a text file
        metrics_path = os.path.join(self.save_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"RE: {re:.6f}\n")
            f.write(f"Mean RMSE: {mean_rmse:.6f}\n")
            for i, val in enumerate(rmse_cls):
                f.write(f"RMSE Class {i+1}: {val:.6f}\n")
            f.write(f"Mean SAD: {mean_sad:.6f}\n")
            for i, val in enumerate(sad_cls):
                f.write(f"SAD Class {i+1}: {val:.6f}\n")

        # Save individual abundance maps
        for i in range(abu_est.shape[-1]):
            plt.figure(figsize=(4, 4))
            plt.imshow(abu_est[:, :, i], cmap='viridis')
            plt.axis('off')
            plt.title(f"Abundance Map {i + 1}")
            plt.colorbar()
            plt.savefig(os.path.join(self.save_dir, f"abundance_map_{i+1}.png"), bbox_inches='tight')
            plt.close()
