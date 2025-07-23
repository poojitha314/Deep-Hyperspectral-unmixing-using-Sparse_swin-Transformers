import torch.utils.data
import scipy.io as sio
import torchvision.transforms as transforms
import numpy as np

class TrainData(torch.utils.data.Dataset):
    def __init__(self, img, target, transform=None, target_transform=None):
        self.img = img.float()
        self.target = target.float()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.img[index], self.target[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.img)

class Data:
    def __init__(self, dataset, device):
        data_path = "C:/Users/polka/OneDrive/Documents/project020425/DeepTrans-HSU-main/data/" + dataset + "_dataset.mat"

        if dataset == 'samson':
            self.P, self.L, self.col = 3, 156, 95

        elif dataset == 'apex':
            self.P, self.L, self.col = 4, 285, 110

        elif dataset == 'dc':
            self.P, self.L, self.col = 6, 191, 290

        elif dataset == 'jasperRidge':
            self.P, self.L = 4, 198
            self.col = 100
            print(f"✅ Manually set jasperRidge col = {self.col}")

        elif dataset == 'Urban':
            self.P, self.L, self.col = 6, 162, 307
            print(f"✅ Set Urban dataset parameters: P={self.P}, L={self.L}, col={self.col}")

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # Load .mat file content
        mat_data = sio.loadmat(data_path)
        print(f"Keys found in {dataset}_dataset.mat:\n{list(mat_data.keys())}\n")

        # Load hyperspectral image
        if 'Y' in mat_data:
            self.Y = torch.from_numpy(mat_data['Y'].T).to(device).float()
            self.Y = self.Y / self.Y.max()
        elif 'A' in mat_data and 'M' in mat_data:
            A = mat_data['A']     # [P, H*W]
            M = mat_data['M']     # [L, P]
            Y = M @ A             # [L, H*W]
            self.Y = torch.from_numpy(Y).to(device).float().T
            self.Y = self.Y / self.Y.max()
            print(f"⚠ 'Y' not found — synthesized from A @ M.T")
        else:
            raise ValueError(f"❌ Cannot find 'Y' or reconstruct from A and M in {dataset}_dataset.mat")

        # Ground truth handling
        if dataset == 'dc':
            S_GT = mat_data['S_GT']  # shape: [H, W, P_actual]
            print(f"Original S_GT shape: {S_GT.shape}")
            S_GT = S_GT.transpose(2, 0, 1)
            actual_P = S_GT.shape[0]

            if actual_P == self.P:
                A_raw = S_GT.reshape(self.P, -1)
            else:
                print(f"[WARNING] Expected {self.P} endmembers, but found {actual_P}. Padding with zeros.")
                reshaped = S_GT.reshape(actual_P, -1)
                padded = np.zeros((self.P, reshaped.shape[1]))
                padded[:reshaped.shape[0], :] = reshaped
                A_raw = padded

            self.A = torch.from_numpy(A_raw).to(device)
            self.M = torch.from_numpy(mat_data['GT'])
            self.M1 = self.M.clone()

        elif 'A' in mat_data and 'M' in mat_data:
            self.A = torch.from_numpy(mat_data['A'].T).to(device)
            self.M = torch.from_numpy(mat_data['M'])
            self.P = self.M.shape[1]  # Infer P from shape of M
            self.M1 = torch.from_numpy(mat_data['M1']) if 'M1' in mat_data else self.M.clone()
            print(f"✅ Loaded A and M for {dataset}, using M as M1.")

        else:
            print(f"[WARNING] Ground truth not found for {dataset}. Using zero maps.")
            self.A = torch.zeros((self.col * self.col, self.P)).to(device)
            self.M = torch.rand((self.L, self.P))
            self.M1 = self.M.clone()

        # Add shared model setup params
        self.patch, self.dim = 10, 400
        self.LR, self.EPOCH = 5e-3, 150
        self.beta, self.gamma = 5e3, 2e-2
        self.weight_decay_param = 3e-5
        self.order_abd = self.order_endmem = tuple(range(self.P))

        self.init_weight = self.M1.unsqueeze(2).unsqueeze(3).float()
        self.loader = self.get_loader(batch_size=self.col**2)

    def get(self, typ):
        if typ == "hs_img":
            return self.Y.float()
        elif typ == "abd_map":
            return self.A.float()
        elif typ == "end_mem":
            return self.M
        elif typ == "init_weight":
            return self.M1

    def get_loader(self, batch_size=1):
        train_dataset = TrainData(img=self.Y, target=self.A, transform=transforms.Compose([]))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True
        )
        return train_loader
