import torch
import torch.nn as nn
from sparse_swin import SparseSwinTransformer  # Ensure you have this file implemented

class SparseSwinAutoEncoder(nn.Module):
    def __init__(self, P, L, size, patch, dim):
        super(SparseSwinAutoEncoder, self).__init__()
        self.P, self.L, self.size, self.dim = P, L, size, dim

        self.encoder = nn.Sequential(
            nn.Conv2d(L, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, (dim * P) // patch**2, kernel_size=1),
            nn.BatchNorm2d((dim * P) // patch**2),
        )

        self.sparse_swin = SparseSwinTransformer(
            img_size=size,
            patch_size=patch,
            in_chans=(dim * P) // patch**2,
            embed_dim=dim,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
        )

        self.upscale = nn.Linear(dim, P * size * size)

        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, bias=False),
            nn.ReLU(),
        )

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        abu_est = self.encoder(x)
        cls_emb = self.sparse_swin(abu_est)

        if len(cls_emb.shape) == 3:
            cls_emb = cls_emb.mean(dim=1)

        batch_size = x.size(0)
        abu_est = self.upscale(cls_emb).view(batch_size, self.P, self.size, self.size)
        abu_est = self.smooth(abu_est)
        re_result = self.decoder(abu_est)
        return abu_est, re_result
