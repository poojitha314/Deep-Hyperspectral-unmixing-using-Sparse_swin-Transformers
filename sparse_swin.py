import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock, PatchEmbed

class SparseTokenSelector(nn.Module):
    def __init__(self, keep_ratio=0.5):
        super(SparseTokenSelector, self).__init__()
        self.keep_ratio = keep_ratio

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        k = int(self.keep_ratio * N)
        scores = x.norm(p=2, dim=2)  # [B, N]
        _, idx = torch.topk(scores, k=k, dim=1)
        idx = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, k, C]
        x_sparse = torch.gather(x, dim=1, index=idx)
        return x_sparse

class SparseSwinTransformer(nn.Module):
    def __init__(
        self, img_size=96, patch_size=4, in_chans=3, embed_dim=96,
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4., qkv_bias=True,
        drop_path_rate=0.1, norm_layer=nn.LayerNorm,
    ):
        super(SparseSwinTransformer, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=0.0)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.blocks = nn.ModuleList()
        idx = 0
        for i_layer in range(len(depths)):
            for _ in range(depths[i_layer]):
                block = SwinTransformerBlock(
                    dim=embed_dim,
                    input_resolution=(img_size // patch_size, img_size // patch_size),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (_ % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[idx],
                    norm_layer=norm_layer
                )
                self.blocks.append(block)
                idx += 1

        self.norm = norm_layer(embed_dim)
        self.sparse_selector = SparseTokenSelector(keep_ratio=0.5)
    

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, N, C]
        x = self.pos_drop(x)

        # Convert [B, N, C] → [B, H, W, C]
        H_p = W_p = int(x.shape[1] ** 0.5)
        x = x.view(B, H_p, W_p, -1)

        for blk in self.blocks:
            x = blk(x)  # Swin expects 4D [B, H, W, C]

        # Back to [B, N, C] for sparse selection
        x = x.view(B, -1, x.shape[-1])
        x = self.sparse_selector(x)  # [B, reduced_N, C]
        x = self.norm(x)
        return x

    
    
