import torch
import torch.nn as nn
from einops import rearrange

class PatchExpand0(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity() #nn.Identity网络结构占位层 不是这个也太逗了 用nn.Linear来Expand
        self.norm = norm_layer(dim // dim_scale) #12

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution #torch.Size([1, 4050, 24])
        x = self.expand(x)
        B, L, C = x.shape #torch.Size([1, 4050, 48])
        # assert L == H * W, "input feature has wrong size" #感觉这个可以先不管 不行的话把H W改成45 90

        # x = x.view(B, H, W, C)
        x = rearrange(x, 'b l (p c)-> b (l p) c', p=4, c=C//4) #torch.Size([1, 16200, 12])
        # x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x
    
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale

        # Expand the input dimension
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity() #nn.Identity网络结构占位层 不是这个也太逗了 用nn.Linear来Expand
        
        # Apply normalization after reshaping
        self.norm = norm_layer(dim // dim_scale) #12
        
        # PixelShuffle to increase spatial resolution
        self.pixel_shuffle = nn.PixelShuffle(dim_scale)

    def forward(self, x):
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        x = self.expand(x)  # Expand feature dimension
        B, L, C = x.shape

        # Reshape to (B, H, W, C) -> (B, C, H, W) for PixelShuffle
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        # Apply PixelShuffle (C must be divisible by dim_scale**2)
        x = self.pixel_shuffle(x)

        # Reshape back to (B, H*W, C//4)
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).view(B, -1, C)

        # Apply LayerNorm on the last dimension
        x = self.norm(x) #expected input with shape [*, 6], but got input of size[1, 16200, 24]

        return x

# Example usage
input_resolution = (45, 90)
dim = 24
patch_expand0 = PatchExpand0(input_resolution, dim)
patch_expand = PatchExpand(input_resolution, dim)
x = torch.randn(1, 4050, 24)
output0 = patch_expand0(x)
output = patch_expand(x)
print(output.shape)  # Expected output shape: (1, 16200, 12)