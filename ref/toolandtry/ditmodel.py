# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp
# from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# #################################################################################
# #               Embedding Layers for Timesteps and Class Labels                 #
# #################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings. 输出是N, D
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module): #要把这个labelembedder改成incident/freq embedder
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size) #所以这个标签是一个整数
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

# class EMincEmbedder(nn.Module): #要把这个labelembedder改成incident/freq embedder
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.emincembed = nn.Embedding(2, hidden_size) #所以这个标签是一个整数

#     def forward(self, in_em):
#         EMinc = in_em[1:3]
#         EMinc = torch.t(torch.stack(EMinc)) #torch.Size([1, 2])
#         EMincembeddings = self.emincembed(EMinc) #torch.Size([1, 2, 784])草不行啊，得到的是两个维度的 每个词单独embed的。。还是用方向矢量吧，这样小数也能做
#         return EMincembeddings

class EMincvecEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.emincmlp = nn.Sequential(
            nn.Linear(3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    def forward(self, in_em):
        EMinc = in_em[1]
        EMincembeddings = self.emincmlp(EMinc) #torch.Size([1, 2, 784])草不行啊，得到的是两个维度的 每个词单独embed的。。还是用方向矢量吧，这样小数也能做
        return EMincembeddings
  
   
class EMfreqEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.emfreqmlp = nn.Sequential(
            nn.Linear(1, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # self.emfreqmlp = nn.Sequential(
        #     nn.Linear(1, hidden_size, bias=True, dtype=torch.float64),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size, bias=True, dtype=torch.float64),
        # )
    def forward(self, in_em):
        EMfreq = in_em[2].unsqueeze(0).t().float()
        EMfreqembeddings = self.emfreqmlp(EMfreq)
        return EMfreqembeddings

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size):
    # def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        # self.linear = nn.Linear(hidden_size, 361*720, bias=True)
        # self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # print(x.shape)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        # print(x.shape)
        x = self.linear(x) #他这里是要先经过final linear 维度成 patch_size * patch_size * out_channels 但是为什么长度不变呢。。
        # print(x.shape)
        return x
    
class FinalLayerjxt(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, length):
    # def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear1 = nn.Linear(hidden_size, 1, bias=True)
        self.linear2 = nn.Linear(length, 361*720, bias=True)
        # self.linear = nn.Linear(hidden_size*length, 361*720, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # print('-------------------')
        # print(x.shape)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        # print(x.shape)
        x = self.linear1(x) #他这里是要先经过final linear 维度成 patch_size * patch_size * out_channels 长度不变是因为输入3维 就动最后一维不动长度。所以他还是最后用的nn.linear，回到了我的问题：can't allocate memory: you tried to allocate 1497139200000 bytes. 那就只能用我的方案，用它做成特征向量再接卷积。
        # print(x.shape)
        x = x.view(x.shape[0],-1)
        # print(x.shape)
        x = self.linear2(x)
        # print(x.shape)
        x = x.view(x.shape[0],361,-1)
        # print('-------------------')
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__( #在第一次生成DiT实例的时候跑init，根据加载参数初始化内置变量
        self,
        # input_size=32,
        # patch_size=2,
        # in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        # class_dropout_prob=0.1,
        # num_classes=1000,
        learn_sigma=True,
        length = 2500,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        # self.in_channels = in_channels
        # self.out_channels = in_channels * 2 if learn_sigma else in_channels
        # self.patch_size = patch_size
        self.num_heads = num_heads

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True) #这里应该就是patchify吧？等timm下好了看看 inputsize=32 patchsize=2 inchannels=4 hiddensize=1152
        self.t_embedder = TimestepEmbedder(hidden_size) #这是图里的timestep embed 我看看大小怎么样 和token是对上的还是怎么说
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob) #这是图里的label embed
        self.eminc_embedder = EMincvecEmbedder(hidden_size)
        # self.eminc_embedder = EMincEmbedder(hidden_size)
        self.emfreq_embedder = EMfreqEmbedder(hidden_size)
        num_patches = length
        self.tokenlength = length
        # num_patches = 22500
        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tokenlength, hidden_size), requires_grad=False) #位置嵌入的参数预加载
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False) #位置嵌入的参数预加载

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ]) #DiT block的核心部分
        self.final_layer = FinalLayerjxt(hidden_size,length=self.tokenlength) #输出层
        # self.final_layer = FinalLayer(hidden_size) #输出层
        # self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels) #输出层
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.tokenlength ** 0.5))
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(22500 ** 0.5))
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.emfreq_embedder.emfreqmlp[0].weight, std=0.02)
        nn.init.normal_(self.emfreq_embedder.emfreqmlp[2].weight, std=0.02)
        nn.init.normal_(self.eminc_embedder.emincmlp[0].weight, std=0.02)
        nn.init.normal_(self.eminc_embedder.emincmlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear1.weight, 0)
        nn.init.constant_(self.final_layer.linear1.bias, 0)
        nn.init.constant_(self.final_layer.linear2.weight, 0)
        nn.init.constant_(self.final_layer.linear2.bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)

    # def unpatchify(self, x):
    #     """
    #     x: (N, T, patch_size**2 * C)
    #     imgs: (N, H, W, C)
    #     """
    #     c = self.out_channels
    #     p = self.x_embedder.patch_size[0]
    #     h = w = int(x.shape[1] ** 0.5)
    #     assert h * w == x.shape[1]

    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    #     return imgs

    def forward(self, x, t, in_em):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = x + self.pos_embed  # (N, T, D)(1,10000,576), where T = H * W / patch_size ** 2 T就是token长度 N是batchsize吧？我就不用embedder了，直接就是token
        # x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2 T就是token长度 N是batchsize吧？
        t = self.t_embedder(t)                   # (N, D) 噪声时间步t是Diffusion的内置操作 应该不用管它
        y1 = self.eminc_embedder(in_em)
        y2 = self.emfreq_embedder(in_em)
        # y = self.y_embedder(y, self.training)    # (N, D) 这两个调制，在事实上是加起来的，也就是torchsize一样
        # c = t + y                                # (N, D)
        c = t + y1 + y2
        # c = t                               # (N, D) (1,576)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D) 然后和主特征tokens：x一起送进DiT block卷 
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels) #最后是final layer
        # x = self.unpatchify(x)                   # (N, out_channels, H, W) #最后把patchify的tokens变回图像 这一步也要动手脚才行。
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
