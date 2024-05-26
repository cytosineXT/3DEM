from pathlib import Path
from functools import partial
from math import ceil, pi, sqrt, degrees
import math

import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

from torchtyping import TensorType

from pytorch_custom_utils import save_load

from beartype import beartype
from beartype.typing import Union, Tuple, Callable, Optional, List, Dict, Any

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from einx import get_at

from x_transformers import Decoder
from x_transformers.attend import Attend
from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates

from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    top_k,
    top_p,
)

from local_attention import LocalMHA

from vector_quantize_pytorch import (
    ResidualVQ,
    ResidualLFQ
)

from net.data import derive_face_edges_from_faces, myderive_face_edges_from_faces
from net.version import __version__

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from classifier_free_guidance_pytorch import (
    classifier_free_guidance,
    TextEmbeddingReturner
)

from torch_geometric.nn.conv import SAGEConv

from gateloop_transformer import SimpleGateLoopLayer

from tqdm import tqdm

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(it):
    return it[0]

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def is_empty(l):
    return len(l) == 0

def is_tensor_empty(t: Tensor):
    return t.numel() == 0

def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return torch.cat(tensors, dim = dim)

def pad_at_dim(t, padding, dim = -1, value = 0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value = value)

def pad_to_length(t, length, dim = -1, value = 0, right = True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim = dim, value = value)

# continuous embed

def ContinuousEmbed(dim_cont):
    return nn.Sequential(
        Rearrange('... -> ... 1'),
        nn.Linear(1, dim_cont),
        nn.SiLU(),
        nn.Linear(dim_cont, dim_cont),
        nn.LayerNorm(dim_cont)
    )

# additional encoder features
# 1. angle (3), 2. area (1), 3. normals (3)  四、入射波入射角（1）or 入射矢量(3) 五、入射波频率

def coords_interanglejxt(x, y, eps = 1e-5): #给点坐标
    edge_vector = x - y #得到了三条边的矢量(从原点出发的)
    normv = l2norm(edge_vector)
    normdot = -torch.einsum('abcd,abcd->abc', normv, torch.cat((normv[:, :, -1:], normv[:, :, :-1]), dim = 2)) 
    radians = normdot.clip(-1 + eps, 1 - eps).arccos()
    angle = torch.tensor([[ [degrees(rad.item()) for rad in row] for row in matrix] for matrix in radians])
    return radians, angle

def vector_anglejxt(x, y, eps = 1e-5): #给矢量
    normdot = -torch.einsum('...d,...d->...', l2norm(x), l2norm(y)) 
    radians = normdot.clip(-1 + eps, 1 - eps).arccos()
    angle = torch.tensor([[[degrees(row.item())] for row in matrix] for matrix in radians])
    return radians, angle

def polar_to_cartesian(theta, phi):
    x = math.sin(math.radians(phi)) * math.cos(math.radians(theta))
    y = math.sin(math.radians(phi)) * math.sin(math.radians(theta))
    z = math.cos(math.radians(phi))
    return [x, y, z]

@torch.no_grad()
def get_derived_face_featuresjxt(
    face_coords: TensorType['b', 'nf', 'nvf', 3, float],  # 3 or 4 vertices with 3 coordinates输入坐标格式的face list
    in_em = [45,90,2] #\theta, \phi, ka
):
    shifted_face_coords = torch.cat((face_coords[:, :, -1:], face_coords[:, :, :-1]), dim = 2) #这是对face_coords循环移位，face_coords[:, :, -1:]取最后一个切片，face_coords[:, :, :-1]取最后一个之前的切片，然后连接在一起。
    angles, angles2  = coords_interanglejxt(face_coords, shifted_face_coords) #得到了每个三角形face的三个内角，弧度形式的，如果要角度形式的要用_的(angle2)
    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim = 2) #这里是坐标相减得到边
    normals = l2norm(torch.cross(edge1, edge2, dim = -1)) #然后也用边叉乘得到法向量，很合理
    area = torch.cross(edge1, edge2, dim = -1).norm(dim = -1, keepdim = True) * 0.5 #两边矢量叉乘模/2得到面积

    incident_angle_vec = polar_to_cartesian(in_em[0],in_em[1]) #得到入射方向的xyz矢量
    incident_angle_mtx = torch.tensor(incident_angle_vec).repeat(area.shape) #得到入射方向的矢量矩阵torch.Size([1, 33564, 3])
    incident_freq_mtx = torch.tensor(in_em[2]).repeat(area.shape) #得到入射波频率的矩阵torch.Size([1, 33564, 1])

    incident_mesh_anglehudu, incident_mesh_anglejiaodu = vector_anglejxt(normals, incident_angle_mtx) #得到入射方向和每个mesh法向的夹角

    return dict(
        angles = angles,
        area = area,
        normals = normals
    )   
'''
face_coords = tensor([[[[-0.4410, -0.0583, -0.1358],
          [-0.4377, -0.0619, -0.1303],
          [-0.4430, -0.0572, -0.1290]],
         ...,
         [[ 0.4457, -0.0392,  0.0039],
          [ 0.4444, -0.0349,  0.0022],
          [ 0.4439, -0.0353,  0.0067]]]])
shifted_face_coords = tensor([[[[-0.4430, -0.0572, -0.1290],
          [-0.4410, -0.0583, -0.1358],
          [-0.4377, -0.0619, -0.1303]],
         ...,
         [[ 0.4439, -0.0353,  0.0067],
          [ 0.4457, -0.0392,  0.0039],
          [ 0.4444, -0.0349,  0.0022]]]])
angles = tensor([[[1.0797, 1.0247, 1.0372],
         [0.5923, 1.2129, 1.3365],
         [1.3376, 0.8337, 0.9703],
         ...,
         [0.8646, 1.0079, 1.2691],
         [0.9441, 1.3377, 0.8598],
         [1.0495, 0.9411, 1.1510]]])
torch.Size([1, 33564, 3])
    角度 tensor([[[61.8628, 58.7087, 59.4285],
         [33.9337, 69.4925, 76.5738],
         [76.6365, 47.7694, 55.5940],
         ...,
         [49.5398, 57.7469, 72.7133],
         [54.0905, 76.6449, 49.2646],
         [60.1344, 53.9186, 65.9470]]])
edge1 = tensor([[[ 0.0020, -0.0011, -0.0068],
         [ 0.0039, -0.0055,  0.0066],
         [-0.0013, -0.0008,  0.0078],
         ...,
         [ 0.0018, -0.0021, -0.0044],
         [-0.0009,  0.0024,  0.0037],
         [ 0.0018, -0.0039, -0.0027]]])
edge2 = tensor([[[ 3.2505e-03, -3.6082e-03,  5.5715e-03],
         [-3.7629e-03,  3.1847e-03,  2.2070e-03],
         [-3.9454e-03,  5.4754e-03, -6.5591e-03],
         ...,
         [ 1.2815e-05,  4.0940e-03,  6.5216e-04],
         [ 1.3146e-03, -4.3237e-03,  1.7414e-03],
         [-1.3146e-03,  4.3237e-03, -1.7414e-03]]])
normals = tensor([[[-0.6752, -0.7332, -0.0809],
         [-0.6926, -0.7013, -0.1688],
         [-0.6796, -0.7101, -0.1840],
         ...,
         [ 0.9110, -0.0677,  0.4068],
         [ 0.9522,  0.3035,  0.0348],
         [ 0.9336,  0.3351,  0.1272]]])
torch.Size([1, 33564, 3])

area = tensor([[[2.2789e-05],
         [2.3805e-05],
         [2.7806e-05],
         ...,
         [9.1201e-06],
         [1.0675e-05],
         [9.9803e-06]]])
torch.Size([1, 33564, 1])
'''
# tensor helper functions

@beartype
def discretize(
    t: Tensor,
    *, #*表示后面的参数都必须以关键字参数的形式传递，也就是说，在调用这个函数时，必须明确指定参数的名称，例如discretize(t, continuous_range=(0.0, 1.0), num_discrete=128)。
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor: #在Python中，-> Tensor是一个函数注解，用来指定函数的返回类型。在这个例子中，-> Tensor表示这个函数的返回值是一个Tensor类型的对象。函数注解并不会改变函数的行为，也就是说，Python解释器不会强制函数的返回值必须符合注解的类型。然而，函数注解可以为其他开发者提供有用的信息，帮助他们理解函数的期望输入和输出。此外，一些工具，如类型检查器和集成开发环境（IDE），可以使用函数注解来提供更好的代码分析和错误检测功能。例如，如果你在一个期望返回Tensor的函数中误写了返回一个整数的代码，类型检查器就会根据函数注解来指出这个错误。
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min = 0, max = num_discrete - 1) #.round四舍五入 .long转为长整形 .clamp限制在min和max中间(一层鲁棒保险)

@beartype
def undiscretize(
    t: Tensor,
    *,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo

@beartype
def gaussian_blur_1d(
    t: Tensor,
    *,
    sigma: float = 1.
) -> Tensor:

    _, _, channels, device, dtype = *t.shape, t.device, t.dtype

    width = int(ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(-half_width, half_width + 1, dtype = dtype, device = device)

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian)

    kernel = repeat(gaussian, 'n -> c 1 n', c = channels)

    t = rearrange(t, 'b n c -> b c n')
    out = F.conv1d(t, kernel, padding = half_width, groups = channels)
    return rearrange(out, 'b c n -> b n c')

@beartype
def scatter_mean(
    tgt: Tensor,
    indices: Tensor,
    src = Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-5
):
    """
    todo: update to pytorch 2.1 and try https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_
    """
    num = tgt.scatter_add(dim, indices, src)
    den = torch.zeros_like(tgt).scatter_add(dim, indices, torch.ones_like(src))
    return num / den.clamp(min = eps)

# resnet block

class PixelNorm(Module):
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return F.normalize(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])

class SqueezeExcite(Module):
    def __init__(
        self,
        dim,
        reduction_factor = 4,
        min_dim = 16
    ):
        super().__init__()
        dim_inner = max(dim // reduction_factor, min_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.SiLU(),
            nn.Linear(dim_inner, dim),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1')
        )

    def forward(self, x, mask = None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

            num = reduce(x, 'b c n -> b c', 'sum')
            den = reduce(mask.float(), 'b 1 n -> b 1', 'sum')
            avg = num / den.clamp(min = 1e-5)
        else:
            avg = reduce(x, 'b c n -> b c', 'mean')

        return x * self.net(avg)

class Block(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = PixelNorm(dim = 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()


    def forward(self, x, mask = None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.proj(x)

        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x


class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out, dropout = dropout)
        self.excite = SqueezeExcite(dim_out)
        self.residual_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x,
        mask = None
    ):
        res = self.residual_conv(x)
        h = self.block1(x, mask = mask)
        h = self.block2(h, mask = mask) 
        h = self.excite(h, mask = mask)
        return h + res

# gateloop layers

class GateLoopBlock(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        use_heinsen = True
    ):
        super().__init__()
        self.gateloops = ModuleList([])

        for _ in range(depth):
            gateloop = SimpleGateLoopLayer(dim = dim, use_heinsen = use_heinsen)
            self.gateloops.append(gateloop)

    def forward(
        self,
        x,
        cache = None
    ):
        received_cache = exists(cache)

        if is_tensor_empty(x):
            return x, None

        if received_cache:
            prev, x = x[:, :-1], x[:, -1:]

        cache = default(cache, [])
        cache = iter(cache)

        new_caches = []
        for gateloop in self.gateloops:
            layer_cache = next(cache, None)
            out, new_cache = gateloop(x, cache = layer_cache, return_cache = True)
            new_caches.append(new_cache)
            x = x + out

        if received_cache:
            x = torch.cat((prev, x), dim = -2)

        return x, new_caches

# main classes

@save_load(version = __version__)
class MeshAutoencoder(Module):
    @beartype
    def __init__(
        self,
        num_discrete_coors = 128,   #坐标离散量
        coor_continuous_range: Tuple[float, float] = (-1., 1.), #连续坐标范围
        dim_coor_embed = 64,        #坐标embedding维度
        num_discrete_area = 128,    #面积离散量
        dim_area_embed = 16,        #面积embedding维度
        num_discrete_normals = 128, #法线离散量
        dim_normal_embed = 64,      #法线embedding维度
        num_discrete_angle = 128,   #角度离散量
        dim_angle_embed = 16,       #角度embedding维度
        encoder_dims_through_depth: Tuple[int, ...] = (
            64, 128, 256, 256, 576
        ),                          #encoder每一层的维度
        init_decoder_conv_kernel = 7,   #encoder初始卷积核大小
        decoder_dims_through_depth: Tuple[int, ...] = (
            128, 128, 128, 128,
            192, 192, 192, 192,
            256, 256, 256, 256, 256, 256,
            384, 384, 384
        ),                          #decoder每一层的维度
        dim_codebook = 192,         #codebook维度
        num_quantizers = 2,           # or 'D' in the paper！！也就是残差量化的层数
        codebook_size = 16384,        # they use 16k, shared codebook between layers
        use_residual_lfq = True,      # whether to use the latest lookup-free quantization这是什么jb
        rq_kwargs: dict = dict(     #残差量化器的参数dict
            quantize_dropout = True,
            quantize_dropout_cutoff_index = 1,
            quantize_dropout_multiple_of = 1,
        ),
        rvq_kwargs: dict = dict(    #矢量残差量化的参数dict
            kmeans_init = True,
            threshold_ema_dead_code = 2,
        ),
        rlfq_kwargs: dict = dict(   #残差lookup-free量化的参数dict
            frac_per_sample_entropy = 1.
        ),
        rvq_stochastic_sample_codes = True, #是否在残差向量量化中使用随机样本码。为True，则在残差向量量化中使用随机样本码。
        sageconv_kwargs: dict = dict(   #SAGEconv参数
            normalize = True,
            project = True
        ),
        commit_loss_weight = 0.1,   #commit loss约束损失
        bin_smooth_blur_sigma = 0.4,  # they blur the one hot discretized coordinate positions二进制平滑模糊sigma值
        attn_encoder_depth = 0,     #encoder注意力深度层数
        attn_decoder_depth = 0,     #decoder注意力深度层数
        local_attn_kwargs: dict = dict( #局部注意力参数
            dim_head = 32,
            heads = 8
        ),
        local_attn_window_size = 64,    #局部注意力窗口大小
        linear_attn_kwargs: dict = dict(    #线性注意力参数
            dim_head = 8,
            heads = 16
        ),
        use_linear_attn = True,     #是否使用线性注意力
        pad_id = -1,                #填充id，这是在处理变长面时使用的填充值
        sageconv_dropout = 0.,      #SAGEconv的dropout率
        attn_dropout = 0.,          #注意力层dropout率
        ff_dropout = 0.,            #前馈网络dropout率
        resnet_dropout = 0.,         #resnet层dropout率
        checkpoint_quantizer = False,#是否对量化器进行内存检查点
        quads = False               #是否使用四边形
    ): #我草 这里面能调的参也太NM多了吧 这炼丹能练死人
        super().__init__()

        self.num_vertices_per_face = 3 if not quads else 4
        total_coordinates_per_face = self.num_vertices_per_face * 3

        # main face coordinate embedding

        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.discretize_face_coords = partial(discretize, num_discrete = num_discrete_coors, continuous_range = coor_continuous_range) #partial是用来把某个已经定义的函数固定一部分参数做成新的函数，这里就是针对face坐标，把descretize离散化函数定制成针对face坐标的离散化函数，方便后面调用discretize_face_coords可以少写几个参数减少出错且更简洁。
        #self居然也可以存函数 我还以为只能存数据或者实例
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed) #这里还只是实例化了，离散embedding数是num_discrete_coors = 128， 每个embedding的维度是dim_coor_embed = 64
        # 后续会在encoder中使用

        # derived feature embedding

        self.discretize_angle = partial(discretize, num_discrete = num_discrete_angle, continuous_range = (0., pi))
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)

        lo, hi = coor_continuous_range
        self.discretize_area = partial(discretize, num_discrete = num_discrete_area, continuous_range = (0., (hi - lo) ** 2))
        self.area_embed = nn.Embedding(num_discrete_area, dim_area_embed)

        self.discretize_normals = partial(discretize, num_discrete = num_discrete_normals, continuous_range = coor_continuous_range)
        self.normal_embed = nn.Embedding(num_discrete_normals, dim_normal_embed)

        # attention related

        attn_kwargs = dict(
            causal = False,
            prenorm = True,
            dropout = attn_dropout,
            window_size = local_attn_window_size,
        )

        # initial dimension

        init_dim = dim_coor_embed * (3 * self.num_vertices_per_face) + dim_angle_embed * self.num_vertices_per_face + dim_normal_embed * 3 + dim_area_embed

        # project into model dimension

        self.project_in = nn.Linear(init_dim, dim_codebook)

        # initial sage conv

        sageconv_kwargs = {**sageconv_kwargs }

        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth #64, 128, 256, 256, 576
        curr_dim = init_encoder_dim

        self.init_sage_conv = SAGEConv(dim_codebook, init_encoder_dim, **sageconv_kwargs)

        self.init_encoder_act_and_norm = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(init_encoder_dim)
        )

        self.encoders = ModuleList([])

        for dim_layer in encoder_dims_through_depth:
            sage_conv = SAGEConv(
                curr_dim,
                dim_layer,
                **sageconv_kwargs
            )

            self.encoders.append(sage_conv) #这里把encoder创好了并贴上了sage
            curr_dim = dim_layer

        self.encoder_attn_blocks = ModuleList([])

        for _ in range(attn_encoder_depth):
            self.encoder_attn_blocks.append(nn.ModuleList([
                TaylorSeriesLinearAttn(curr_dim, prenorm = True, **linear_attn_kwargs) if use_linear_attn else None,
                LocalMHA(dim = curr_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(curr_dim), FeedForward(curr_dim, glu = True, dropout = ff_dropout))
            ]))

        # residual quantization

        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        self.project_dim_codebook = nn.Linear(curr_dim, dim_codebook * self.num_vertices_per_face)

        if use_residual_lfq:
            self.quantizer = ResidualLFQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                commitment_loss_weight = 1.,
                **rlfq_kwargs,
                **rq_kwargs
            )
        else:
            self.quantizer = ResidualVQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                shared_codebook = True,
                commitment_weight = 1.,
                stochastic_sample_codes = rvq_stochastic_sample_codes,
                **rvq_kwargs,
                **rq_kwargs
            )

        self.checkpoint_quantizer = checkpoint_quantizer # whether to memory checkpoint the quantizer

        self.pad_id = pad_id # for variable lengthed faces, padding quantized ids will be set to this value

        # decoder

        decoder_input_dim = dim_codebook * 3

        self.decoder_attn_blocks = ModuleList([])

        for _ in range(attn_decoder_depth):
            self.decoder_attn_blocks.append(nn.ModuleList([
                TaylorSeriesLinearAttn(decoder_input_dim, prenorm = True, **linear_attn_kwargs) if use_linear_attn else None,
                LocalMHA(dim = decoder_input_dim, **attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(decoder_input_dim), FeedForward(decoder_input_dim, glu = True, dropout = ff_dropout))
            ]))

        init_decoder_dim, *decoder_dims_through_depth = decoder_dims_through_depth
        curr_dim = init_decoder_dim

        assert is_odd(init_decoder_conv_kernel)

        self.init_decoder_conv = nn.Sequential(
            nn.Conv1d(dim_codebook * self.num_vertices_per_face, init_decoder_dim, kernel_size = init_decoder_conv_kernel, padding = init_decoder_conv_kernel // 2),
            nn.SiLU(),
            Rearrange('b c n -> b n c'),
            nn.LayerNorm(init_decoder_dim),
            Rearrange('b n c -> b c n')
        )

        self.decoders = ModuleList([])

        for dim_layer in decoder_dims_through_depth:
            resnet_block = ResnetBlock(curr_dim, dim_layer, dropout = resnet_dropout)

            self.decoders.append(resnet_block)
            curr_dim = dim_layer

        self.to_coor_logits = nn.Sequential(
            nn.Linear(curr_dim, num_discrete_coors * total_coordinates_per_face),
            Rearrange('... (v c) -> ... v c', v = total_coordinates_per_face)
        )

        # loss related

        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

    @beartype
    def encode(
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],
        faces:            TensorType['b', 'nf', 'nvf', int],
        face_edges:       TensorType['b', 'e', 2, int],
        face_mask:        TensorType['b', 'nf', bool],
        face_edges_mask:  TensorType['b', 'e', bool],
        return_face_coordinates = False,
        in_em = [45,90,2] 
    ):
        """
        einops:
        b - batch
        nf - number of faces
        nv - number of vertices (3)
        nvf - number of vertices per face (3 or 4) - triangles vs quads
        c - coordinates (3)
        d - embed dim
        """
#----------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------face预处理 清洗-------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------
        batch, num_vertices, num_coors, device = *vertices.shape, vertices.device #*表示将tuple元素展开
        _, num_faces, num_vertices_per_face = faces.shape #不关心faces的batch，取出后两个元素

        assert self.num_vertices_per_face == num_vertices_per_face #断言 检查预设每面顶点数量是否与输入数据的格式相等，不相等的话抛出异常

        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0) #这应该是把face用face mash清洗了一遍.

        # continuous face coords
        face_coords = get_at('b [nv] c, b nf mv -> b nf mv c', vertices, face_without_pad) #用清洗过的索引face和点坐标得到有坐标的face
#--------------------------------------------------------face预处理 得到特征--------------------------------------------------------------------------
        # compute derived features and embed
        # 先对内角、面积、法向量进行离散化和embedding
        derived_features = get_derived_face_featuresjxt(face_coords,in_em)
        '''
        {'angles': tensor([[[0.0155, 0.0131, 0.0128],
                [0.0175, 0.0103, 0.0189],
                [0.0172, 0.0175, 0.0128],
                ...,
                [0.0109, 0.0093, 0.0098],
                [0.0099, 0.0102, 0.0129],
                [0.0104, 0.0102, 0.0101]]]),
        'area': tensor([[[0.5000],
                [0.5000],
                [0.5000],
                ...,
                [0.5000],
                [0.5000],
                [0.5000]]]), 
        'normals': tensor([[[-0.6752, -0.7332, -0.0809],
                [-0.6926, -0.7013, -0.1688],
                [-0.6796, -0.7101, -0.1840],
                ...,
                [ 0.9110, -0.0677,  0.4068],
                [ 0.9522,  0.3035,  0.0348],
                [ 0.9336,  0.3351,  0.1272]]])}
        '''
        discrete_angle = self.discretize_angle(derived_features['angles'])
        angle_embed = self.angle_embed(discrete_angle)
        discrete_area = self.discretize_area(derived_features['area'])
        area_embed = self.area_embed(discrete_area)
        discrete_normal = self.discretize_normals(derived_features['normals'])
        normal_embed = self.normal_embed(discrete_normal)
        
        # discretize vertices for face coordinate embedding
        # 然后对面的顶点坐标本身进行离散化和embedding
        discrete_face_coords = self.discretize_face_coords(face_coords) #先把face_coords离散化
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') # 9 or 12 coordinates per face #重新排布
        face_coor_embed = self.coor_embed(discrete_face_coords) #在这里把face做成embedding
        face_coor_embed = rearrange(face_coor_embed, 'b nf c d -> b nf (c d)') #再重新排布一下

        # combine all features and project into model dimension
        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed], 'b nf *') #对于每个面，把所有embedding打包成一个embedding
        '''
        face_embed = tensor([[[ 2.4310e-01,  2.9111e-01,  2.6580e-01,  ..., -1.5440e-01,
          -2.0420e-01,  3.7383e-01],
         [ 2.4310e-01,  2.9111e-01,  2.6580e-01,  ..., -1.2294e+00,
          -1.9337e+00,  8.4947e-01],
         [ 2.4310e-01,  2.9111e-01,  2.6580e-01,  ...,  2.4511e-01,
           1.0733e+00, -5.5779e-01],
         ...,
         [ 4.8746e-01, -1.1228e+00,  1.5092e-01,  ...,  2.2940e-01,
          -8.1553e-01, -1.2686e+00],
         [ 4.8746e-01, -1.1228e+00,  1.5092e-01,  ..., -1.4312e+00,
          -2.7907e-01,  2.4965e-05],
         [ 4.8746e-01, -1.1228e+00,  1.5092e-01,  ...,  7.6037e-01,
           2.8320e-01, -2.2724e-01]]], grad_fn=<CatBackward0>)

        face_embed.shape = torch.Size([1, 33564, 832])
        '''
        face_embed = self.project_in(face_embed) #通过一个nn.linear线性层映射到codebook的维度
        '''
        face_embed = tensor([[[-0.0323,  0.5359,  0.0723,  ..., -0.4654, -0.0722, -0.6046],
         [ 0.4752, -0.0139, -0.2183,  ..., -0.3369, -0.9358, -0.1397],
         [ 0.5779,  0.2645, -0.2043,  ...,  0.6114, -0.7514, -0.0844],
         ...,
         [ 0.6366,  0.2408, -0.3227,  ...,  0.7712, -0.7024, -0.0026],
         [ 0.1601,  0.5243,  0.3687,  ...,  0.2951, -0.7114, -0.2899],
         [ 0.5180,  0.6088, -0.6458,  ...,  0.7819, -1.0257,  0.3251]]],grad_fn=<ViewBackward0>)

        face_embed.shape = torch.Size([1, 33564, 192])
        '''
        # handle variable lengths by using masked_select and masked_scatter
        # first handle edges
        # needs to be offset by number of faces for each batch
        face_index_offsets = reduce(face_mask.long(), 'b nf -> b', 'sum') #tensor([33564])
        face_index_offsets = F.pad(face_index_offsets.cumsum(dim = 0), (1, -1), value = 0) #tensor([0])
        face_index_offsets = rearrange(face_index_offsets, 'b -> b 1 1') #tensor([[[0]]])
        #这在干什么暑实没看懂，但是后面两行结果看懂了，就是矩阵变个形状，转置一下
        face_edges = face_edges + face_index_offsets
        face_edges = face_edges[face_edges_mask]
        '''
        face_edges = tensor([[    0,     0],
        [    0,     2],
        [    0,    32],
        ...,
        [33563, 33558],
        [33563, 33562],
        [33563, 33563]])'''
        face_edges = rearrange(face_edges, 'be ij -> ij be')
        '''
        face_edges = tensor([[    0,     0,     0,  ..., 33563, 33563, 33563],
        [    0,     2,    32,  ..., 33558, 33562, 33563]])'''

        # next prepare the face_mask for using masked_select and masked_scatter

        orig_face_embed_shape = face_embed.shape[:2]
        '''此时face_embed.size = torch.Size([1, 33564, 192])
        fece_embed = tensor([[[ 3.5105e-01, -5.2091e-01, -1.6991e-01,  ...,  7.4971e-01,
          -6.5130e-01,  4.1747e-01],
         [-7.0602e-01,  7.6897e-05,  6.4241e-01,  ...,  5.4609e-01,
          -7.8778e-02,  8.0678e-01],
         [-3.6842e-01, -5.6197e-01,  2.9247e-01,  ...,  7.2794e-01,
          -2.3526e-01,  9.7578e-02],
         ...,
         [-5.5673e-01, -7.2171e-01, -5.6359e-01,  ..., -7.0011e-01,
           1.1193e-01, -6.2541e-01],
         [-6.6481e-01, -8.2996e-01, -1.1909e+00,  ..., -4.4708e-01,
           1.9495e-01, -4.6977e-01],
         [-7.4308e-01, -7.4542e-01, -5.4384e-01,  ..., -6.6457e-01,
          -3.9707e-02, -9.2363e-01]]], grad_fn=<ViewBackward0>)
        '''
        face_embed = face_embed[face_mask]
        '''face_embed.shape = torch.Size([33564, 192])
        单纯少了一层空壳。
        face_embed = tensor([[-3.9904e-01,  9.9372e-01,  2.9541e-03,  ...,  1.5124e-01,
          4.9425e-01,  7.3053e-01],
        [-8.1653e-01,  1.2270e+00,  2.6121e-01,  ...,  8.7453e-01,
          5.6445e-01,  3.4768e-01],
        [-1.1087e+00,  1.0626e+00,  5.1821e-01,  ...,  3.6101e-01,
          7.0221e-01,  3.5245e-01],
        ...,
        [ 1.0146e+00,  4.2866e-01, -1.9318e-01,  ...,  6.3483e-01,
          8.1754e-01,  5.4597e-01],
        [ 6.7420e-01,  3.3596e-01,  3.4468e-01,  ...,  1.1383e+00,
          4.7155e-01,  6.5357e-01],
        [ 7.9727e-01,  5.3248e-01,  4.5458e-01,  ...,  7.5117e-01,
          5.3903e-01,  2.4292e-04]], grad_fn=<IndexBackward0>)
        '''
#----------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------face SAGEConv-------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------
        # initial sage conv followed by activation and norm 这里就开始用SAGEconv了！
        '''
        GraphSAGE算法的主要思想是，一个节点的表示可以通过其邻居节点的表示来计算。具体来说，GraphSAGE算法会对每个节点的邻居节点的表示进行聚合（例如，计算均值），然后用这个聚合结果来更新原节点的表示。
        这个SAGEConv类是一个PyTorch模块，它实现了GraphSAGE的卷积操作。这个类的主要方法是forward，它接受节点特征和边的索引作为输入，然后返回更新后的节点特征。
        类的初始化方法__init__接受一些参数，例如输入和输出特征的维度，聚合方式（例如，均值或最大值），是否对输出进行归一化等。在初始化方法中，它创建了一些线性层（Linear），这些线性层在forward方法中被用来转换节点特征。
        forward方法首先检查是否需要对输入特征进行线性变换和激活函数，然后调用propagate方法进行信息传播，即，将信息从邻居节点传播到目标节点。然后，它使用线性层对传播的结果进行转换，并根据需要添加根节点的权重和进行归一化。
        message和message_and_aggregate方法是MessagePassing类的两个重要方法，它们定义了信息传播的方式。在这个类中，message方法只是返回邻居节点的特征，而message_and_aggregate方法则使用邻接矩阵和节点特征来聚合邻居节点的信息。
        总的来说，这个类实现了GraphSAGE算法的主要步骤，包括信息传播和特征转换，它可以用于图形数据的各种深度学习任务。'''
        face_embed = self.init_sage_conv(face_embed, face_edges) #嵌入从192维变64维。这行把face_embed塞进了self.init_sage_conv = SAGEConv(dim_codebook=192, init_encoder_dim=64, **sageconv_kwargs)在前文实例化了的SAGEConv。做完之后，就是特征卷积聚合，face_embed从192维变成了64维，方便塞进encoder的ini层中。
        '''face_embed.shape = torch.Size([33564, 64])
        face_embed = tensor([[ 0.1401,  0.0086, -0.1003,  ...,  0.0069,  0.1318, -0.1211],
        [ 0.2461,  0.1744, -0.1026,  ..., -0.1006,  0.1507, -0.1749],
        [ 0.2844,  0.1525, -0.0438,  ..., -0.1603,  0.0426, -0.1349],
        ...,
        [ 0.1021,  0.0715, -0.0056,  ...,  0.1674,  0.0986,  0.1957],
        [ 0.1171,  0.0891, -0.0291,  ...,  0.0194, -0.0763,  0.1664],
        [ 0.0897,  0.1052, -0.1047,  ...,  0.0350,  0.0845,  0.0356]],  grad_fn=<DivBackward0>)'''
        face_embed = self.init_encoder_act_and_norm(face_embed) #nn.Sequential是NN层封装容器，封装了两个模块：nn.SiLU()和nn.LayerNorm(init_encoder_dim)。当一个输入传递给self.init_encoder_act_and_norm时，输入首先通过nn.SiLU()（一个激活函数），然后输出被传递到nn.LayerNorm(init_encoder_dim)（一个layer归一化操作）。不会改变矩阵大小，只会改变矩阵值
        '''face_embed.shape = torch.Size([33564, 64])
        face_embed = tensor([[ 1.3283,  0.1524, -0.7129,  ...,  0.1382,  1.2493, -0.8667],
        [ 2.0755,  1.3860, -0.8978,  ..., -0.8836,  1.1671, -1.3941],
        [ 2.5943,  1.3270, -0.3090,  ..., -1.1361,  0.3735, -0.9649],
        ...,
        [ 0.5631,  0.3020, -0.3212,  ...,  1.1430,  0.5325,  1.4043],
        [ 0.9013,  0.6579, -0.3038,  ...,  0.0773, -0.6564,  1.3453],
        [ 0.6725,  0.8077, -0.8662,  ...,  0.2088,  0.6275,  0.2141]],  grad_fn=<NativeLayerNormBackward0>)'''

        for conv in self.encoders:
            face_embed = conv(face_embed, face_edges) #这中前几个conv里是不是没有ReLU+BatchNorm？和论文里不一样？卷完成了torch.Size([33564, 576])

        shape = (*orig_face_embed_shape, face_embed.shape[-1])# (1, 33564, 576) = (*torch.Size([1, 33564]), 576)

        '''face_embed.shape = torch.Size([33564, 576])
        face_embed = tensor([[ 0.0237, -0.0056,  0.0037,  ...,  0.0550,  0.0152, -0.0247],
        [ 0.0473,  0.0081, -0.0107,  ...,  0.0405,  0.0286, -0.0288],
        [ 0.0160, -0.0050, -0.0115,  ...,  0.0591,  0.0141, -0.0341],
        ...,
        [ 0.0203,  0.0022, -0.0053,  ...,  0.0343,  0.0348, -0.0104],
        [ 0.0136,  0.0119, -0.0151,  ...,  0.0361,  0.0346, -0.0124],
        [ 0.0230,  0.0084, -0.0071,  ...,  0.0397,  0.0240, -0.0124]],  grad_fn=<DivBackward0>)'''
        face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed) #多了一层[]而已
        '''face_embed.shape = torch.Size([1, 33564, 576])
        face_embed = tensor([[[ 0.0237, -0.0056,  0.0037,  ...,  0.0550,  0.0152, -0.0247],
         [ 0.0473,  0.0081, -0.0107,  ...,  0.0405,  0.0286, -0.0288],
         [ 0.0160, -0.0050, -0.0115,  ...,  0.0591,  0.0141, -0.0341],
         ...,
         [ 0.0203,  0.0022, -0.0053,  ...,  0.0343,  0.0348, -0.0104],
         [ 0.0136,  0.0119, -0.0151,  ...,  0.0361,  0.0346, -0.0124],
         [ 0.0230,  0.0084, -0.0071,  ...,  0.0397,  0.0240, -0.0124]]],  grad_fn=<MaskedScatterBackward0>)'''

        for linear_attn, attn, ff in self.encoder_attn_blocks: #这一段直接没跑
            if exists(linear_attn):
                face_embed = linear_attn(face_embed, mask = face_mask) + face_embed

            face_embed = attn(face_embed, mask = face_mask) + face_embed
            face_embed = ff(face_embed) + face_embed

        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords

    @beartype
    def quantize(
        self,
        *,
        faces: TensorType['b', 'nf', 'nvf', int],
        face_mask: TensorType['b', 'n', bool],
        face_embed: TensorType['b', 'nf', 'd', float],
        pad_id = None,
        rvq_sample_codebook_temp = 1.
    ):
        pad_id = default(pad_id, self.pad_id)
        batch, num_faces, device = *faces.shape[:2], faces.device

        max_vertex_index = faces.amax()
        num_vertices = int(max_vertex_index.item() + 1)

        face_embed = self.project_dim_codebook(face_embed)
        face_embed = rearrange(face_embed, 'b nf (nvf d) -> b nf nvf d', nvf = self.num_vertices_per_face)

        vertex_dim = face_embed.shape[-1]
        vertices = torch.zeros((batch, num_vertices, vertex_dim), device = device)

        # create pad vertex, due to variable lengthed faces

        pad_vertex_id = num_vertices
        vertices = pad_at_dim(vertices, (0, 1), dim = -2, value = 0.)

        faces = faces.masked_fill(~rearrange(face_mask, 'b n -> b n 1'), pad_vertex_id)

        # prepare for scatter mean

        faces_with_dim = repeat(faces, 'b nf nvf -> b (nf nvf) d', d = vertex_dim)

        face_embed = rearrange(face_embed, 'b ... d -> b (...) d')

        # scatter mean

        averaged_vertices = scatter_mean(vertices, faces_with_dim, face_embed, dim = -2)

        # mask out null vertex token

        mask = torch.ones((batch, num_vertices + 1), device = device, dtype = torch.bool)
        mask[:, -1] = False

        # rvq specific kwargs

        quantize_kwargs = dict(mask = mask)

        if isinstance(self.quantizer, ResidualVQ):
            quantize_kwargs.update(sample_codebook_temp = rvq_sample_codebook_temp)

        # a quantize function that makes it memory checkpointable

        def quantize_wrapper_fn(inp):
            unquantized, quantize_kwargs = inp
            return self.quantizer(unquantized, **quantize_kwargs)

        # maybe checkpoint the quantize fn

        if self.checkpoint_quantizer:
            quantize_wrapper_fn = partial(checkpoint, quantize_wrapper_fn, use_reentrant = False)

        # residual VQ

        quantized, codes, commit_loss = quantize_wrapper_fn((averaged_vertices, quantize_kwargs))

        # gather quantized vertexes back to faces for decoding
        # now the faces have quantized vertices

        face_embed_output = get_at('b [n] d, b nf nvf -> b nf (nvf d)', quantized, faces)

        # vertex codes also need to be gathered to be organized by face sequence
        # for autoregressive learning

        codes_output = get_at('b [n] q, b nf nvf -> b (nf nvf) q', codes, faces)

        # make sure codes being outputted have this padding

        face_mask = repeat(face_mask, 'b nf -> b (nf nvf) 1', nvf = self.num_vertices_per_face)
        codes_output = codes_output.masked_fill(~face_mask, self.pad_id)

        # output quantized, codes, as well as commitment loss

        return face_embed_output, codes_output, commit_loss #codes_output是量化器输出的token

    @beartype
    def decode(
        self,
        quantized: TensorType['b', 'n', 'd', float],
        face_mask:  TensorType['b', 'n', bool]
    ):
        conv_face_mask = rearrange(face_mask, 'b n -> b 1 n')

        x = quantized #x.shape = torch.Size([1, 33564, 576])

        for linear_attn, attn, ff in self.decoder_attn_blocks:
            if exists(linear_attn):
                x = linear_attn(x, mask = face_mask) + x

            x = attn(x, mask = face_mask) + x
            x = ff(x) + x

        x = rearrange(x, 'b n d -> b d n') #x.shape = torch.Size([1, 576, 33564])
        x = x.masked_fill(~conv_face_mask, 0.)
        x = self.init_decoder_conv(x) #x.shape = torch.Size([1, 128, 33564]) 和encoder的initNN一样，576维特征变为128维

        for resnet_block in self.decoders:
            x = resnet_block(x, mask = conv_face_mask) #128 192 256 384
            '''decoder_dims_through_depth: Tuple[int, ...] = (
            128, 128, 128, 128,
            192, 192, 192, 192,
            256, 256, 256, 256, 256, 256,
            384, 384, 384  ),    '''

        return rearrange(x, 'b d n -> b n d')

    @beartype
    @torch.no_grad()
    def decode_from_codes_to_faces(
        self,
        codes: Tensor,
        face_mask: Optional[TensorType['b', 'n', bool]] = None,
        return_discrete_codes = False
    ):
        codes = rearrange(codes, 'b ... -> b (...)')

        if not exists(face_mask):
            face_mask = reduce(codes != self.pad_id, 'b (nf nvf q) -> b nf', 'all', nvf = self.num_vertices_per_face, q = self.num_quantizers)

        # handle different code shapes

        codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)

        # decode

        quantized = self.quantizer.get_output_from_indices(codes)
        quantized = rearrange(quantized, 'b (nf nvf) d -> b nf (nvf d)', nvf = self.num_vertices_per_face)

        decoded = self.decode(
            quantized,
            face_mask = face_mask
        )

        decoded = decoded.masked_fill(~face_mask[..., None], 0.)
        pred_face_coords = self.to_coor_logits(decoded)

        pred_face_coords = pred_face_coords.argmax(dim = -1)

        pred_face_coords = rearrange(pred_face_coords, '... (v c) -> ... v c', v = self.num_vertices_per_face)

        # back to continuous space

        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range
        )

        # mask out with nan

        continuous_coors = continuous_coors.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1 1'), float('nan'))

        if not return_discrete_codes:
            return continuous_coors, face_mask

        return continuous_coors, pred_face_coords, face_mask

    @torch.no_grad()
    def tokenize(self, vertices, faces, face_edges = None, **kwargs):
        assert 'return_codes' not in kwargs

        inputs = [vertices, faces, face_edges]
        inputs = [*filter(exists, inputs)]
        ndims = {i.ndim for i in inputs}

        assert len(ndims) == 1
        batch_less = first(list(ndims)) == 2

        if batch_less:
            inputs = [rearrange(i, '... -> 1 ...') for i in inputs]

        input_kwargs = dict(zip(['vertices', 'faces', 'face_edges'], inputs))

        self.eval()

        codes = self.forward(
            **input_kwargs,
            return_codes = True,
            **kwargs
        )

        if batch_less:
            codes = rearrange(codes, '1 ... -> ...')

        return codes

    @beartype
    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, float],
        faces:          TensorType['b', 'nf', 'nvf', int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        texts: Optional[List[str]] = None,
        return_codes = False,
        return_loss_breakdown = False,
        return_recon_faces = True,
        only_return_recon_faces = False,
        rvq_sample_codebook_temp = 1.,
        in_em = [45,90,2] 
    ):
        if not exists(face_edges):
            face_edges = myderive_face_edges_from_faces(faces, pad_id = self.pad_id)

        num_faces, num_face_edges, device = faces.shape[1], face_edges.shape[1], faces.device

        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')

        encoded, face_coordinates = self.encode( #从这儿进encode里 返回的encoded就是那一个跑了一溜SAGEConv得到的face_embed.size = torch.Size([1, 33564, 576]), face_coordinates.shape = torch.Size([1, 33564, 9])是一个面3个点9个坐标点？为啥一个面是tensor([35, 60, 55, 35, 60, 55, 35, 60, 55]) 我知道了因为128^3离散化了
            vertices = vertices, #顶点
            faces = faces, #面
            face_edges = face_edges, #图论边
            face_edges_mask = face_edges_mask, #边mask
            face_mask = face_mask, #面mask
            return_face_coordinates = True,
            in_em = [45,90,2] 
        )

        quantized, codes, commit_loss = self.quantize( #从这儿进量化器里 返回的quantized应该就是torch.Size([1, 33564, 576])量化后的face_embed因为矩阵大小没变但是值变了，codes不知道是个啥codes.shape = torch.Size([1, 100692, 2])，感觉就是那个codebook，回头研究吧.
            face_embed = encoded,
            faces = faces,
            face_mask = face_mask,
            rvq_sample_codebook_temp = rvq_sample_codebook_temp
        ) #出来的codes是tokenized的东西，也不是codebook，是token！

        if return_codes:
            assert not return_recon_faces, 'cannot return reconstructed faces when just returning raw codes'

            codes = codes.masked_fill(~repeat(face_mask, 'b nf -> b (nf nvf) 1', nvf = self.num_vertices_per_face), self.pad_id) #这是tokenized出来的东西
            return codes

        decode = self.decode( #从这儿进decoder里，进decoder的只有quantized，没有codes！所以是什么我也不用关心了其实，我只要把他大小对准塞进去就行。
            quantized, #quantized.shape = torch.Size([1, 33564, 576])
            face_mask = face_mask
        )
        #出来decode.shape = torch.Size([1, 33564, 384])
        pred_face_coords = self.to_coor_logits(decode) #又是一个容器，装了一个nn.linear一个rearrange，映射到torch.Size([1, 33564, 9, 128]),也就是把384维的特征映射回9*128的one-hot 128^3的离散空间坐标值。。草怎么不是onehot 也不是概率。。但是确实是离散化的坐标。可能待会连续回去就好了
        #pred_face_coords.shape = torch.Size([1, 33564, 9, 128])

        # compute reconstructed faces if needed
        if return_recon_faces or only_return_recon_faces:

            recon_faces = undiscretize( #解离散化后recon_faces.shape = torch.Size([1, 33564, 9])
                pred_face_coords.argmax(dim = -1),
                num_discrete = self.num_discrete_coors,
                continuous_range = self.coor_continuous_range,
            )

            recon_faces = rearrange(recon_faces, 'b nf (nvf c) -> b nf nvf c', nvf = self.num_vertices_per_face) #recon_faces.shape = torch.Size([1, 33564, 3, 3])
            face_mask = rearrange(face_mask, 'b nf -> b nf 1 1')
            recon_faces = recon_faces.masked_fill(~face_mask, float('nan'))
            face_mask = rearrange(face_mask, 'b nf 1 1 -> b nf')
            # 重建的面坐标存在recon_faces中

        if only_return_recon_faces:
            return recon_faces

        # prepare for recon loss
        pred_face_coords = rearrange(pred_face_coords, 'b ... c -> b c (...)') #pred_face_coords.shape从torch.Size([1, 33564, 9, 128])→torch.Size([1, 128, 302076])
        face_coordinates = rearrange(face_coordinates, 'b ... -> b 1 (...)') #face_coordinates从torch.Size([1, 33564, 9])→torch.Size([1, 1, 302076])

        # reconstruction loss on discretized coordinates on each face
        # they also smooth (blur) the one hot positions, localized label smoothing basically

        with autocast(enabled = False):
            pred_log_prob = pred_face_coords.log_softmax(dim = 1) #矩阵大小torch.Size([1, 128, 302076])没变，矩阵元素值经过了log_softmax

            target_one_hot = torch.zeros_like(pred_log_prob).scatter(1, face_coordinates, 1.) #大小还是没变torch.Size([1, 128, 302076])，但是成了0-1 onehot值 torch.mean(target_one_hot)=tensor(0.0078)；torch.max(target_one_hot)=tensor(1.0)

            if self.bin_smooth_blur_sigma >= 0.:
                target_one_hot = gaussian_blur_1d(target_one_hot, sigma = self.bin_smooth_blur_sigma) #大小还是没变torch.Size([1, 128, 302076])，但是成了0-1 onehot值 torch.mean(target_one_hot) = tensor(0.0078)；torch.max(target_one_hot)=tensor(0.9192) 好像是做了一个不改变均值的缩放，把最大值拉低了.草但是最小值还是0，这是怎么做到的。。

            # cross entropy with localized smoothing
            recon_losses = (-target_one_hot * pred_log_prob).sum(dim = 1)

            face_mask = repeat(face_mask, 'b nf -> b (nf r)', r = self.num_vertices_per_face * 3)
            recon_loss = recon_losses[face_mask].mean()

        # calculate total loss
        #recon_loss = tensor(4.9608, grad_fn=<MeanBackward0>); commit_loss = tensor([0.8444, 0.0765], grad_fn=<StackBackward0>)
        total_loss = recon_loss + \
                     commit_loss.sum() * self.commit_loss_weight

        # calculate loss breakdown if needed
        #loss_breakdown = (tensor(4.9608, grad_fn=<MeanBackward0>), tensor([0.8444, 0.0765], grad_fn=<StackBackward0>))
        loss_breakdown = (recon_loss, commit_loss)

        # some return logic
        if not return_loss_breakdown:
            if not return_recon_faces:
                return total_loss

            return recon_faces, total_loss

        if not return_recon_faces:
            return total_loss, loss_breakdown

        return recon_faces, total_loss, loss_breakdown

@save_load(version = __version__)
class MeshTransformer(Module):
    @beartype
    def __init__(
        self,
        autoencoder: MeshAutoencoder,
        *,
        dim: Union[int, Tuple[int, int]] = 512,
        max_seq_len = 8192,
        flash_attn = True,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 16,
        attn_kwargs: dict = dict(
            ff_glu = True,
            num_mem_kv = 4
        ),
        cross_attn_num_mem_kv = 4, # needed for preventing nan when dropping out text condition
        dropout = 0.,
        coarse_pre_gateloop_depth = 2,
        fine_pre_gateloop_depth = 2,
        gateloop_use_heinsen = False,
        fine_attn_depth = 2,
        fine_attn_dim_head = 32,
        fine_attn_heads = 8,
        pad_id = -1,
        condition_on_text = False,
        text_condition_model_types = ('t5',),
        text_condition_cond_drop_prob = 0.25,
        quads = False
    ):
        super().__init__()
        self.num_vertices_per_face = 3 if not quads else 4

        assert autoencoder.num_vertices_per_face == self.num_vertices_per_face, 'autoencoder and transformer must both support the same type of mesh (either all triangles, or all quads)'

        dim, dim_fine = (dim, dim) if isinstance(dim, int) else dim

        self.autoencoder = autoencoder
        set_module_requires_grad_(autoencoder, False)

        self.codebook_size = autoencoder.codebook_size
        self.num_quantizers = autoencoder.num_quantizers

        self.sos_token = nn.Parameter(torch.randn(dim_fine))
        self.eos_token_id = self.codebook_size

        # they use axial positional embeddings

        assert divisible_by(max_seq_len, self.num_vertices_per_face * self.num_quantizers), f'max_seq_len ({max_seq_len}) must be divisible by (3 x {self.num_quantizers}) = {3 * self.num_quantizers}' # 3 or 4 vertices per face, with D codes per vertex

        self.token_embed = nn.Embedding(self.codebook_size + 1, dim)

        self.quantize_level_embed = nn.Parameter(torch.randn(self.num_quantizers, dim))
        self.vertex_embed = nn.Parameter(torch.randn(self.num_vertices_per_face, dim))

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len

        # text condition

        self.condition_on_text = condition_on_text
        self.conditioner = None

        cross_attn_dim_context = None

        if condition_on_text:
            self.conditioner = TextEmbeddingReturner(
                model_types = text_condition_model_types,
                cond_drop_prob = text_condition_cond_drop_prob
            )
            cross_attn_dim_context = self.conditioner.dim_latent

        # for summarizing the vertices of each face

        self.to_face_tokens = nn.Sequential(
            nn.Linear(self.num_quantizers * self.num_vertices_per_face * dim, dim),
            nn.LayerNorm(dim)
        )

        self.coarse_gateloop_block = GateLoopBlock(dim, depth = coarse_pre_gateloop_depth, use_heinsen = gateloop_use_heinsen) if coarse_pre_gateloop_depth > 0 else None

        # main autoregressive attention network
        # attending to a face token

        self.decoder = Decoder(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            cross_attend = condition_on_text,
            cross_attn_dim_context = cross_attn_dim_context,
            cross_attn_num_mem_kv = cross_attn_num_mem_kv,
            **attn_kwargs
        )

        # projection from coarse to fine, if needed

        self.maybe_project_coarse_to_fine = nn.Linear(dim, dim_fine) if dim != dim_fine else nn.Identity()

        # address a weakness in attention

        self.fine_gateloop_block = GateLoopBlock(dim, depth = fine_pre_gateloop_depth) if fine_pre_gateloop_depth > 0 else None

        # decoding the vertices, 2-stage hierarchy

        self.fine_decoder = Decoder(
            dim = dim_fine,
            depth = fine_attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            **attn_kwargs
        )

        # to logits

        self.to_logits = nn.Linear(dim_fine, self.codebook_size + 1)

        # padding id
        # force the autoencoder to use the same pad_id given in transformer

        self.pad_id = pad_id
        autoencoder.pad_id = pad_id

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    @torch.no_grad()
    def embed_texts(self, texts: Union[str, List[str]]):
        single_text = not isinstance(texts, list)
        if single_text:
            texts = [texts]

        assert exists(self.conditioner)
        text_embeds = self.conditioner.embed_texts(texts).detach()

        if single_text:
            text_embeds = text_embeds[0]

        return text_embeds

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        prompt: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        return_codes = False,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_scale = 1.,
        cache_kv = True,
        max_seq_len = None,
        face_coords_to_file: Optional[Callable[[Tensor], Any]] = None
    ):
        max_seq_len = default(max_seq_len, self.max_seq_len)

        if exists(prompt):
            assert not exists(batch_size)

            prompt = rearrange(prompt, 'b ... -> b (...)')
            assert prompt.shape[-1] <= self.max_seq_len

            batch_size = prompt.shape[0]

        if self.condition_on_text:
            assert exists(texts) ^ exists(text_embeds), '`text` or `text_embeds` must be passed in if `condition_on_text` is set to True'
            if exists(texts):
                text_embeds = self.embed_texts(texts)

            batch_size = default(batch_size, text_embeds.shape[0])

        batch_size = default(batch_size, 1)

        codes = default(prompt, torch.empty((batch_size, 0), dtype = torch.long, device = self.device))

        curr_length = codes.shape[-1]

        cache = (None, None)

        for i in tqdm(range(curr_length, max_seq_len)):

            # example below for triangles, extrapolate for quads
            # v1([q1] [q2] [q1] [q2] [q1] [q2]) v2([eos| q1] [q2] [q1] [q2] [q1] [q2]) -> 0 1 2 3 4 5 6 7 8 9 10 11 12 -> v1(F F F F F F) v2(T F F F F F) v3(T F F F F F)

            can_eos = i != 0 and divisible_by(i, self.num_quantizers * self.num_vertices_per_face)  # only allow for eos to be decoded at the end of each face, defined as 3 or 4 vertices with D residual VQ codes

            output = self.forward_on_codes(
                codes,
                text_embeds = text_embeds,
                return_loss = False,
                return_cache = cache_kv,
                append_eos = False,
                cond_scale = cond_scale,
                cfg_routed_kwargs = dict(
                    cache = cache
                )
            )

            if cache_kv:
                logits, cache = output

                if cond_scale == 1.:
                    cache = (cache, None)
            else:
                logits = output

            logits = logits[:, -1]

            if not can_eos:
                logits[:, -1] = -torch.finfo(logits.dtype).max

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)

            if temperature == 0.:
                sample = filtered_logits.argmax(dim = -1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim = -1)
                sample = torch.multinomial(probs, 1)

            codes, _ = pack([codes, sample], 'b *')

            # check for all rows to have [eos] to terminate

            is_eos_codes = (codes == self.eos_token_id)

            if is_eos_codes.any(dim = -1).all():
                break

        # mask out to padding anything after the first eos

        mask = is_eos_codes.float().cumsum(dim = -1) >= 1
        codes = codes.masked_fill(mask, self.pad_id)

        # remove a potential extra token from eos, if breaked early

        code_len = codes.shape[-1]
        round_down_code_len = code_len // self.num_quantizers * self.num_quantizers
        codes = codes[:, :round_down_code_len]

        # early return of raw residual quantizer codes

        if return_codes:
            codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)
            return codes

        self.autoencoder.eval()
        face_coords, face_mask = self.autoencoder.decode_from_codes_to_faces(codes)

        if not exists(face_coords_to_file):
            return face_coords, face_mask

        files = [face_coords_to_file(coords[mask]) for coords, mask in zip(face_coords, face_mask)]
        return files

    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, int],
        faces:          TensorType['b', 'nf', 'nvf', int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        codes:          Optional[Tensor] = None,
        cache:          Optional[LayerIntermediates] = None,
        **kwargs
    ):
        if not exists(codes):
            codes = self.autoencoder.tokenize(
                vertices = vertices,
                faces = faces,
                face_edges = face_edges
            )

        return self.forward_on_codes(codes, cache = cache, **kwargs)

    @classifier_free_guidance
    def forward_on_codes(
        self,
        codes = None,
        return_loss = True,
        return_cache = False,
        append_eos = True,
        cache = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_drop_prob = None
    ):
        # handle text conditions

        attn_context_kwargs = dict()

        if self.condition_on_text:
            assert exists(texts) ^ exists(text_embeds), '`text` or `text_embeds` must be passed in if `condition_on_text` is set to True'

            if exists(texts):
                text_embeds = self.conditioner.embed_texts(texts)

            if exists(codes):
                assert text_embeds.shape[0] == codes.shape[0], 'batch size of texts or text embeddings is not equal to the batch size of the mesh codes'

            _, maybe_dropped_text_embeds = self.conditioner(
                text_embeds = text_embeds,
                cond_drop_prob = cond_drop_prob
            )

            attn_context_kwargs = dict(
                context = maybe_dropped_text_embeds.embed,
                context_mask = maybe_dropped_text_embeds.mask
            )

        # take care of codes that may be flattened

        if codes.ndim > 2:
            codes = rearrange(codes, 'b ... -> b (...)')

        # get some variable

        batch, seq_len, device = *codes.shape, codes.device

        assert seq_len <= self.max_seq_len, f'received codes of length {seq_len} but needs to be less than or equal to set max_seq_len {self.max_seq_len}'

        # auto append eos token

        if append_eos:
            assert exists(codes)

            code_lens = ((codes == self.pad_id).cumsum(dim = -1) == 0).sum(dim = -1)

            codes = F.pad(codes, (0, 1), value = 0)

            batch_arange = torch.arange(batch, device = device)

            batch_arange = rearrange(batch_arange, '... -> ... 1')
            code_lens = rearrange(code_lens, '... -> ... 1')

            codes[batch_arange, code_lens] = self.eos_token_id

        # if returning loss, save the labels for cross entropy

        if return_loss:
            assert seq_len > 0
            codes, labels = codes[:, :-1], codes

        # token embed (each residual VQ id)

        codes = codes.masked_fill(codes == self.pad_id, 0)
        codes = self.token_embed(codes)

        # codebook embed + absolute positions

        seq_arange = torch.arange(codes.shape[-2], device = device)

        codes = codes + self.abs_pos_emb(seq_arange)

        # embedding for quantizer level

        code_len = codes.shape[1]

        level_embed = repeat(self.quantize_level_embed, 'q d -> (r q) d', r = ceil(code_len / self.num_quantizers))
        codes = codes + level_embed[:code_len]

        # embedding for each vertex

        vertex_embed = repeat(self.vertex_embed, 'nv d -> (r nv q) d', r = ceil(code_len / (self.num_vertices_per_face * self.num_quantizers)), q = self.num_quantizers)
        codes = codes + vertex_embed[:code_len]

        # create a token per face, by summarizing the 3 or 4 vertices
        # this is similar in design to the RQ transformer from Lee et al. https://arxiv.org/abs/2203.01941

        num_tokens_per_face = self.num_quantizers * self.num_vertices_per_face

        curr_vertex_pos = code_len % num_tokens_per_face # the current intra-face vertex-code position id, needed for caching at the fine decoder stage

        code_len_is_multiple_of_face = divisible_by(code_len, num_tokens_per_face)

        next_multiple_code_len = ceil(code_len / num_tokens_per_face) * num_tokens_per_face

        codes = pad_to_length(codes, next_multiple_code_len, dim = -2)

        # grouped codes will be used for the second stage

        grouped_codes = rearrange(codes, 'b (nf n) d -> b nf n d', n = num_tokens_per_face)

        # create the coarse tokens for the first attention network

        face_codes = grouped_codes if code_len_is_multiple_of_face else grouped_codes[:, :-1]
        face_codes = rearrange(face_codes, 'b nf n d -> b nf (n d)')
        face_codes = self.to_face_tokens(face_codes)

        face_codes_len = face_codes.shape[-2]

        # cache logic

        (
            cached_attended_face_codes,
            coarse_cache,
            fine_cache,
            coarse_gateloop_cache,
            fine_gateloop_cache
        ) = cache if exists(cache) else ((None,) * 5)

        if exists(cache):
            cached_face_codes_len = cached_attended_face_codes.shape[-2]
            need_call_first_transformer = face_codes_len > cached_face_codes_len
        else:
            need_call_first_transformer = True

        should_cache_fine = not divisible_by(curr_vertex_pos + 1, num_tokens_per_face)

        # attention on face codes (coarse)

        if need_call_first_transformer:
            if exists(self.coarse_gateloop_block):
                face_codes, coarse_gateloop_cache = self.coarse_gateloop_block(face_codes, cache = coarse_gateloop_cache)

            attended_face_codes, coarse_cache = self.decoder(
                face_codes,
                cache = coarse_cache,
                return_hiddens = True,
                **attn_context_kwargs
            )

            attended_face_codes = safe_cat((cached_attended_face_codes, attended_face_codes), dim = -2)
        else:
            attended_face_codes = cached_attended_face_codes

        # maybe project from coarse to fine dimension for hierarchical transformers

        attended_face_codes = self.maybe_project_coarse_to_fine(attended_face_codes)

        # auto prepend sos token

        sos = repeat(self.sos_token, 'd -> b d', b = batch)

        attended_face_codes_with_sos, _ = pack([sos, attended_face_codes], 'b * d')

        grouped_codes = pad_to_length(grouped_codes, attended_face_codes_with_sos.shape[-2], dim = 1)
        fine_vertex_codes, _ = pack([attended_face_codes_with_sos, grouped_codes], 'b n * d')

        fine_vertex_codes = fine_vertex_codes[..., :-1, :]

        # gateloop layers

        if exists(self.fine_gateloop_block):
            fine_vertex_codes = rearrange(fine_vertex_codes, 'b nf n d -> b (nf n) d')
            orig_length = fine_vertex_codes.shape[-2]
            fine_vertex_codes = fine_vertex_codes[:, :(code_len + 1)]

            fine_vertex_codes, fine_gateloop_cache = self.fine_gateloop_block(fine_vertex_codes, cache = fine_gateloop_cache)

            fine_vertex_codes = pad_to_length(fine_vertex_codes, orig_length, dim = -2)
            fine_vertex_codes = rearrange(fine_vertex_codes, 'b (nf n) d -> b nf n d', n = num_tokens_per_face)

        # fine attention - 2nd stage

        if exists(cache):
            fine_vertex_codes = fine_vertex_codes[:, -1:]

            if exists(fine_cache):
                for attn_intermediate in fine_cache.attn_intermediates:
                    ck, cv = attn_intermediate.cached_kv
                    ck, cv = map(lambda t: rearrange(t, '(b nf) ... -> b nf ...', b = batch), (ck, cv))
                    ck, cv = map(lambda t: t[:, -1, :, :curr_vertex_pos], (ck, cv))
                    attn_intermediate.cached_kv = (ck, cv)

        one_face = fine_vertex_codes.shape[1] == 1

        fine_vertex_codes = rearrange(fine_vertex_codes, 'b nf n d -> (b nf) n d')

        if one_face:
            fine_vertex_codes = fine_vertex_codes[:, :(curr_vertex_pos + 1)]

        attended_vertex_codes, fine_cache = self.fine_decoder(
            fine_vertex_codes,
            cache = fine_cache,
            return_hiddens = True
        )

        if not should_cache_fine:
            fine_cache = None

        if not one_face:
            # reconstitute original sequence

            embed = rearrange(attended_vertex_codes, '(b nf) n d -> b (nf n) d', b = batch)
            embed = embed[:, :(code_len + 1)]
        else:
            embed = attended_vertex_codes

        # logits

        logits = self.to_logits(embed)

        if not return_loss:
            if not return_cache:
                return logits

            next_cache = (
                attended_face_codes,
                coarse_cache,
                fine_cache,
                coarse_gateloop_cache,
                fine_gateloop_cache
            )

            return logits, next_cache

        # loss

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.pad_id
        )

        return ce_loss
