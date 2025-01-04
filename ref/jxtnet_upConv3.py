# from pathlib import Path
from functools import partial
from math import ceil, pi, sqrt, degrees
import math
import time
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchtyping import TensorType

from pytorch_custom_utils import save_load

from beartype import beartype
from beartype.typing import  Tuple, Optional

from einops import rearrange, repeat, reduce, pack
from einops.layers.torch import Rearrange
from einx import get_at

from x_transformers.x_transformers import RMSNorm, FeedForward

from local_attention import LocalMHA
from vector_quantize_pytorch import (
    ResidualVQ,
    ResidualLFQ
)
from net.utils import transform_to_log_coordinates, batch_psnr, batch_ssim
from net.data import derive_face_edges_from_faces
from net.version import __version__
from taylor_series_linear_attention import TaylorSeriesLinearAttn
from torch_geometric.nn.conv import SAGEConv
from gateloop_transformer import SimpleGateLoopLayer
import numpy as np
# from tqdm import tqdm

# import torch.autograd as autograd
# class SmoothgradLoss(nn.Module):
#     def __init__(self, alpha=1.0, beta=100, lambda_gp=0.1):
#         super(SmoothgradLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.lambda_gp = lambda_gp

#     def forward(self, decoded, GT):
#         # Calculate the MSE loss
#         mse_loss = nn.MSELoss(reduction='sum')
#         loss_mse = mse_loss(decoded, GT[:, :, :, 0])

#         # # Calculate the smoothness term
#         # diff1 = torch.abs(decoded[:, :-1, :] - decoded[:, 1:, :])
#         # diff2 = torch.abs(decoded[:, :, :-1] - decoded[:, :, 1:])
#         # smoothness_loss = torch.mean(diff1) + torch.mean(diff2) * self.alpha
#         batch_size, height, width = decoded.size()
#         weight = torch.randn(batch_size, 1, height, width, device=decoded.device, requires_grad=True)
#         gradients = autograd.grad(outputs=decoded, inputs=weight,
#                                   grad_outputs=torch.ones_like(decoded),
#                                   create_graph=True, retain_graph=True,
#                                   only_inputs=True, allow_unused=True)[0]
#         gradient_penalty = self.lambda_gp * torch.mean(gradients.pow(2))
#         print(f"smoothloss_gradient_penalty:{gradient_penalty},mseloss:{loss_mse}")
#         # Combine the MSE, smoothness, and gradient penalty losses
#         total_loss = loss_mse  + gradient_penalty #+ smoothness_loss * self.beta

        # return total_loss

# helper functions

def median_filter2d(img, kernel_size=5):
    pad_size = kernel_size // 2    # 计算 padding 大小
    img_padded = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode='reflect')    # 对图像进行 padding
    batch_size, channels, height, width = img_padded.shape    # 获取图像的尺寸
    unfolded = F.unfold(img_padded, kernel_size=kernel_size)    # 展开图像矩阵
    unfolded = unfolded.view(batch_size, channels, kernel_size * kernel_size, -1)    # 计算中值
    median = unfolded.median(dim=2)[0]
    median = median.view(batch_size, channels, height - 2 * pad_size, width - 2 * pad_size)    # 恢复图像尺寸
    return median

def gaussian_kernel(size, sigma, device):
    """生成一个高斯核"""
    kernel = torch.tensor([[(1/(2.0*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2))
                            for x in range(size)] for y in range(size)]).float().to(device)
    kernel /= kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)

def gaussian_filter2d(img, kernel_size=5, sigma=4.0, device='cuda:0'):
    """应用高斯滤波"""
    kernel = gaussian_kernel(kernel_size, sigma, device)
    channels = img.shape[1]
    kernel = kernel.repeat(channels, 1, 1, 1)
    padding = kernel_size // 2
    filtered_img = F.conv2d(img, kernel, padding=padding, groups=channels)
    return filtered_img

class SmoothLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(SmoothLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, decoded, GT):
        # Calculate the MSE loss
        mse_loss = nn.MSELoss(reduction='sum')
        loss_mse = mse_loss(decoded, GT)

        diff1 = torch.abs(decoded[:,  :-1, :] - decoded[:,  1:, :])
        diff2 = torch.abs(decoded[:,  :, :-1] - decoded[:,  :, 1:])
        smoothness_loss = torch.sum(diff1) + torch.sum(diff2) * self.alpha
        # print(f"smoothloss:{smoothness_loss*self.beta:.4f},mseloss:{loss_mse:.4f}")
        total_loss = loss_mse + smoothness_loss * self.beta
        return total_loss

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
    normv = l2norm(edge_vector) #torch.Size([2, 20804, 3, 3])
    normdot = -torch.einsum('abcd,abcd->abc', normv, torch.cat((normv[:, :, -1:], normv[:, :, :-1]), dim = 2))  #torch.Size([2, 20804, 3])
    radians = normdot.clip(-1 + eps, 1 - eps).arccos() #[0][0]=tensor([1.0587, 1.2161, 0.8668], device='cuda:0')
    angle = torch.tensor([[ [degrees(rad.item()) for rad in row] for row in matrix] for matrix in radians]) #[0][0]=tensor([60.6594, 69.6776, 49.6630])
    return radians, angle #torch.Size([2, 20804, 3])

def coords_interanglejxt2(x, y, eps=1e-5): #不要用爱恩斯坦求和 会变得不幸
    edge_vector = x - y
    normv = l2norm(edge_vector) #torch.Size([2, 20804, 3, 3])
    normdot = -(normv * torch.cat((normv[..., -1:], normv[..., :-1]), dim=3)).sum(dim=2) #应为torch.Size([2, 20804])
    normdot = torch.clamp(normdot, -1 + eps, 1 - eps)
    radians = torch.acos(normdot) #tensor([1.1088, 0.8747, 1.1511], device='cuda:0')
    angle = torch.rad2deg(radians) #tensor([63.5302, 50.1188, 65.9518], device='cuda:0')
    return radians, angle

def vector_anglejxt(x, y, eps = 1e-5): #给矢量
    normdot = -torch.einsum('...d,...d->...', l2norm(x), l2norm(y)) 
    radians = normdot.clip(-1 + eps, 1 - eps).arccos() #tensor(1.4104, device='cuda:0')
    angle = torch.tensor([[[degrees(row.item())] for row in matrix] for matrix in radians]) #tensor([80.8117])
    return radians, angle

def vector_anglejxt2(x, y, eps=1e-5):
    normdot = -(l2norm(x) * l2norm(y)).sum(dim=-1)
    normdot = torch.clamp(normdot, -1 + eps, 1 - eps)
    radians = normdot.acos() #tensor(1.4104, device='cuda:0')
    angle = torch.rad2deg(radians) #tensor(80.8117, device='cuda:0')
    return radians, angle

def polar_to_cartesian(theta, phi):
    x = math.sin(math.radians(phi)) * math.cos(math.radians(theta))
    y = math.sin(math.radians(phi)) * math.sin(math.radians(theta))
    z = math.cos(math.radians(phi))
    return [x, y, z]

def polar_to_cartesian2(theta, phi):
    theta_rad = torch.deg2rad(theta)
    phi_rad = torch.deg2rad(phi)
    x = torch.sin(phi_rad) * torch.cos(theta_rad)
    y = torch.sin(phi_rad) * torch.sin(theta_rad)
    z = torch.cos(phi_rad)
    return torch.stack([x, y, z], dim=1)

def jxtget_face_coords(vertices, face_indices):
    """
    获取具有坐标的面。
    参数：
    - vertices: 点坐标张量，形状为 (b, nv, mv)
    - face_indices: 面的索引张量，形状为 (b, nf, c)
    返回：
    - face_coords: 具有坐标的面，形状为 (b, nf, c, mv)
    """
    batch_size, num_faces, num_vertices_per_face = face_indices.shape#face_indices.shape=torch.Size([2, 20804, 3]),vertices.shape = torch.Size([2, 10400, 3])
    # num_coordinates = vertices.shape[-1]
    reshaped_face_indices = face_indices.reshape(batch_size, -1)  # 做一次reshape，将face_indices变为1D张量，然后用它来索引点坐标张量 # 形状为 (b, nf*c)
    face_coords = torch.gather(vertices, 1, reshaped_face_indices.unsqueeze(-1).expand(-1, -1, vertices.shape[-1])) # 使用索引张量获取具有坐标的面
    face_coords = face_coords.reshape(batch_size, num_faces, num_vertices_per_face, -1)# 还原形状
    return face_coords

@torch.no_grad()
def get_derived_face_featuresjxt(
    face_coords: TensorType['b', 'nf', 'nvf', 3, float],  # 3 or 4 vertices with 3 coordinates输入坐标格式的face list
    in_em, #\theta, \phi, ka
    device
):
    # ticcc = time.time()
    shifted_face_coords = torch.cat((face_coords[:, :, -1:], face_coords[:, :, :-1]), dim = 2).to(device) #这是对face_coords循环移位，face_coords[:, :, -1:]取最后一个切片，face_coords[:, :, :-1]取最后一个之前的切片，然后连接在一起。
    # print(f'Derived Step0用时：{(time.time()-ticcc):.4f}s')
    # ticcc = time.time()

    angles, _  = coords_interanglejxt2(face_coords, shifted_face_coords) #得到了每个三角形face的三个内角，弧度形式的，如果要角度形式的要用_的(angle2) 耗时1.64！！
    # angles, angles2  = coords_interanglejxt2(face_coords, shifted_face_coords) #得到了每个三角形face的三个内角，弧度形式的，如果要角度形式的要用_的(angle2) 耗时1.64！！
    # print(f'Derived Step1用时应该已经到头了：{(time.time()-ticcc):.4f}s')
    # ticcc = time.time()

    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim = 2) #这里是坐标相减得到边
    # print(f'Derived Step2用时：{(time.time()-ticcc):.4f}s')
    # ticcc = time.time()

    normals = l2norm(torch.cross(edge1, edge2, dim = -1)) #然后也用边叉乘得到法向量，很合理
    # print(f'Derived Step3用时：{(time.time()-ticcc):.4f}s')
    # ticcc = time.time()

    area = torch.cross(edge1, edge2, dim = -1).norm(dim = -1, keepdim = True) * 0.5 #两边矢量叉乘模/2得到面积
    # print(f'Derived Step4用时：{(time.time()-ticcc):.4f}s')
    # ticcc = time.time()

    incident_angle_vec = polar_to_cartesian2(in_em[:,0],in_em[:,1]) #得到入射方向的xyz矢量
    incident_angle_mtx = incident_angle_vec.unsqueeze(1).repeat(1, area.shape[1], 1) #得到入射方向的矢量矩阵torch.Size([batchsize, 33564, 3])
    incident_freq_mtx = in_em[:,2].unsqueeze(1).unsqueeze(2).repeat(1, area.shape[1], 1) #得到入射波频率的矩阵torch.Size([1, 33564, 1]) 感觉取对数不是那个意思，对数坐标只是看起来的，不是实际上的？
    # print(f'Derived Step5用时：{(time.time()-ticcc):.4f}s')
    # ticcc = time.time()

    incident_mesh_anglehudu, _ = vector_anglejxt2(normals, incident_angle_mtx) #得到入射方向和每个mesh法向的夹角,是在0到180度的，0-90说明面在物体屁股后面，90-180说明是正面  耗时0.68！！！
    # incident_mesh_anglehudu, incident_mesh_anglejiaodu = vector_anglejxt2(normals, incident_angle_mtx) #得到入射方向和每个mesh法向的夹角,是在0到180度的，0-90说明面在物体屁股后面，90-180说明是正面  耗时0.68！！！
    # print(f'Derived Step6用时：{(time.time()-ticcc):.4f}s')
    # ticcc = time.time()

    return dict(
        angles = angles,
        area = area,
        normals = normals,
        emnoangle = incident_mesh_anglehudu,
        emangle = incident_angle_mtx,
        emfreq = incident_freq_mtx
    ) , incident_angle_vec #这里就算回了freq=em[0][2]好像也没啥用吧，没离散化的 入射方向矢量倒是有用！
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
        num_discrete_coors = 512,   #坐标离散量改成512
        coor_continuous_range: Tuple[float, float] = (-1., 1.), #连续坐标范围
        dim_coor_embed = 64,        #坐标embedding维度
        num_discrete_area = 128,    #面积离散量
        dim_area_embed = 16,        #面积embedding维度
        num_discrete_normals = 128, #法线离散量
        dim_normal_embed = 64,      #法线embedding维度
        num_discrete_angle = 128,   #角度离散量
        dim_angle_embed = 16,       #角度embedding维度
        num_discrete_emnoangle = 128,   #法线、入射夹角离散量 jxt
        dim_emnoangle_embed = 16,       #法线、入射夹角embedding维度 jxt
        num_discrete_emangle = 128,     #入射角离散量 jxt
        dim_emangle_embed = 64,         #入射角embedding维度 jxt
        num_discrete_emfreq = 512,      #频率(Ka)离散量 jxt 2024年5月11日15:51:05 128改为512
        dim_emfreq_embed = 16,          #频率(Ka)embedding维度 jxt


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
        quads = False,               #是否使用四边形
        middim = 64
    ): #我草 这里面能调的参也太NM多了吧 这炼丹能练死人
        super().__init__()
        #----------------------------------------------------jxt decoder----------------------------------------------------------
        self.conv1d1 = nn.Conv1d(784, 1, kernel_size=10, stride=10, dilation=1 ,padding=0)
        self.fc1d1 = nn.Linear(2250, middim*45*90)

        # Decoder3
        self.upconv1 = nn.ConvTranspose2d(middim, int(middim/2), kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层1
        self.conv1_1 = nn.Conv2d(int(middim/2), int(middim/2), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.conv1_2 = nn.Conv2d(int(middim/2), int(middim/2), kernel_size=3, stride=1, padding=1)  # 添加的卷积层2
        self.bn1_1 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层1
        self.bn1_2 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层2
        self.upconv2 = nn.ConvTranspose2d(int(middim/2), int(middim/4), kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层1
        self.conv2_1 = nn.Conv2d(int(middim/4), int(middim/4), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.conv2_2 = nn.Conv2d(int(middim/4), int(middim/4), kernel_size=3, stride=1, padding=1)  # 添加的卷积层2
        self.bn2_1 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层1
        self.bn2_2 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层2
        self.upconv3 = nn.ConvTranspose2d(int(middim/4), int(middim/8), kernel_size=2, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(int(middim/8))
        self.conv3_1 = nn.Conv2d(int(middim/8), int(middim/8), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.conv3_2 = nn.Conv2d(int(middim/8), int(middim/8), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.bn3_1 = nn.BatchNorm2d(int(middim/8))  # 添加的批量归一化层1
        self.bn3_2 = nn.BatchNorm2d(int(middim/8))  # 添加的批量归一化层2
        self.conv1x1 = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)   #1×1卷积，把多的维度融合了


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

        self.discretize_emnoangle = partial(discretize, num_discrete = num_discrete_emnoangle, continuous_range = (0., pi)) #0还是-pi? 是0到pi!! jxt
        self.emnoangle_embed = nn.Embedding(num_discrete_emnoangle, dim_emnoangle_embed) #jxt
        self.discretize_emangle = partial(discretize, num_discrete = num_discrete_emangle, continuous_range = coor_continuous_range) #jxt
        self.emangle_embed = nn.Embedding(num_discrete_emangle, dim_emangle_embed) #jxt
        self.discretize_emfreq = partial(discretize, num_discrete = num_discrete_emfreq, continuous_range = (0.,1.0)) #2024年5月11日15:28:15我草 是不是没必要离散，这个情况，是不是其实我的freq本身其实就已经离散的了，不用我再人为离散化一次？只是embedding的时候他映射到embedding空间之后，隐含的空间关系就能实现我“连续回归”的目的？而且280个点离散到128个离散值，本身就有问题吧你妈的
        self.emfreq_embed = nn.Embedding(num_discrete_emfreq, dim_emfreq_embed) #jxt

        # attention related

        attn_kwargs = dict(
            causal = False,
            prenorm = True,
            dropout = attn_dropout,
            window_size = local_attn_window_size,
        )

        # initial dimension

        init_dim = dim_coor_embed * (3 * self.num_vertices_per_face) + dim_angle_embed * self.num_vertices_per_face + dim_normal_embed * 3 + dim_area_embed + dim_emangle_embed * 3 + dim_emnoangle_embed + dim_emfreq_embed

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

        '''torch.Size([1, 33564, 384])'''
        self.to_coor_logits = nn.Sequential(
            nn.Linear(curr_dim, num_discrete_coors * total_coordinates_per_face), #384, 128*9
            Rearrange('... (v c) -> ... v c', v = total_coordinates_per_face) #v=9 (9*128)->9 128
        )
        '''torch.Size([1, 33564, 9, 128])'''
        
        '''torch.Size([1, 33564, 384])'''
        self.to_coor_logits_emincident = nn.Sequential(
            nn.Linear(curr_dim, num_discrete_coors * (total_coordinates_per_face+3)), #384, 128*(9+3),3是theta phi freq jxt
            Rearrange('... (v c) -> ... v c', v = (total_coordinates_per_face+3)) #v=9 (9+3 *128)->9+3 128 jxt
        )
        '''torch.Size([1, 33564, 12, 128])'''
        # loss related

        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma
        
    # @numba.jit(nopython=False) 
    @beartype
    def encode( #Encoder
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],
        faces:            TensorType['b', 'nf', 'nvf', int],
        face_edges:       TensorType['b', 'e', 2, int],
        face_mask,
        return_face_coordinates = False,
        in_em
    ):
        '''1
        Encoder Step3用时：0.0605s
        Encoder Step4用时：0.0005s
        Encoder Step5用时fc映射也没法加速：0.0237s
        Encoder Step60 感觉可删用时：0.0004s
        Encoder Step600 感觉可删用时：0.0238s
        Encoder Step6000 感觉可删用时：0.0005s
        Encoder Step60000 感觉可删用时：0.0002s
        Encoder Step600000 感觉可删用时：0.0306s
        Encoder Step6 感觉可删用时：0.0005s'''
        '''2
        Encoder Step1用时应该已经到头了，时间来自derive里：0.0034s
        Encoder Step3用时：0.0022s
        Encoder Step4用时：0.0002s
        Encoder Step5用时fc映射也没法加速：0.0002s
        Encoder Step6 感觉可删用时：0.0000s
        Encoder Step7 感觉可删用时：0.0014s
        Encoder Step图卷积用时：0.0047s
        Encoder Step用时：0.0002s
        Encoder用时：0.0126s'''
        '''3
        Encoder用时：0.0166s'''
        # timeen = time.time()
        # ticc = time.time()
#----------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------face预处理 清洗-------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------
        device =vertices.device #*表示将tuple元素展开
        # print(f'Encoder Step00 可删用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

        face_coords = jxtget_face_coords(vertices, faces) 
        #输出torch.Size([2, 20804, 3, 3])  tensor([-0.4463, -0.0323, -0.0037], device='cuda:0') 成功！
        # print(f'Encoder Step000 可删用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
#--------------------------------------------------------face预处理 得到特征--------------------------------------------------------------------------
        # compute derived features and embed
        # 先对内角、面积、法向量进行离散化和embedding
        in_em[:,2]=transform_to_log_coordinates(in_em[:,2]) #频率转换为对数坐标 加在encoder里！

        derived_features , in_em_angle_vec = get_derived_face_featuresjxt(face_coords, in_em, device) #这一步用了2s
        # print(f'Encoder Step1用时应该已经到头了，时间来自derive里：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

        discrete_angle = self.discretize_angle(derived_features['angles'])
        angle_embed = self.angle_embed(discrete_angle)
        discrete_area = self.discretize_area(derived_features['area'])
        area_embed = self.area_embed(discrete_area)
        discrete_normal = self.discretize_normals(derived_features['normals'])
        normal_embed = self.normal_embed(discrete_normal)

        discrete_emnoangle = self.discretize_emnoangle(derived_features['emnoangle']) #jxt
        emnoangle_embed = self.emnoangle_embed(discrete_emnoangle) #jxt
        discrete_emangle = self.discretize_emangle(derived_features['emangle']) #jxt torch.Size([2, 20804, 3])好像有点问题，这dim2的三个值怎么都是一样的
        emangle_embed = self.emangle_embed(discrete_emangle) #jxt torch.Size([2, 20804, 3])
        # cemangle = self.emangle_embed()
        discrete_emfreq = self.discretize_emfreq(derived_features['emfreq']) #emfreq本来就是离散的 #jxt 2024年5月11日13:36:50我草是不是发现了一个bug，没用对离散法。
        emfreq_embed = self.emfreq_embed(discrete_emfreq) #jxt 还是要先整形离散化。。RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)，虽然是离散的采点，但是得也是整数离散值才行！
        # emfreq_embed = self.emfreq_embed(derived_features['emfreq']) #jxt 还是要先整形离散化。。RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)，虽然是离散的采点，但是得也是整数离散值才行！
        # print(f'Encoder Step2用时这一步有12步离散+嵌入看起来非常复杂应该也没戏了：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        #emangle_embed[:,0,:,:].shape=torch.Size([2, 3, 64]) #这里，进一步升级用，conditioning，可以把这个angleembed用Silu+Linear映射成(32400)然后加到每一个维度上去？或者直接映射成(32400*576)然后相加，看Diffusion就是这么做的。！！！！！！！2024年5月11日21:19:58

        # discretize vertices for face coordinate embedding
        # 然后对面的顶点坐标本身进行离散化和embedding
        discrete_face_coords = self.discretize_face_coords(face_coords) #先把face_coords离散化
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') # 9 or 12 coordinates per face #重新排布
        face_coor_embed = self.coor_embed(discrete_face_coords) #在这里把face做成embedding
        face_coor_embed = rearrange(face_coor_embed, 'b nf c d -> b nf (c d)') #再重新排布一下
        # print(f'Encoder Step3用时离散化也没法加速：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

        # combine all features and project into model dimension
        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed, emnoangle_embed, emangle_embed, emfreq_embed], 'b nf *') #对于每个面，把所有embedding打包成一个embedding
        # print(f'Encoder Step4用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        em_embed, _ = pack([emangle_embed, emfreq_embed], 'b nf *') #torch.Size([2, 20804, 3, 64]) , torch.Size([2, 20804, 1, 16]) 64*3+16=204

        face_embed = self.project_in(face_embed) #通过一个nn.linear线性层映射到codebook的维度 从1056到192
        # print(f'Encoder Step5用时fc映射也没法加速：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

        # handle variable lengths by using masked_select and masked_scatter他说是通过masked_select和masked_scatter来处理变长序列
        # first handle edges 先处理图论边edges
        # needs to be offset by number of faces for each batch
        '''Encoder Step60 感觉可删用时：0.0004s
        Encoder Step600 感觉可删用时：0.0238s
        Encoder Step6000 感觉可删用时：0.0005s
        Encoder Step60000 感觉可删用时：0.0002s
        Encoder Step600000 感觉可删用时：0.0306s
        Encoder Step6 感觉可删用时：0.0005s'''
        # face_index_offsets = reduce(face_mask.long(), 'b nf -> b', 'sum') #tensor([33564])
        # print(f'Encoder Step60 感觉可删用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        # face_index_offsets = F.pad(face_index_offsets.cumsum(dim = 0), (1, -1), value = 0) #tensor([0])耗时
        # print(f'Encoder Step600 感觉可删用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        # face_index_offsets = rearrange(face_index_offsets, 'b -> b 1 1') #tensor([[[0]]])
        # print(f'Encoder Step6000 感觉可删用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        # #这在干什么暑实没看懂，但是后面两行结果看懂了，就是矩阵变个形状，转置一下

        # face_edges = face_edges + face_index_offsets #torch.Size([2, 83216, 2])
        # print(f'Encoder Step60000 感觉可删用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        # face_edges = face_edges[face_edges_mask] #torch.Size([166432, 2])耗时
        # print(f'Encoder Step600000 感觉可删用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        face_edges = face_edges.reshape(2, -1)
        # face_edges = rearrange(face_edges, 'b i j -> b i') #torch.Size([2, 166432])
        # print(f'Encoder Step6 感觉可删用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        

        # next prepare the face_mask for using masked_select and masked_scatter 然后准备facemask
        orig_face_embed_shape = face_embed.shape[:2]
        # face_embed = face_embed[face_mask]
        face_embed = face_embed.reshape(-1, face_embed.shape[-1])
        # print(f'Encoder Step7 感觉可删用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
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
        face_embed = self.init_encoder_act_and_norm(face_embed) #nn.Sequential是NN层封装容器，封装了两个模块：nn.SiLU()和nn.LayerNorm(init_encoder_dim)。当一个输入传递给self.init_encoder_act_and_norm时，输入首先通过nn.SiLU()（一个激活函数），然后输出被传递到nn.LayerNorm(init_encoder_dim)（一个layer归一化操作）。不会改变矩阵大小，只会改变矩阵值

        for conv in self.encoders:
            face_embed = conv(face_embed, face_edges) #这中前几个conv里是不是没有ReLU+BatchNorm？和论文里不一样？卷完成了torch.Size([33564, 576])
        # print(f'Encoder Step图卷积用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

        shape = (*orig_face_embed_shape, face_embed.shape[-1])# (1, 33564, 576) = (*torch.Size([1, 33564]), 576)
        face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed) #多了一层[]而已
        # print(f'Encoder Step用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        
        # print(f'\nEncoder用时：{(time.time()-timeen):.4f}s')
        for linear_attn, attn, ff in self.encoder_attn_blocks: #这一段直接没跑
            if exists(linear_attn):
                face_embed = linear_attn(face_embed, mask = face_mask) + face_embed

            face_embed = attn(face_embed, mask = face_mask) + face_embed
            face_embed = ff(face_embed) + face_embed

        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords, em_embed, in_em_angle_vec
    
    # @numba.jit(nopython=True) 
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
        timeq=time.time()
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
        # print(f'Quantizer用时：{(time.time()-timeq):.4f}s')
        return face_embed_output, codes_output, commit_loss

    # @numba.jit(nopython=True) 
    @beartype
    def decode( #decoder输入：torchsize(1,33564,576) 长度33564待定，维度576固定
        self,
        x, #原先把xpadding成36000 现在padding成21000 现在padding成32400
        em_embed,#torch.Size([2, 20804, 3, 64]) , torch.Size([2, 20804, 1, 16]) 64*3+16=204
        in_em_angle_vec #没有离散embedding的入射矢量也拿过来了，怎么用呢
    ):
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2])
        # time0 = time.time()
        x = torch.cat([x, em_embed], dim=2) #成了576+208=784维了
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2])
        pad_size = 22500 - x.size(1)
        x = F.pad(x, (0, 0, 0, pad_size)) #x.shape=torch.Size([2, 20804, 576])
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2])

        x = x.view(x.size(0), -1, 22500)  # adjust to match the input shape, 1,576,33564
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2])
        
        # ------------------------1D Conv+FC-----------------------------
        x = self.conv1d1(x)
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2])
        x = self.fc1d1(x)
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2])

        x = x.view(x.size(0), -1, 45, 90)
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])

        # ------------------------2D upConv------------------------------
        x = self.upconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])

        x = self.upconv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = F.relu(x)
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])

        x = self.upconv3(x)
        x = self.bn3(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = F.relu(x)

        x = self.conv1x1(x)
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])

        x = x[:, :, :, :-1]
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])
        # print(f'Decodr用时：{(time.time()-time0):.4f}s')
        return x.squeeze(dim=1)

    # @numba.jit() 
    @beartype
    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, float],
        faces:          TensorType['b', 'nf', 'nvf', int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        # texts: Optional[List[str]] = None,
        # return_codes = False,
        # return_loss_breakdown = False,
        # return_recon_faces = True,
        # only_return_recon_faces = False,
        # rvq_sample_codebook_temp = 1.,
        in_em,
        GT,
        logger,
        device
    ):
        ticc = time.time()
        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)

        # num_faces, num_face_edges, device = faces.shape[1], face_edges.shape[1], faces.device

        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        # face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')
#------------------------------------------------------------------进Encoder---------------------------------------------------------------------------------------------
        '''torch.Size([1, 33564, 3])'''#
        # encoded, face_coordinates, em_embed, in_em_angle_vec = self.encode( #从这儿进encode里 返回的encoded就是那一个跑了一溜SAGEConv得到的face_embed.size = torch.Size([1, 33564, 576]), face_coordinates.shape = torch.Size([1, 33564, 9])是一个面3个点9个坐标点？为啥一个面是tensor([35, 60, 55, 35, 60, 55, 35, 60, 55]) 我知道了因为128^3离散化了
        encoded, __, em_embed, in_em_angle_vec = self.encode( #从这儿进encode里 返回的encoded就是那一个跑了一溜SAGEConv得到的face_embed.size = torch.Size([1, 33564, 576]), face_coordinates.shape = torch.Size([1, 33564, 9])是一个面3个点9个坐标点？为啥一个面是tensor([35, 60, 55, 35, 60, 55, 35, 60, 55]) 我知道了因为128^3离散化了
            vertices = vertices, #顶点
            faces = faces, #面
            face_edges = face_edges, #图论边
            face_mask = face_mask, #面mask
            return_face_coordinates = True,
            in_em = in_em
        )
        # logger.info(f'Encoder用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
#------------------------------------------------------------------进quantizer--------------------------------------------------------------------------------------------

        # quantized, codes, commit_loss = self.quantize( #从这儿进量化器里 返回的quantized应该就是torch.Size([1, 33564, 576])量化后的face_embed因为矩阵大小没变但是值变了，codes不知道是个啥codes.shape = torch.Size([1, 100692, 2])，感觉就是那个codebook，回头研究吧.
        #     face_embed = encoded,
        #     faces = faces,
        #     face_mask = face_mask,
        #     rvq_sample_codebook_temp = 1.0
        # )
        # logger.info(f'Quantier用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

        # # torch.save(quantized, 'quantized33564.pt') #jxt
        # # torch.save(codes, 'codes100692.pt') #jxt
        # # if return_codes:
        # #     assert not return_recon_faces, 'cannot return reconstructed faces when just returning raw codes'
        # #     codes = codes.masked_fill(~repeat(face_mask, 'b nf -> b (nf nvf) 1', nvf = self.num_vertices_per_face), self.pad_id)
        # #     return codes
#------------------------------------------------------------------进Decoder---------------------------------------------------------------------------------------------

        decoded = self.decode( #从这儿进decoder里，进decoder的只有quantized，没有codes！所以是什么我也不用关心了其实，我只要把他大小对准塞进去就行。
            encoded, #quantized.shape = torch.Size([1, 33564, 576])
            # quantized, #quantized.shape = torch.Size([1, 33564, 576])
            em_embed,
            in_em_angle_vec
        )
        # logger.info(f'Decoder用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        '''torch.Size([1, 361, 720])'''
#------------------------------------------------------------------出Decoder了，后面都是后处理---------------------------------------------------------------------------------------------
        # decoded = decoded * 2 #人为倍增一下
        # torch.save(GT,'gt.pt')
        # torch.save(decoded,'decoded.pt')

        #平滑后处理：中值滤波+高斯滤波+修改后的smoothloss
        decoded = decoded.unsqueeze(1)  # 添加 channel 维度
        decoded = median_filter2d(decoded, kernel_size=5)# 应用中值滤波
        decoded = gaussian_filter2d(decoded, kernel_size=5, sigma=4, device=device)#两个都用 这个效果好
        decoded = decoded.squeeze(1)

        if GT == None:
            return decoded
        else:
            # print(f'decoded:{decoded.shape},GT:{GT.shape}') #decoded:torch.Size([361, 720]),GT:torch.Size([1, 361, 720])
            smooth_loss = SmoothLoss()
            loss= smooth_loss(decoded, GT)
            # mse_loss = nn.MSELoss(reduction='sum')
            # loss = mse_loss(decoded, GT) #GT是没问题的，会随着batchsize变

            # batch_size = GT.size(0)
            psnr_list = batch_psnr(decoded, GT)
            ssim_list = batch_ssim(decoded, GT)
            # mean_psnr = sum(psnr_list) / batch_size
            mean_psnr = psnr_list.mean()
            mean_ssim = ssim_list.mean()
            # logger.info(f"PSNR: {psnr_list} , Mean PSNR: {mean_psnr:.2f}, SSIM: {ssim_list}, Mean SSIM: {mean_ssim:.4f}")

            return loss, decoded, mean_psnr, psnr_list, mean_ssim, ssim_list