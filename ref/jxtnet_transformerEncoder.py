# from pathlib import Path
from functools import partial
from math import  pi, degrees
import math
import time
import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torchtyping import TensorType

from beartype import beartype
from beartype.typing import Tuple

from einops import rearrange, reduce, pack
from einops.layers.torch import Rearrange

from net.utils import transform_to_log_coordinates, batch_psnr, batch_ssim
import numpy as np

def total_variation(images):
    ndims = images.dim()
    if ndims == 3:
        # The input is a single image with shape [height, width, channels].
        # Calculate the difference of neighboring pixel-values.
        pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
        pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
        # Sum for all axis.
        tot_var = torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2))
    elif ndims == 4:
        # The input is a batch of images with shape: [batch, height, width, channels].
        # Calculate the difference of neighboring pixel-values.
        pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
        pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
        # Sum for the last 3 axes, resulting in a 1-D tensor with the total variation for each image.
        tot_var = torch.sum(torch.abs(pixel_dif1), dim=(1, 2, 3)) + torch.sum(torch.abs(pixel_dif2), dim=(1, 2, 3))
    else:
        raise ValueError("'images' must be either 3 or 4-dimensional.")
    return tot_var

class TVL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(TVL1Loss, self).__init__()
        # self.alpha = alpha
        self.beta = beta

    def forward(self, decoded, GT):
        # Calculate the MSE loss
        # L1_loss = nn.L1Loss(reduction='mean')
        L1_loss = nn.L1Loss(reduction='sum')
        loss_L1 = L1_loss(decoded, GT)
        tvloss= total_variation(decoded)
        # logger.info(f" tvloss:{tvloss*self.beta:.4f}, L1loss:{loss_L1:.4f}")

        total_loss = loss_L1 + tvloss * self.beta
        return total_loss

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


def exists(v):
    return v is not None

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

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
    reshaped_face_indices = face_indices.reshape(batch_size, -1).long()  # 做一次reshape，将face_indices变为1D张量，然后用它来索引点坐标张量 # 形状为 (b, nf*c)
    face_coords = torch.gather(vertices, 1, reshaped_face_indices.unsqueeze(-1).expand(-1, -1, vertices.shape[-1])) # 使用索引张量获取具有坐标的面
    face_coords = face_coords.reshape(batch_size, num_faces, num_vertices_per_face, -1)# 还原形状
    return face_coords

@torch.no_grad()
def get_derived_face_featuresjxt(
    face_coords: TensorType['b', 'nf', 'nvf', 3, float],  # 3 or 4 vertices with 3 coordinates输入坐标格式的face list
    in_em, #\theta, \phi, ka
    geoinfo,
    device,
    logger,
    self
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

    incident_angle_vec = polar_to_cartesian2(in_em[1],in_em[2]).to(device) #得到入射方向的xyz矢量
    incident_angle_mtx = incident_angle_vec.unsqueeze(1).expand(-1, area.shape[1], -1) #得到入射方向的矢量矩阵torch.Size([batchsize, 33564, 3])
    # incident_angle_mtx = incident_angle_vec.unsqueeze(1).repeat(1, area.shape[1], 1) #得到入射方向的矢量矩阵torch.Size([batchsize, 33564, 3])
    incident_freq_mtx = in_em[3].unsqueeze(1).unsqueeze(2).expand(-1, area.shape[1], -1) 
    # incident_freq_mtx = in_em[3].unsqueeze(1).unsqueeze(2).repeat(1, area.shape[1], 1) #得到入射波频率的矩阵torch.Size([1, 33564, 1]) 感觉取对数不是那个意思，对数坐标只是看起来的，不是实际上的？
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
        emfreq = incident_freq_mtx.to(device),
    ),incident_angle_vec #,mixfreqgeo#, incident_angle_vec #这里就算回了freq=em[0][2]好像也没啥用吧，没离散化的 入射方向矢量倒是有用！
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

def discretize3(
    t: Tensor,
    *,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo
    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5
    
    # 使用直通梯度法进行离散化
    t_discrete = t.round().long().clamp(min=0, max=num_discrete - 1)
    t_continuous = t_discrete.float() / (num_discrete - 1) * (hi - lo) + lo
    # 在反向传播时,保留梯度信息
    t_out = t_discrete.detach() + t_continuous - t_continuous.detach()
    return t_out

# tensor helper functions
class DiscretizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, continuous_range, num_discrete):
        lo, hi = continuous_range
        assert hi > lo
        input = (input - lo) / (hi - lo)
        input *= num_discrete
        input -= 0.5
        output = input.round().long().clamp(min=0, max=num_discrete - 1)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 0.5] = 0
        return grad_input, None, None
    
def discretize2(t: Tensor, *, continuous_range: Tuple[float, float], num_discrete: int = 128) -> Tensor:
    return DiscretizeSTE.apply(t, continuous_range, num_discrete)

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


# main classes
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
        middim = 64,
        device = 'cpu',
        hidden_size = 576,
        paddingsize = 22500
        
    ): #我草 这里面能调的参也太NM多了吧 这炼丹能练死人
        super().__init__()
        self.device = device
        self.paddingsize = paddingsize
        self.hidden_size = hidden_size
        #----------------------------------------------------jxt encoder----------------------------------------------------------
        self.discretize_face_coords = partial(discretize, num_discrete = num_discrete_coors, continuous_range = coor_continuous_range) #partial是用来把某个已经定义的函数固定一部分参数做成新的函数，这里就是针对face坐标，把descretize离散化函数定制成针对face坐标的离散化函数，方便后面调用discretize_face_coords可以少写几个参数减少出错且更简洁。
        #self居然也可以存函数 我还以为只能存数据或者实例
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed) #这里还只是实例化了，离散embedding数是num_discrete_coors = 128， 每个embedding的维度是dim_coor_embed = 64 后续会在encoder中使用

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
        self.emangle_embed = nn.Embedding(num_discrete_emangle, dim_emangle_embed) #jxt 128 64
        self.discretize_emfreq = partial(discretize, num_discrete = num_discrete_emfreq, continuous_range = (0.,1.0)) #2024年5月11日15:28:15我草 是不是没必要离散，这个情况，是不是其实我的freq本身其实就已经离散的了，不用我再人为离散化一次？只是embedding的时候他映射到embedding空间之后，隐含的空间关系就能实现我“连续回归”的目的？而且280个点离散到128个离散值，本身就有问题吧你妈的
        self.discretize_emfreq2 = partial(discretize2, num_discrete=num_discrete_emfreq, continuous_range=(0.,1.0))
        self.discretize_emfreq3 = partial(discretize3, num_discrete=num_discrete_emfreq, continuous_range=(0.,1.0))
        # self.discretize_emfreq2 = partial(gumbel_softmax, num_discrete=num_discrete_emfreq, continuous_range=(0.,1.0))

        self.emfreq_embed = nn.Embedding(num_discrete_emfreq, dim_emfreq_embed) #jxt
        # self.enfc0 = nn.Linear(4,22500,device=device) #为什么我的embedding层都能有梯度学出来，linear就不能学呢
        self.enmlp0 = nn.Sequential(
            nn.Linear(4, 4, bias=True,device=device),
            nn.SiLU(),
            nn.Linear(4, self.paddingsize, bias=True,device=device),
        )
        # self.enfc0.weight.data = self.enfc0.weight.data.to(torch.float64)
        # self.enfc0.bias.data = self.enfc0.bias.data.to(torch.float64)

        # initial dimension

        # project into model dimension
        # self.project_in = nn.Linear(1057, dim_codebook)
        self.project_in2 = nn.Linear(1057, hidden_size)

        # jxt transformer encoder
        self.transencoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=256),num_layers=6).to(device)
        #----------------------------------------------------jxt encoder----------------------------------------------------------


        #----------------------------------------------------jxt decoder----------------------------------------------------------
        # self.conv1d1 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=10, stride=10, dilation=1 ,padding=0)
        # self.fcneck = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size, bias=True,device=device),
        #     nn.SiLU(),
        #     nn.Linear(self.hidden_size, self.hidden_size, bias=True,device=device),
        # )
        self.conv1d1 = nn.Conv1d(785, 1, kernel_size=10, stride=10, dilation=1 ,padding=0)
        # self.conv1d1 = nn.Conv1d(784, 1, kernel_size=10, stride=10, dilation=1 ,padding=0)
        self.fc1d1 = nn.Linear(2250, middim*45*90)
        self.upconv1 = nn.ConvTranspose2d(middim, int(middim / 2), kernel_size=2, stride=2)
        self.in1 = nn.InstanceNorm2d(int(middim / 2))
        self.conv1_1 = nn.Conv2d(int(middim / 2), int(middim / 2), kernel_size=3, stride=1, padding=1)
        self.in1_1 = nn.InstanceNorm2d(int(middim / 2))
        self.conv1_2 = nn.Conv2d(int(middim / 2), int(middim / 2), kernel_size=3, stride=1, padding=1)
        self.in1_2 = nn.InstanceNorm2d(int(middim / 2))

        self.upconv2 = nn.ConvTranspose2d(int(middim / 2), int(middim / 4), kernel_size=2, stride=2)
        self.in2 = nn.InstanceNorm2d(int(middim / 4))
        self.conv2_1 = nn.Conv2d(int(middim / 4), int(middim / 4), kernel_size=3, stride=1, padding=1)
        self.in2_1 = nn.InstanceNorm2d(int(middim / 4))
        self.conv2_2 = nn.Conv2d(int(middim / 4), int(middim / 4), kernel_size=3, stride=1, padding=1)
        self.in2_2 = nn.InstanceNorm2d(int(middim / 4))

        self.upconv3 = nn.ConvTranspose2d(int(middim / 4), int(middim / 8), kernel_size=2, stride=2, output_padding=1)
        self.in3 = nn.InstanceNorm2d(int(middim / 8))
        self.conv3_1 = nn.Conv2d(int(middim / 8), int(middim / 8), kernel_size=3, stride=1, padding=1)
        self.in3_1 = nn.InstanceNorm2d(int(middim / 8))
        self.conv3_2 = nn.Conv2d(int(middim / 8), int(middim / 8), kernel_size=3, stride=1, padding=1)
        self.in3_2 = nn.InstanceNorm2d(int(middim / 8))
        self.conv1x1 = nn.Conv2d(int(middim / 8), 1, kernel_size=1, stride=1, padding=0)
        #----------------------------------------------------jxt decoder----------------------------------------------------------


    @beartype
    def encode( #Encoder
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],
        faces:            TensorType['b', 'nf', 'nvf', int],
        # face_mask,
        geoinfo,
        return_face_coordinates = False,
        in_em,
        logger
    ):
        device =vertices.device
        face_coords = jxtget_face_coords(vertices, faces) 
        pad_size = self.paddingsize - face_coords.size(1)
        face_coords = F.pad(face_coords, (0, 0, 0, 0, 0, pad_size)) #x.shape=torch.Size([2, 20804, 576])
        # face_mask = F.pad(face_mask, (0, pad_size)) #x.shape=torch.Size([2, 20804, 576])

        geoinfo = torch.Tensor(geoinfo).to(device).requires_grad_()

        in_obj = in_em[0]

        #----------------------------入射频率编码---------------------------------------------------
        in_emfreq = in_em[3].clone() #原始频率保存
        in_em[3]=transform_to_log_coordinates(in_em[3]).to(device) #频率转换为对数坐标 加在encoder里！
        lg_emfreq = in_em[3].clone() #对数频率保存
        in_em1 = in_em #[plane,theta,phi,ln(freq)]所有信息存在en_em1里
        '''
        [('b943', 'b943', 'bb7c', 'b7fd', 'bb7c', 'b7fd', 'b943', 'b979', 'b979', 'b943'),
         tensor([150, 180, 180,  90, 120,  30, 150,  90, 180,  90]), 
         tensor([ 60, 210,  30, 150,  30, 210,  60, 330, 150,  30]), 
         tensor([0.9142, 0.7557, 0.9015, 0.6822, 0.5867, 0.5681, 0.7617, 0.5124, 0.5747, 0.9309], device='cuda:0', dtype=torch.float64)]
        '''
        mixfreqgeo = torch.cat([geoinfo, in_em[3].unsqueeze(1)], dim=1).float() #对数频率加上几何信息
        incident_freq_mtx=self.enmlp0(mixfreqgeo) #加上几何信息的对数频率经过fc，理想中应该生成高端的归一化电尺寸
        # incident_freq_mtx=self.enfc0(mixfreqgeo) #加上几何信息的对数频率经过fc，理想中应该生成高端的归一化电尺寸
        # incident_freq_mtx=self.enkan0(mixfreqgeo)
        Ka_emfreq = incident_freq_mtx.clone() #归一化电尺寸保存
        # logger.info(f'物体{in_obj}，频率{in_emfreq}，对数化频率{lg_emfreq}，fc后归一化电尺度{kan_emfreq[0]}，sigmoid后{incident_freq_mtx[0]}')
        # logger.info(f'物体{in_obj}，频率{in_emfreq}，对数化频率{lg_emfreq}，fc后归一化电尺度{Ka_emfreq[0]}')
        # geomtx = (torch.Tensor(geoinfo).unsqueeze(1).expand(-1, area.shape[1], -1)).to(device)
        #----------------------------入射频率编码---------------------------------------------------

        #输出torch.Size([2, 20804, 3, 3])  tensor([-0.4463, -0.0323, -0.0037], device='cuda:0') 成功！
        # print(f'Encoder Step000 可删用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

#--------------------------------------------------------face预处理 得到特征--------------------------------------------------------------------------
        # compute derived features and embed
        # 先对内角、面积、法向量进行离散化和embedding
        derived_features, EMincvec= get_derived_face_featuresjxt(face_coords, in_em, geoinfo, device, logger, self) #这一步用了2s
        # print(f'Encoder Step1用时应该已经到头了，时间来自derive里：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

        in_em2 = [in_em1[0],EMincvec,in_em1[3]]

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
        # discrete_emfreq = self.discretize_emfreq(derived_features['emfreq']) #emfreq本来就是离散的 #jxt 2024年5月11日13:36:50我草是不是发现了一个bug，没用对离散法。
        discrete_emfreq = self.discretize_emfreq(incident_freq_mtx) #emfreq本来就是离散的 #jxt 2024年5月11日13:36:50我草是不是发现了一个bug，没用对离散法。
        discrete_emfreq_grad = incident_freq_mtx.clone()
        discrete_emfreq_grad[...] = discrete_emfreq
        # discrete_emfreq2 = self.discretize_emfreq2(incident_freq_mtx) #emfreq本来就是离散的 #jxt 2024年5月11日13:36:50我草是不是发现了一个bug，没用对离散法。

        emfreq_embed = self.emfreq_embed(discrete_emfreq_grad.long()) #好像是带梯度的 但是我忘了有没有搞定了

        discrete_face_coords = self.discretize_face_coords(face_coords) #先把face_coords离散化
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') # 9 or 12 coordinates per face #重新排布
        face_coor_embed = self.coor_embed(discrete_face_coords) #在这里把face做成embedding
        face_coor_embed = rearrange(face_coor_embed, 'b nf c d -> b nf (c d)') #再重新排布一下
        # print(f'Encoder Step3用时离散化也没法加速：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

        # face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed, emnoangle_embed, emangle_embed, emfreq_embed, derived_features['emfreq'], derived_features['geoinfo'], mixfreqgeo_embed], 'b nf *') #对于每个面，把所有embedding打包成一个embedding
        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed, emnoangle_embed, emangle_embed, emfreq_embed, incident_freq_mtx], 'b nf *') #对于每个面，把所有embedding打包成一个embedding
        # print(f'Encoder Step4用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        # em_embed, _ = pack([emangle_embed, emfreq_embed, derived_features['emfreq'], derived_features['geoinfo'], mixfreqgeo_embed], 'b nf *') #torch.Size([2, 20804, 3, 64]) , torch.Size([2, 20804, 1, 16]) 64*3+16=204
        em_embed, _ = pack([emangle_embed, emfreq_embed, incident_freq_mtx], 'b nf *') #torch.Size([2, 20804, 3, 64]) , torch.Size([2, 20804, 1, 16]) 64*3+16=204
        face_embed = self.project_in2(face_embed) #通过一个nn.linear线性层映射到codebook的维度 从1056到192
        # print(f'Encoder Step5用时fc映射也没法加速：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

        # next prepare the face_mask for using masked_select and masked_scatter 然后准备facemask
        orig_face_embed_shape = face_embed.shape[:2]
        # face_embed = face_embed[face_mask]
        face_embed = face_embed.reshape(-1, face_embed.shape[-1])
        # print(f'Encoder Step7 感觉可删用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

        face_embed = face_embed.reshape(-1,face_embed.shape[0],face_embed.shape[-1])#从(1,25000,576)变成(25000,1,576)
        face_embed = self.transencoder(face_embed)
        face_embed = face_embed.reshape(-1,face_embed.shape[0],face_embed.shape[-1])#从(25000,1,576)变回(1,25000,576)

        shape = (*orig_face_embed_shape, face_embed.shape[-1])# (1, 33564, 576) = (*torch.Size([1, 33564]), 576)
        # face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed) #多了一层[]而已
        # print(f'Encoder Step用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()        
        face_embed = face_embed.reshape(face_embed.shape[1],face_embed.shape[0],-1)
        # print(f'\nEncoder用时：{(time.time()-timeen):.4f}s')
        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords, em_embed, in_em2#, in_em_angle_vec

    @beartype
    def decode( #decoder输入：torchsize(1,33564,576) 长度33564待定，维度576固定
        self,
        x,
        em_embed,#torch.Size([2, 20804, 3, 64]) , torch.Size([2, 20804, 1, 16]) 64*3+16=204 +1=785
    ):
        x = torch.cat([x, em_embed], dim=2) #成了576+209=785维了
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
        x = F.silu(x)
        x = self.in1(x)

        x = self.conv1_1(x)
        x = F.silu(x)
        x = self.in1_1(x)

        x = self.conv1_2(x)
        x = F.silu(x)
        x = self.in1_2(x)

        x = self.upconv2(x)
        x = F.silu(x)
        x = self.in2(x)

        x = self.conv2_1(x)
        x = F.silu(x)
        x = self.in2_1(x)

        x = self.conv2_2(x)
        x = F.silu(x)
        x = self.in2_2(x)

        x = self.upconv3(x)
        x = F.silu(x)
        x = self.in3(x)

        x = self.conv3_1(x)
        x = F.silu(x)
        x = self.in3_1(x)

        x = self.conv3_2(x)
        x = F.silu(x)
        x = self.in3_2(x)

        x = self.conv1x1(x)
        # print(x.shape, x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])
        # x = self.huigui(x)

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
        geoinfo,
        in_em,
        GT,
        logger,
        device,
        lgrcs,
    ):
        ticc = time.time()
        # face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
#------------------------------------------------------------------进Encoder---------------------------------------------------------------------------------------------
        '''torch.Size([1, 33564, 3])'''
        encoded, __, em_embed, in_em2 = self.encode( #从这儿进encode里 返回的encoded就是那一个跑了一溜SAGEConv得到的face_embed.size = torch.Size([1, 33564, 576]), face_coordinates.shape = torch.Size([1, 33564, 9])是一个面3个点9个坐标点？为啥一个面是tensor([35, 60, 55, 35, 60, 55, 35, 60, 55]) 我知道了因为128^3离散化了
            vertices = vertices, #顶点
            faces = faces, #面
            # face_mask = face_mask, #面mask
            geoinfo = geoinfo,
            return_face_coordinates = True,
            in_em = in_em,
            logger = logger
        )
        # logger.info(f'Encoder用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

#------------------------------------------------------------------进Decoder---------------------------------------------------------------------------------------------
        decoded = self.decode( #从这儿进decoder里，进decoder的只有quantized，没有codes！所以是什么我也不用关心了其实，我只要把他大小对准塞进去就行。
            encoded, #quantized.shape = torch.Size([1, 33564, 576])
            # quantized, #quantized.shape = torch.Size([1, 33564, 576])
            em_embed,
            #in_em_angle_vec
            # in_em=in_em2,
            # diffusion = diffusionplugin
        )
        # logger.info(f'Decoder用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()
        '''torch.Size([1, 361, 720])'''
#------------------------------------------------------------------出Decoder了，后面都是算loss等后处理---------------------------------------------------------------------

        #平滑后处理：中值滤波+高斯滤波+修改后的smoothloss
        decoded = decoded.unsqueeze(1)  # 添加 channel 维度
        decoded = median_filter2d(decoded, kernel_size=5)# 应用中值滤波
        decoded = gaussian_filter2d(decoded, kernel_size=5, sigma=4, device=device)#两个都用 这个效果好
        decoded = decoded.squeeze(1)

        if GT == None:
            return decoded
        else:
            if lgrcs == True:
                epsilon = 0.001 #防止lg0的鲁棒机制
                # logger.info(f'初始GT:{GT[0]}')
                GT = torch.log10(torch.max(GT, torch.tensor(epsilon, device=GT.device))) #只要这里加一行把gt变成lg后的gt就行了。。其他甚至都完全不用改
                # logger.info(f'lg后GT:{GT[0]}')
                # logger.info(f'再变回去的GT:{torch.pow(10, GT)[0]}')
                # GT = torch.pow(10, GT) #反变换在这里

            TVL1loss = TVL1Loss(beta=0.1)
            loss = TVL1loss(decoded,GT)/(GT.shape[0])
            psnr_list = batch_psnr(decoded, GT)
            ssim_list = batch_ssim(decoded, GT)
            mean_psnr = psnr_list.mean()
            mean_ssim = ssim_list.mean()
            # mean_mse =  F.mse loss(decoded, GT, reduction='mean')
            # mseloss = nn.MSELoss(reduction='mean')
            with torch.no_grad():
                mean_mse = ((decoded-GT) ** 2).sum() / GT.numel()
            # logger.info(f"PSNR: {psnr_list} , Mean PSNR: {mean_psnr:.2f}, SSIM: {ssim_list}, Mean SSIM: {mean_ssim:.4f}")

            return loss, decoded, mean_psnr, psnr_list, mean_ssim, ssim_list, mean_mse