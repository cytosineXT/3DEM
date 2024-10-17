# from pathlib import Path
from functools import partial
from math import  pi, degrees
# import math
# import time
import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torchtyping import TensorType

from beartype import beartype
from beartype.typing import Tuple

from einops import rearrange, pack
from einops.layers.torch import Rearrange

from net.utils import transform_to_log_coordinates, psnr, ssim, toc, checksize
import numpy as np
from net.mytransformer import PositionalEncoding,TransformerWithPooling
from net.myswinunet import SwinTransformerSys
from pytictoc import TicToc
import trimesh
import glob
import os

t = TicToc()
t.tic()

def incidentangle_norm(x):
    return torch.stack([x[:,:,0]/180,x[:,:,1]/360]).squeeze().t().unsqueeze(1)

def total_variation(images):
    ndims = images.dim()
    if ndims == 3:
        # The input is a single image with shape [height, width, channels].
        # Calculate the difference of neighboring pixel-values.
        pixel_dif1 = images[:, :, 1:] - images[:, :, :-1]
        # pixel_dif1 = images[1:, :, :] - images[:-1, :, :] #改正了对batchsize做差分的错误。。。
        pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]
        # Sum for all axis.
        tot_var = torch.mean(torch.abs(pixel_dif1)) + torch.mean(torch.abs(pixel_dif2))
        # tot_var = torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2))
    elif ndims == 4:
        # The input is a batch of images with shape: [batch, height, width, channels].
        # Calculate the difference of neighboring pixel-values.
        pixel_dif1 = images[:, :, :, 1:] - images[:, :, :, :-1]
        pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
        # Sum for the last 3 axes, resulting in a 1-D tensor with the total variation for each image.
        tot_var = torch.mean(torch.abs(pixel_dif1), dim=(1, 2, 3)) + torch.mean(torch.abs(pixel_dif2), dim=(1, 2, 3))
        # tot_var = torch.sum(torch.abs(pixel_dif1), dim=(1, 2, 3)) + torch.sum(torch.abs(pixel_dif2), dim=(1, 2, 3))
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
        L1_loss = nn.L1Loss(reduction='mean')
        # L1_loss = nn.L1Loss(reduction='sum')
        loss_L1 = L1_loss(decoded, GT)
        tvloss= total_variation(decoded)
        # logger.info(f" tvloss:{tvloss*self.beta:.4f}, L1loss:{loss_L1:.4f}")
        total_loss = loss_L1 + tvloss * self.beta
        # print(f'l1loss:{loss_L1},tvloss:{tvloss},totalloss:{total_loss}')
        return total_loss

# Convert spherical coordinates to cartesian
def spherical_to_cartesian(theta, phi, device="cpu"):
    return torch.tensor([torch.sin(phi) * torch.cos(theta),
                         torch.sin(phi) * torch.sin(theta),
                         torch.cos(phi)], device=device, dtype=torch.float32)

# Calculate the projected area of a triangle for a given view vector
def calculate_projected_area(normal, area, view_vector):
    dot_product = torch.dot(normal, view_vector)
    return area * torch.abs(dot_product)

# Calculate angular similarity between two views
def angular_similarity(mesh, view1, view2, weight_area=0.5, weight_normals=0.5, device="cpu"):
    view_vector1 = spherical_to_cartesian(view1[0], view1[1], device=device)
    view_vector2 = spherical_to_cartesian(view2[0], view2[1], device=device)

    normals = torch.tensor(mesh.face_normals, dtype=torch.float32, device=device)
    areas = torch.tensor(mesh.area_faces, dtype=torch.float32, device=device)

    area_view1 = 0.
    area_view2 = 0.
    for face_idx in range(len(mesh.faces)):
        normal = normals[face_idx]
        area = areas[face_idx]
        projected_area1 = calculate_projected_area(normal, area, view_vector1)
        projected_area2 = calculate_projected_area(normal, area, view_vector2)
        if torch.dot(normal, view_vector1) < 0:
            area_view1 += projected_area1
        if torch.dot(normal, view_vector2) < 0:
            area_view2 += projected_area2

    projected_areas_difference = torch.abs(area_view1 - area_view2)
    similarity = 1 / (1 + 10 * projected_areas_difference.item())
    similarity = max(0, min(similarity, 1))
    return similarity, 1 - similarity

# Calculate frequency similarity (absolute difference)
def frequency_similarity(freq1, freq2):
    diff = torch.abs(freq1 - freq2)
    similarity = 1 - diff  # 直接使用差值的反转作为相似性
    return similarity, diff

# Custom contrastive loss for angles and frequencies

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1, obj_folder='planes/'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.obj_folder = obj_folder

    def load_mesh(self, plane_name):
        # Get the first 4 characters of the plane name
        plane_prefix = plane_name[:4]

        # Find the .obj file that matches the plane_prefix in the folder
        matching_files = glob.glob(os.path.join(self.obj_folder, f'{plane_prefix}*.obj'))
        if len(matching_files) == 0:
            raise FileNotFoundError(f"No matching .obj file found for {plane_name}")
        
        # Load the first matching file (assuming there's only one match)
        obj_file = matching_files[0]
        # print(f"Loading mesh for {plane_name} from {obj_file}")
        
        # Load the mesh using trimesh
        mesh = trimesh.load(obj_file)
        return mesh

    def forward(self, rcs, in_em,device, beta=0.1):
        batch_size = rcs.shape[0]  # 6 in your case

        # Split in_em into plane names, incident angles (2 values), and frequency (1 value)
        plane_names = in_em[0]  # shape: (batch_size,)
        angles = in_em[1:3]     # shape: (6, 2) - second and third columns are the incident angles
        freqs = in_em[3]        # shape: (6,) - fourth column is the frequency

        # Initialize the total loss and pair counter
        total_loss = 0.0
        pair_count = 0

        # Loop over all pairs of RCS matrices in the batch
        for i in range(batch_size):
            # Load the mesh for the current plane name
            # mesh_i = self.load_mesh(plane_names[i])

            for j in range(i + 1, batch_size):
                # Only compare samples from the same plane
                if plane_names[i] == plane_names[j]:
                    # Extract the RCS matrices for the ith and jth samples
                    rcs_i = rcs[i]  # shape: (360, 720)
                    rcs_j = rcs[j]  # shape: (360, 720)

                    # Compute the MSE loss between the two RCS matrices
                    rcs_loss = F.mse_loss(rcs_i, rcs_j)

                    # Load the mesh for the jth plane name
                    mesh_j = self.load_mesh(plane_names[j])

                    # Compute the angle similarity between the ith and jth samples
                    view1 = (np.radians(angles[0][i]), np.radians(angles[1][i]))
                    view2 = (np.radians(angles[0][j]), np.radians(angles[1][j]))
                    
                    # Use the mesh from the ith plane (since both planes are the same here)
                    angle_sim, angle_dif = angular_similarity(mesh_j, view1, view2, weight_area=0.5, weight_normals=0., device=device)

                    # Compute the frequency similarity between the ith and jth samples
                    freq_sim, freq_dif = frequency_similarity(freqs[i], freqs[j])

                    # Calculate angle and frequency losses
                    angle_loss = rcs_loss * angle_sim - max(0, rcs_loss - self.margin) * angle_dif
                    freq_loss = rcs_loss * freq_sim - max(0, rcs_loss - self.margin) * freq_dif

                    # Combine the two losses
                    pairwise_loss = beta * angle_loss + freq_loss

                    # Accumulate the pairwise loss
                    total_loss += pairwise_loss

                    # Increment the pair counter since we found a matching pair
                    pair_count += 1

        # Return the total contrastive loss, averaged by the number of valid pairs
        if pair_count > 0:
            return total_loss / pair_count
        else:
            # If no pairs were found, return zero loss
            return torch.tensor(0.0, device=rcs.device)

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
def coords_interanglejxt2(x, y, eps=1e-5): #不要用爱恩斯坦求和 会变得不幸
    edge_vector = x - y
    normv = l2norm(edge_vector) #torch.Size([2, 20804, 3, 3])
    normdot = -(normv * torch.cat((normv[..., -1:], normv[..., :-1]), dim=3)).sum(dim=2) #应为torch.Size([2, 20804])
    normdot = torch.clamp(normdot, -1 + eps, 1 - eps)
    radians = torch.acos(normdot) #tensor([1.1088, 0.8747, 1.1511], device='cuda:0')
    angle = torch.rad2deg(radians) #tensor([63.5302, 50.1188, 65.9518], device='cuda:0')
    return radians, angle

def vector_anglejxt2(x, y, eps=1e-5):
    normdot = -(l2norm(x) * l2norm(y)).sum(dim=-1)
    normdot = torch.clamp(normdot, -1 + eps, 1 - eps)
    radians = normdot.acos() #tensor(1.4104, device='cuda:0')
    angle = torch.rad2deg(radians) #tensor(80.8117, device='cuda:0')
    return radians, angle

def polar_to_cartesian2(theta, phi):
    theta_rad = torch.deg2rad(theta)
    phi_rad = torch.deg2rad(phi)
    x = torch.sin(phi_rad) * torch.cos(theta_rad)
    y = torch.sin(phi_rad) * torch.sin(theta_rad)
    z = torch.cos(phi_rad)
    return torch.stack([x, y, z], dim=1)

@torch.no_grad()
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
    shifted_face_coords = torch.cat((face_coords[:, :, -1:], face_coords[:, :, :-1]), dim = 2).to(device) #这是对face_coords循环移位，face_coords[:, :, -1:]取最后一个切片，face_coords[:, :, :-1]取最后一个之前的切片，然后连接在一起。
    angles, _  = coords_interanglejxt2(face_coords, shifted_face_coords) #得到了每个三角形face的三个内角，弧度形式的，如果要角度形式的要用_的(angle2) 耗时1.64！！
    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim = 2) #这里是坐标相减得到边
    normals = l2norm(torch.cross(edge1, edge2, dim = -1)) #然后也用边叉乘得到法向量，很合理
    area = torch.cross(edge1, edge2, dim = -1).norm(dim = -1, keepdim = True) * 0.5 #两边矢量叉乘模/2得到面积
    incident_angle_vec = polar_to_cartesian2(in_em[1],in_em[2]).to(device) #得到入射方向的xyz矢量
    incident_angle_mtx = incident_angle_vec.unsqueeze(1).expand(-1, area.shape[1], -1) #得到入射方向的矢量矩阵torch.Size([batchsize, 33564, 3])
    incident_freq_mtx = in_em[3].unsqueeze(1).unsqueeze(2).expand(-1, area.shape[1], -1) 
    incident_mesh_anglehudu, _ = vector_anglejxt2(normals, incident_angle_mtx) #得到入射方向和每个mesh法向的夹角,是在0到180度的，0-90说明面在物体屁股后面，90-180说明是正面  耗时0.68！！！

    return dict(
        angles = angles,
        area = area,
        normals = normals,
        emnoangle = incident_mesh_anglehudu,
        emangle = incident_angle_mtx,
        emfreq = incident_freq_mtx.to(device),
    ),incident_angle_vec #,mixfreqgeo#, incident_angle_vec #这里就算回了freq=em[0][2]好像也没啥用吧，没离散化的 入射方向矢量倒是有用！


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
class MeshEncoderDecoder(Module):
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
        decoder_outdim = 12,
        device = 'cpu',
        hidden_size = 576,
        paddingsize = 22500,
        encoder_layer = 6,
        alpha = 0.1,
        
    ): #我草 这里面能调的参也太NM多了吧 这炼丹能练死人
        super().__init__()
        t.toc('  初始化开始',restart=True)
        self.alpha = alpha
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

        #-----------------------------------------------------------------------------------------------频率专题-------------------------------------------------------------------
        #感觉还是要离散化一下，现在直接用mlp做嵌入学不到东西。2024年9月21日20:02:05 
        self.discretize_emfreq = partial(discretize, num_discrete = num_discrete_emfreq, continuous_range = (0.,1.0)) #2024年5月11日15:28:15我草 是不是没必要离散，这个情况，是不是其实我的freq本身其实就已经离散的了，不用我再人为离散化一次？只是embedding的时候他映射到embedding空间之后，隐含的空间关系就能实现我“连续回归”的目的？而且280个点离散到128个离散值，本身就有问题吧你妈的
        self.emfreq_embed = nn.Embedding(num_discrete_emfreq, dim_emfreq_embed) #jxt
        self.emfreq_embed1 = nn.Embedding(num_discrete_emfreq, int(self.paddingsize/(2**encoder_layer))) #jxt
        self.emfreq_embed2 = nn.Embedding(num_discrete_emfreq, decoder_outdim*8*45*90) #jxt

        #-----------------------------------------------------------------------------------------------频率专题-------------------------------------------------------------------


        # self.enfc0 = nn.Linear(4,22500,device=device) #为什么我的embedding层都能有梯度学出来，linear就不能学呢
        # self.enmlp0 = nn.Sequential(
        #     nn.Linear(4, 4, bias=True,device=device),
        #     nn.SiLU(),
        #     nn.Linear(4, self.paddingsize, bias=True,device=device),
        # )
        # self.enmlp00 = nn.Sequential(
        #     nn.Linear(1, 4, bias=True,device=device),
        #     nn.SiLU(),
        #     nn.Linear(4, self.paddingsize, bias=True,device=device),
        # )
        # self.enfc0.weight.data = self.enfc0.weight.data.to(torch.float64)
        # self.enfc0.bias.data = self.enfc0.bias.data.to(torch.float64)

        # initial dimension

        # project into model dimension
        # self.project_in = nn.Linear(1057, dim_codebook)
        self.project_in2 = nn.Linear(1057, hidden_size)

        # jxt transformer encoder
        # self.transencoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=256),num_layers=6).to(device)
        #----------------------------------------------------jxt encoder----------------------------------------------------------
        self.transformer_model = TransformerWithPooling(d_model=hidden_size, nhead=4, dim_feedforward=256, num_layers=encoder_layer, pool_size=2, activation='silu').to(device)
        self.pe = PositionalEncoding(d_model=hidden_size).to(device)
        self.conv1d1 = nn.Conv1d(576, 1, kernel_size=1, stride=1, dilation=1 ,padding=0).to(device)
        self.fc1d1 = nn.Sequential(
                nn.Linear(int(self.paddingsize/(2**encoder_layer)), int(self.paddingsize/(2**encoder_layer))),
                nn.SiLU(),
                nn.Linear(int(self.paddingsize/(2**encoder_layer)), decoder_outdim*8*45*90),#388800
                nn.LayerNorm(decoder_outdim*8*45*90)).to(device)
        self.swinunet = SwinTransformerSys(embed_dim=decoder_outdim,window_size=9).to(device) #我给的是这个 其他都是自己算出来的
        self.incident_angle_linear1 = nn.Linear(2, int(self.paddingsize/(2**encoder_layer)))
        # self.sig1 = nn.Sigmoid()
        # self.sig2 = nn.Sigmoid()
        # self.incident_freq_linear1 = nn.Linear(1, int(self.paddingsize/(2**encoder_layer)))
        # self.incident_freq_linear1 = nn.Sequential(
        #         nn.Linear(1, 8),
        #         nn.SiLU(),
        #         nn.Linear(8,int(self.paddingsize/(2**encoder_layer)))).to(device)
        self.incident_angle_linear2 = nn.Linear(2, decoder_outdim*8*45*90)
        # # self.incident_freq_linear2 = nn.Linear(1, decoder_outdim*8*45*90)
        # self.incident_freq_linear2 = nn.Sequential(
        #         nn.Linear(1, 8),
        #         nn.SiLU(),
        #         nn.Linear(8,decoder_outdim*8*45*90)).to(device)
        t.toc('  初始化结束',restart=True)


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
        # ------------------------------带几何信息的-------------------------------------------------------
        # mixfreqgeo = torch.cat([geoinfo, in_em[3].unsqueeze(1)], dim=1).float() #对数频率加上几何信息
        # incident_freq_mtx=self.enmlp0(mixfreqgeo) #加上几何信息的对数频率经过fc，理想中应该生成高端的归一化电尺寸
        # ------------------------------带几何信息的-------------------------------------------------------

        incident_freq_mtx=lg_emfreq.unsqueeze(1).repeat(1,self.paddingsize).float() #不加几何信息的对数频率经过fc
        
        # incident_freq_mtx=self.enfc0(mixfreqgeo) #加上几何信息的对数频率经过fc，理想中应该生成高端的归一化电尺寸
        # incident_freq_mtx=self.enkan0(mixfreqgeo)
        # Ka_emfreq = incident_freq_mtx.clone() #归一化电尺寸保存
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

        # in_em2 = [in_em1[0],EMincvec,in_em1[3]]

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

        #-----------------------------------------------------------------------------------------------频率专题-------------------------------------------------------------------
        # 2024年9月26日16:47:22 感觉这里真的有点问题，前面用mlp0把频率和几何尺寸（长4）变成了归一化电尺寸（18000）的频率矩阵 然后这里又把经过mlp0的变量塞进了离散嵌入，得到了emfreq_embed，感觉有点夸张，首先离散方法就错了，不是0-1了，然后这样是不是也有问题了。
        # ------------------------------带几何信息的-------------------------------------------------------
        # # discrete_emfreq = self.discretize_emfreq(derived_features['emfreq']) #emfreq本来就是离散的 #jxt 2024年5月11日13:36:50我草是不是发现了一个bug，没用对离散法。
        # discrete_emfreq = self.discretize_emfreq(incident_freq_mtx) #emfreq本来就是离散的 #jxt 2024年5月11日13:36:50我草是不是发现了一个bug，没用对离散法。
        # discrete_emfreq_grad = incident_freq_mtx.clone()
        # discrete_emfreq_grad[...] = discrete_emfreq
        # # discrete_emfreq2 = self.discretize_emfreq2(incident_freq_mtx) #emfreq本来就是离散的 #jxt 2024年5月11日13:36:50我草是不是发现了一个bug，没用对离散法。
        # # #在做face预处理的时候用了一次freq的freq_embed，得到的是多少来着我忘了，但是在后面decoder里用的又不是这个。
        # emfreq_embed = self.emfreq_embed(discrete_emfreq_grad.long()) #好像是带梯度的 但是我忘了有没有搞定了 torch.Size([6, 18000, 16])
        # ------------------------------带几何信息的-------------------------------------------------------
        #下面是不带几何信息的
        discrete_emfreq = self.discretize_emfreq(incident_freq_mtx) 
        emfreq_embed = self.emfreq_embed(discrete_emfreq).float()
        #-----------------------------------------------------------------------------------------------频率专题-------------------------------------------------------------------



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
        # em_embed, _ = pack([emangle_embed, emfreq_embed, incident_freq_mtx], 'b nf *') #torch.Size([2, 20804, 3, 64]) , torch.Size([2, 20804, 1, 16]) 64*3+16=204
        # face_embed = self.project_in2(face_embed) #通过一个nn.linear线性层映射到codebook的维度 从1056到192
        # print(f'Encoder Step5用时fc映射也没法加速：{(time.time()-ticc):.4f}s')
        # ticc = time.time()

        face_embed = self.project_in2(face_embed) #通过一个nn.linear线性层映射到codebook的维度 从1056到192
        checksize(face_embed)
        # orig_face_embed_shape = face_embed.shape[:2]

        '''
        Transformer输入：   (seq_len, batch_size, C)    (L B C)
        1DConv输入：        (batch_size, C, seq_len)
        2DConv输入：        (batch_size, C, H, W)
        Linear输入：        仅对最后一个维度从输入变成输出
        '''
        # face_embed=torch.randn(6, 22500, 576).to(device)
        # discrete_face_coords, em_embed, in_em1 = 0,0,in_em.to(device) #居然和前处理也没关系。。。

        face_embed = face_embed.reshape(-1,face_embed.shape[0],face_embed.shape[-1])#从(1,25000,576)变成(25000,1,576)
        '''(L B C)'''
        checksize(face_embed)
        face_embed = self.pe(face_embed)
        face_embed = self.transformer_model(face_embed)
        face_embed = face_embed.reshape(-1,face_embed.shape[0],face_embed.shape[-1])#从(25000,1,576)变回(1,25000,576)

        # shape = (*orig_face_embed_shape, face_embed.shape[-1])# (1, 33564, 576) = (*torch.Size([1, 33564]), 576)
        # face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed) #多了一层[]而已
        # print(f'Encoder Step用时：{(time.time()-ticc):.4f}s')
        # ticc = time.time()        
        face_embed = face_embed.reshape(face_embed.shape[1],face_embed.shape[0],-1)
        # print(f'\nEncoder用时：{(time.time()-timeen):.4f}s')
        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords, in_em1, lg_emfreq#, in_em_angle_vec

    @beartype
    def decode( #decoder输入：torchsize(1,33564,576) 长度33564待定，维度576固定
        self,
        x, #torch.Size([6, 576, 281])
        in_em1,
        device,
        lg_emfreq
    ):
        in_angle = torch.stack([in_em1[1]/180, in_em1[2]/360]).t().float().to(device).unsqueeze(1)#我草 我直接在这儿除不就好了 我是呆呆比 还写了个incidentangle_norm()

        #-----------------------------------------------------------------------------------------------频率专题-------------------------------------------------------------------
        # in_freq = in_em1[3].t().float().unsqueeze(1).unsqueeze(1).to(device) #这里是又得到了，然后用的mlp做嵌入 但是应该用离散embed做嵌入，能不能把之前的拿过来，得到的是什么样子的变量
        '''tensor([[[0.7492]],  [[0.8482]],  [[0.9227]],   [[0.9204]],   [[0.9010]],   [[0.7291]]], device='cuda:0')'''

        condangle1 = self.incident_angle_linear1(in_angle)
        condangle2 = self.incident_angle_linear2(in_angle)
        # condangle1 = self.sig1(self.incident_angle_linear1(in_angle)) #为了避免值太大干扰主变量？
        # condangle2 = self.sig2(self.incident_angle_linear2(in_angle))
        # condfreq1 = self.incident_freq_linear1(in_freq)  #缩放因子，加强频率的影响，因为现在看来频率没啥影响，网络还没学到根据频率而变..不大行 发现是应该要归一化
        # condfreq2 = self.incident_freq_linear2(in_freq)

        discretized_freq = self.discretize_emfreq(lg_emfreq) 
        condfreq1 = self.emfreq_embed1(discretized_freq).unsqueeze(1)
        condfreq2 = self.emfreq_embed2(discretized_freq).unsqueeze(1)
        #-----------------------------------------------------------------------------------------------频率专题-------------------------------------------------------------------
        
        #---------------conv1d+fc bottleneck---------------
        x = x.reshape(x.shape[1], x.shape[2], -1)  # 1DConv输入：Reshape to (batch_size, input_channel, seq_len)
        checksize(x)
        x = self.conv1d1(x) #571变1
        checksize(x)
        x = x + condangle1 #torch.Size([6, 1, 281])
        x = x + condfreq1

        x = self.fc1d1(x)
        checksize(x)
        x = x + condangle2
        x = x + condfreq2 #试图放大freq的影响 要是还不够，就在decoder Transformer里每一层都加上freq因素，就像原始的带skip connection的Unet一样

        #-------------SwinTransformer Decoder--------------
        x = x.reshape(x.shape[0],45*90,-1)
        checksize(x)
        x = self.swinunet(x,discretized_freq)
        checksize(x)
        return x.squeeze(dim=1)

    @beartype
    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, float],
        faces:          TensorType['b', 'nf', 'nvf', int],
        geoinfo,
        in_em,
        GT,
        logger=None,
        device='cpu',
        lgrcs=False,
    ):
        # t.toc('  刚进来',restart=True)
        
        # tic = time.time()
#------------------------------------------------------------------进Encoder---------------------------------------------------------------------------------------------
        # print('\n')
        checksize(faces)
        encoded, __, in_em1, lg_emfreq = self.encode( #从这儿进encode里 返回的encoded就是那一个跑了一溜SAGEConv得到的face_embed.size = torch.Size([1, 33564, 576]), face_coordinates.shape = torch.Size([1, 33564, 9])是一个面3个点9个坐标点？为啥一个面是tensor([35, 60, 55, 35, 60, 55, 35, 60, 55]) 我知道了因为128^3离散化了
            vertices = vertices, #顶点
            faces = faces, #面
            geoinfo = geoinfo,
            return_face_coordinates = True,
            in_em = in_em,
            logger = logger
        )
        # encoded = torch.randn(21, 6, 576).to(device) #破案了 原来真的是encoder的问题，那就是token太长了的原因 他的前几层贡献的
        # in_em1 = in_em

        # t.toc('  encoder',restart=True)
        # print('\nencoder:')
        # tic = toc(tic) #耗时0.0145s
#------------------------------------------------------------------进Decoder---------------------------------------------------------------------------------------------
        decoded = self.decode( #从这儿进decoder里，进decoder的只有quantized，没有codes！所以是什么我也不用关心了其实，我只要把他大小对准塞进去就行。
            encoded, #quantized.shape = torch.Size([1, 33564, 576])
            in_em1,
            device,
            lg_emfreq
        )
        # t.toc('  decoder',restart=True)
        # print('decoder:')
        # tic = toc(tic) #耗时0.0109s
#------------------------------------------------------------------出Decoder了，后面都是算loss等后处理---------------------------------------------------------------------
        #平滑后处理：中值滤波+高斯滤波+修改后的smoothloss
        decoded = decoded.unsqueeze(1)  # 添加 channel 维度
        decoded = median_filter2d(decoded, kernel_size=5)# 应用中值滤波
        # decoded = median_filter2d(decoded, kernel_size=5)# 应用中值滤波
        # print('中值滤波')
        # tic = toc(tic)
        decoded = gaussian_filter2d(decoded, kernel_size=5, sigma=4, device=device)#两个都用 这个效果好
        decoded = decoded.squeeze(1)
        # print('高斯滤波:')
        # tic = toc(tic)

        if GT == None:
            # tic = toc(tic)
            return decoded
        else:
            GT = GT[:,:-1,:] #361*720变360*720
            #------------------------------------------------------------------------
            if lgrcs == True:
                epsilon = 0.001 #防止lg0的鲁棒机制
                # logger.info(f'初始GT:{GT[0]}')
                GT = torch.log10(torch.max(GT, torch.tensor(epsilon, device=GT.device))) #只要这里加一行把gt变成lg后的gt就行了。。其他甚至都完全不用改
                # logger.info(f'lg后GT:{GT[0]}')
                # logger.info(f'再变回去的GT:{torch.pow(10, GT)[0]}')
                # GT = torch.pow(10, GT) #反变换在这里
            #------------------------------------------------------------------------
            TVL1loss = TVL1Loss(beta=1.0) #我草 发现一个错误  pixel_dif1 = images[1:, :, :] - images[:-1, :, :] 但是GT.shape = torch.Size([1, 360, 720])，第一项是batchsize。。。
            loss = TVL1loss(decoded,GT)/(GT.shape[0]) #平均了batch的loss
            # print('TVL1loss:')
            # tic = toc(tic)

            conloss = ContrastiveLoss(margin=0.15)
            loss2 = conloss(decoded, in_em, device=device)
            # total_loss = loss
            total_loss = loss + self.alpha * loss2 #alpha是权重
            # logger.info(f'L1loss={loss:.4f}, Contrastiveloss={loss2:.4f}, alpha={self.alpha}, total_loss={total_loss:.4f}')
            with torch.no_grad():
                psnr_list = psnr(decoded, GT)
                # print('psnrssim:')
                # tic = toc(tic)
                ssim_list = ssim(decoded, GT)
                # tic = toc(tic)
                mean_psnr = psnr_list.mean()
                # tic = toc(tic)
                mean_ssim = ssim_list.mean()
                # tic = toc(tic)
                # mean_psnr, psnr_list, mean_ssim, ssim_list = 0, [], 0, []

            # mean_mse =  F.mse loss(decoded, GT, reduction='mean')
            # mseloss = nn.MSELoss(reduction='mean')
            # with torch.no_grad():
                mean_mse = ((decoded-GT) ** 2).sum() / GT.numel()
            # logger.info(f"PSNR: {psnr_list} , Mean PSNR: {mean_psnr:.2f}, SSIM: {ssim_list}, Mean SSIM: {mean_ssim:.4f}")
            # print('mean_mse:')
            # tic = toc(tic) #耗时2.0270s
            # t.toc('  后处理',restart=True)
            # return loss, decoded, mean_psnr, psnr_list, mean_ssim, ssim_list, mean_mse
            return total_loss, decoded, mean_psnr, psnr_list, mean_ssim, ssim_list, mean_mse