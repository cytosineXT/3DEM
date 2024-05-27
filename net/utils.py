from pathlib import Path
import os
import torch.utils.data.dataset as Dataset
import torch
import torch.nn.functional as F
import logging
import numpy as np
import trimesh

def ssim(img1, img2, window_size=11, size_average=True):
    img1 = img1.unsqueeze(1) # [batch_size, 1, height, width]
    img2 = img2.unsqueeze(1) # [batch_size, 1, height, width]
    gtmax = torch.max(img1)
    demax = torch.max(img2) #1.7787
    maxx = torch.max(gtmax, demax)
    C1 = (0.01 * maxx) ** 2
    C2 = (0.03 * maxx) ** 2

    def gaussian(window_size, sigma):
        gauss = torch.exp(-(torch.arange(window_size) - window_size // 2) ** 2 / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    channel = img1.size(0)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def batch_ssim(img1, img2):
    batch_size = img1.size(0)
    ssim_list = []
    for i in range(batch_size):
        ssim_val = ssim(img1[i].unsqueeze(0), img2[i].unsqueeze(0))
        ssim_list.append(ssim_val)
    return torch.tensor(ssim_list)

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2, reduction='mean')
    gtmax = torch.max(img1)
    demax = torch.max(img2)
    maxx = torch.max(gtmax, demax)
    psnr = 10 * torch.log10(maxx * maxx / mse)
    return psnr
    
def batch_psnr(img1, img2):
    batch_size = img1.size(0)
    psnrlist=[]
    for i in range(batch_size):
        psnrr=psnr(img1[i],img2[i])
        psnrlist.append(psnrr)
    return torch.tensor(psnrlist)

def transform_to_log_coordinates(x):
    log_x = (torch.log10(torch.where(x <= 0, torch.finfo(torch.float).eps, x)) - torch.log10(torch.tensor(0.001))) / (torch.log10(torch.tensor(8)) - torch.log10(torch.tensor(0.001))) # 将负值和零转换为正值,取对数,归一化处理
    return log_x

def get_model_memory(model,logger):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    memory = params * 4 / (1024**3)
    logger.info(f'模型占用{memory:.4f}GB')  # 以GB为单位
    # print(f'模型占用{memory:.4f}GB')  # 以GB为单位
    return memory

def get_tensor_memory(x,logger):
    memory = x.element_size() * x.nelement() / (1024**3)
    logger.info(f'Tensor变量占用{memory:.4f}GB')  # 以GB为单位
    # print(f'Tensor变量占用{memory:.4f}GB')  # 以GB为单位
    return memory

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def increment_path(path, exist_ok=False, sep="", mkdir=True):
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

class meshRCSDataset(Dataset.Dataset):
    def __init__(self, Mesh, RCSmap):    #初始化，定义数据内容和标签
        self.meshdata = Mesh
        self.RCSmap = RCSmap
    def __len__(self):    #返回数据集大小
        return len(self.meshdata)
    def __getitem__(self, index):    #得到数据内容和标签
        # meshdata = torch.Tensor(self.meshdata[index])
        meshdata = self.meshdata[index]
        RCSmap = torch.Tensor(self.RCSmap[index])
        return meshdata, RCSmap
    
def find_matching_files(prefixlist, directory):
    objlist = []
    ptlist = []
    for prefix in prefixlist:
        for filename in os.listdir(directory):
            if filename[:4]== prefix:
                plane = filename[:-4]
                loadobj = plane + '.obj'
                loadpt = plane + '.pt'
                objlist.append(os.path.join(directory, loadobj))
                ptlist.append(os.path.join(directory, loadpt))
                break
    return objlist , ptlist


def load_and_process(file_path):
    """
    读取并处理单个文件，返回faces, verts, faceedge张量。
    """
    mesh = trimesh.load_mesh(file_path)
    area = mesh.area
    volume = mesh.volume
    scale = mesh.scale
    faces = torch.tensor(mesh.faces, dtype=torch.int).unsqueeze(0)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0)
    pt_path = file_path.replace('.obj', '.pt')
    faceedge = torch.load(pt_path)
    return faces, verts, faceedge, [area, volume, scale]

def process_files(file_paths, device='cpu'):
    """
    处理一组文件并返回批量张量。
    
    Args:
    - file_paths (list of str): 文件路径列表。
    - device (str or torch.device): 使用的设备，默认为'cpu'。
    
    Returns:
    - planesur_faces (torch.Tensor): faces批量张量。
    - planesur_verts (torch.Tensor): verts批量张量。
    - planesur_faceedges (torch.Tensor): faceedges批量张量。
    """
    # 设置设备
    device = torch.device(device if isinstance(device, str) else device)

    # 读取所有文件并合并到batch
    faces_list, verts_list, faceedges_list, geoinfo_list = [], [], [], []
    for file_path in file_paths:
        faces, verts, faceedge, geoinfo = load_and_process(file_path)
        faces_list.append(faces)
        verts_list.append(verts)
        faceedges_list.append(faceedge)
        geoinfo_list.append(geoinfo)

    # 找到需要填充到的最大大小
    max_faces_size = max(f.shape[1] for f in faces_list)
    max_verts_size = max(v.shape[1] for v in verts_list)
    max_faceedges_size = max(fe.shape[1] for fe in faceedges_list)

    # 填充所有张量到相同大小
    padded_faces_list = [F.pad(f, (0, 0, 0, max_faces_size - f.shape[1])) for f in faces_list]
    padded_verts_list = [F.pad(v, (0, 0, 0, max_verts_size - v.shape[1])) for v in verts_list]
    padded_faceedges_list = [F.pad(fe, (0, 0, 0, max_faceedges_size - fe.shape[1])) for fe in faceedges_list]

    # 合并到batch
    planesur_faces = torch.cat(padded_faces_list, dim=0).to(device)
    planesur_verts = torch.cat(padded_verts_list, dim=0).to(device)
    planesur_faceedges = torch.cat(padded_faceedges_list, dim=0).to(device)

    return planesur_faces, planesur_verts, planesur_faceedges, geoinfo_list