import torch
import time
import time
from net.jxtnet_GNN import MeshEncoderDecoder
# from net.jxtnet_Transupconvflop import MeshEncoderDecoder
import torch.utils.data.dataloader as DataLoader
import os
import re
from net.utils import  meshRCSDataset, find_matching_files, process_files
from pytictoc import TicToc
t = TicToc()
t.tic()
import random
import numpy as np
import argparse
from thop import profile

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
    #  torch.backends.cudnn.benchmark = False  # 关闭优化搜索
     torch.backends.cudnn.deterministic = True
     np.random.seed(seed)
     random.seed(seed)
# 设置随机数种子
def parse_args():
    parser = argparse.ArgumentParser(description="Script with customizable parameters using argparse.")
    parser.add_argument('--epoch', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--use_preweight', type=bool, default=False, help='Whether to use pretrained weights')
    parser.add_argument('--draw', type=bool, default=True, help='Whether to enable drawing')

    parser.add_argument('--trainname', type=str, default='mul2347', help='logname')
    parser.add_argument('--rcsdir', type=str, default='/home/ljm/workspace/datasets/traintest', help='Path to rcs directory')

    parser.add_argument('--seed', type=int, default=777, help='Random seed for reproducibility')
    parser.add_argument('--gama', type=float, default=0.001, help='Loss threshold or gamma parameter')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='CUDA device to use')
    return parser.parse_args()

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model(
            vertices=inputs[0],
            faces=inputs[1],
            face_edges = inputs[2],
            geoinfo=inputs[3],
            in_em=inputs[4],
            GT=inputs[5]
        )

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
tic0 = time.time()
tic = time.time()
print('代码开始时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))  

args = parse_args()

# 使用命令行参数
epoch = args.epoch
use_preweight = args.use_preweight
draw = args.draw
rcsdir = args.rcsdir
seed = args.seed
gama = args.gama
cudadevice = args.cuda
name = args.trainname

in_ems = []
rcss = []

for file in os.listdir(rcsdir):
    if '.pt' in file:
        # print(file)
        plane, theta, phi, freq= re.search(r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d.+).pt", file).groups()
        theta = int(theta)
        phi = int(phi)
        freq = float(freq)
        in_em = [plane,theta,phi,freq]
        # print(in_em)
        try:
            rcs = torch.load(os.path.join(rcsdir,file))
        except Exception as e:
            1
        in_ems.append(in_em)
        rcss.append(rcs)

dataset = meshRCSDataset(in_ems, rcss)
dataloader = DataLoader.DataLoader(dataset, batch_size=1, num_workers=0) #创建DataLoader迭代器
# device = torch.device(cudadevice if torch.cuda.is_available() else "cpu")
device = 'cpu'

autoencoder = MeshEncoderDecoder( #这里实例化，是进去跑了init 草 但是这里还是用的paddingsize
    num_discrete_coors = 128,
    device= 'cpu',
    # paddingsize = 18000,
    paddingsize = 22500,
    decoder_outdim = 12, #决定了decoder的size 12L 6M 3S
    encoder_layer = 6, #决定了encoder的层数
).to(device)

flag = 1
GTflag = 1
for i in range(epoch):
    psnr_list = []
    ssim_list = []
    mse_list = []
    jj=0
    epoch_loss = 0.
    timeepoch = time.time()

    for in_em1,rcs1 in dataloader:
        jj=jj+1
        in_em0 = in_em1.copy()
        # optimizer.zero_grad()
        objlist , ptlist = find_matching_files(in_em1[0], "./planes")
        planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device) #为了解决多batch变量长度不一样的问题 在这一步就已经padding到等长了
        wrapped_model = WrappedModel(autoencoder)
        flops, params = profile(wrapped_model, (planesur_verts, planesur_faces, planesur_faceedges, geoinfo, in_em1, rcs1.to(device)))

        print('flops: ', flops, 'params: ', params)
        print('Gflops: %.2f , params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
        break
    

