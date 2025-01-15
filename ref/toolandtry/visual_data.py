'''I solemnly swear that I am up to no good'''

import torch
import time
from tqdm import tqdm
import os
import re
import matplotlib
matplotlib.use('agg')
from NNvalfast import plotRCS2, plot2DRCS

tic0 = time.time()
tic = time.time()
print('代码开始时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))  
cudadevice = 'cuda:0'
in_ems = []
rcss = []
corrupted_files = []

# rcsdir = r'/home/jiangxiaotian/datasets/traintest' #T7920 Liang
# rcsdir = r'/home/jiangxiaotian/datasets/mul2347_pretrain' #T7920 Liang
# rcsdir = r'/home/jiangxiaotian/datasets/mul2347_train' #T7920 Liang
# rcsdir = r'/home/jiangxiaotian/datasets/mul2347_6val' #T7920 Liang
# rcsdir = r'/home/jiangxiaotian/datasets/mul2347_6smallval' #T7920 Liang
# rcsdir = r'/home/jiangxiaotian/datasets/mul2347_6smallval' #T7920 Liang
rcsdir = r'/home/jiangxiaotian/datasets/mul2347_mie_pretrain' #T7920 Liang

total_size = 0
for root, dirs, files in os.walk(rcsdir):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        total_size += os.path.getsize(file_path)
total_size_mb = total_size / (1024 * 1024)  # 转换为 MB
total_size_gb = total_size / (1024 * 1024 * 1024)  # 转换为 GB
print(f"数据集文件夹大小(内存占用)：{total_size_mb:.2f} MB 或 {total_size_gb:.2f} GB")

for file in tqdm(os.listdir(rcsdir),desc=f'数据集加载进度',ncols=100,postfix='后缀'):
    if '.pt' in file:
        # print(file)
        plane, theta, phi, freq= re.search(r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d.+).pt", file).groups()
        theta = int(theta)
        phi = int(phi)
        freq = float(freq)
        in_em = [plane,theta,phi,freq]    # print(in_em)
        savedir = os.path.join(rcsdir,file).replace('.pt','.png')
        try:
            rcs = torch.Tensor(torch.load(os.path.join(rcsdir,file)))
        except Exception as e:
            corrupted_files.append(os.path.join(rcsdir,file))
            print(f"Error loading file {os.path.join(rcsdir,file)}: {e}")
        plot2DRCS(rcs=rcs, savedir=savedir,logger=None,cutmax=None)
        # in_ems.append(in_em)
        # rcss.append(rcs)
        print(f'已画{in_em}的GT图，保存到{savedir}')

print(f"损坏的文件：{corrupted_files}")
print(f'结束时间：{time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))}')
print(f'作图用时： {time.strftime("%H:%M:%S", time.gmtime(time.time()-tic0))}')
