#C:/ProgramData/Anaconda3/envs/jxtnet/python.exe -u d:/workspace/jxtnet/NNtrain_dataset.py > log0423san.txt
# python -u NNtrain_dataset.py > ./log0423san.txt
#SCREEN ctrl+D删除 ctrl+AD关闭 screen -S name创建 screen -r name回复 screen -ls查看list
#tmux attach -t name恢复 
import torch
import time
from tqdm import tqdm
from net.jxtnet_padding2w import MeshAutoencoder
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import os
import re
import numpy as np
import trimesh

class meshRCSDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Mesh, RCSmap):
        self.meshdata = Mesh
        self.RCSmap = RCSmap
    #返回数据集大小
    def __len__(self):
        return len(self.meshdata)
    #得到数据内容和标签
    def __getitem__(self, index):
        meshdata = torch.Tensor(self.meshdata[index])
        RCSmap = torch.Tensor(self.RCSmap[index])
        return meshdata, RCSmap

start_time0 = time.time()
print('代码开始时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))  

# rcsdir = r'D:\datasets\RCS_map' #T640
rcsdir = r'/mnt/Disk/jiangxiaotian/datasets/b827_xiezhen' #T7920 
# rcsdir = r'/mnt/Disk/jiangxiaotian/datasets/RCS_mapsmall' #T7920 test
# pretrainweight = r'D:\workspace\jxtnet\output\last0420.pt' #T640
pretrainweight = r'/home/ljm/workspace/jxt/jxtnet/last0423.pt' #T7920
loadobj = r'./datasets/b82731071bd39b66e4c15ad8a2edd2e.obj'
loadpt = r'./datasets/b827_edges.pt'
in_ems = []
rcss = []
batchsize = 1
epoch = 100
shuffle = False
use_preweight = False
multigpu = False

# start_timedata = time.time()
# print('开始加载数据集')
for file in tqdm(os.listdir(rcsdir),desc=f'数据集加载进度',ncols=100,postfix='后缀'):
    # print(file)
    theta, phi, freq= re.search(r"RCSmap_theta(\d+)phi(\d+)f(\d.+).pt", file).groups()
    theta = int(theta)
    phi = int(phi)
    freq = float(freq)
    in_em = [theta,phi,freq]
    # print(in_em)
    rcs = torch.load(os.path.join(rcsdir,file))

    in_ems.append(in_em)
    rcss.append(rcs)
dataset = meshRCSDataset(in_ems, rcss)
dataloader = DataLoader.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=0) #创建DataLoader迭代器
# end_timedata = time.time()
# print('数据集加载完成，耗时:',time.strftime("%H:%M:%S", time.gmtime(end_timedata-start_timedata)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f'device:{device}')

# in_em = torch.tensor([90,330,2]).to(device) #入射波\theta \phi freq
# planesur_face = torch.load('planesur_face.pt').to(device)
# planesur_vert = torch.load('planesur_vert.pt').to(device)
# planesur_faceedge = torch.load('face_edges.pt').to(device) #这个face_edges是图论边，不是物理边，那这个生成边的代码不用动。
# if batchsize > 1 :
#     planesur_face = torch.load('planesur_face.pt').repeat(batchsize, 1, 1).to(device)
#     planesur_vert = torch.load('planesur_vert.pt').repeat(batchsize, 1, 1).to(device)
#     planesur_faceedge = torch.load('face_edges.pt').repeat(batchsize, 1, 1).to(device)
mesh = trimesh.load_mesh(loadobj)
planesur_face = torch.tensor(mesh.faces,dtype=int).unsqueeze(0).to(device)
planesur_vert = torch.tensor(mesh.vertices,dtype=torch.float32).unsqueeze(0).to(device)
planesur_faceedge = torch.load(loadpt).to(device)
print(f"物体：{loadobj}， verts={planesur_vert.shape}， faces={planesur_face.shape}， edge={planesur_faceedge.shape}")

autoencoder = MeshAutoencoder( #这里实例化，是进去跑了init
    num_discrete_coors = 128
)
if use_preweight == True:
    autoencoder.load_state_dict(torch.load(pretrainweight), strict=False)
    print(f'成功加载预训练权重{pretrainweight}')

# if device.type=='cuda' and torch.cuda.device_count() > 1 and multigpu == True:
#     print(f"use {torch.cuda.device_count()} GPUs!")
#     autoencoder = DP(autoencoder)
autoencoder = autoencoder.to(device)
optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

epoch_loss1 = 0.
for i in range(epoch):
    epoch_loss = 0.
    tqdm.write(f'epoch:{i+1}')
    for in_emm,rcs1 in tqdm(dataloader,desc=f'epoch:{i+1}，数据集遍历进度',ncols=130,postfix=f'上一轮的epoch:{i},loss_mean:{(epoch_loss1/dataset.__len__()):.4f}'):
        loss, outrcs = autoencoder( #这里使用网络，是进去跑了forward
            vertices = planesur_vert,
            faces = planesur_face, #torch.Size([batchsize, 33564, 3])
            face_edges = planesur_faceedge,
            in_em = in_emm.to(device),
            GT = rcs1.to(device) #这里放真值
        )
        optimizer.zero_grad()
        if batchsize > 1:
            tqdm.write(f'loss:{loss.tolist()}')
            loss=loss.mean()
            loss.backward() #这一步很花时间，但是没加optimizer是白给的
        else:
            outem = [int(in_emm[0][0]), int(in_emm[0][1]), float(f'{in_emm[0][2]:.3f}')]
            tqdm.write(f'em:{outem},loss:{loss.item():.4f}')
            loss.backward()
        epoch_loss=epoch_loss + loss.item()
        optimizer.step()
        torch.cuda.empty_cache()
    if i % 50 == 0:
    #     torch.save(autoencoder.state_dict(), f"D:\\workspace\\jxtnet\\output\\epoch{i+1}0420.pt") #T640
    # torch.save(autoencoder.state_dict(), r"D:\workspace\jxtnet\output\last0420.pt") #T640
        torch.save(autoencoder.state_dict(), f"/home/ljm/workspace/jxt/jxtnet/output/b827xiezhen_epoch{i+1}0430.pt") #T7920
    torch.save(autoencoder.state_dict(), r"/home/ljm/workspace/jxt/jxtnet/output/b827xiezhen_last0430.pt") #T7920

    epoch_loss1 = epoch_loss
    tqdm.write(f'---------------------epoch:{i+1},loss_mean:{(epoch_loss1/dataset.__len__()):.4f}---------------------↑')


print('训练结束时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
end_time0 = time.time()
print('训练用时：', time.strftime("%H:%M:%S", time.gmtime(end_time0-start_time0)))
#2024年4月2日22:25:07 终于从头到尾跟着跑完了一轮 明天开始魔改！
#2024年4月6日17:24:56 encoder和decoder加入了EM因素，NN魔改完成，接下来研究如何训练。
#2024年4月6日18:13:55 loss.backward optimizer .to(device)搞定，循环已开始，接下来研究如何dataloader
#2024年4月15日22:16:57 dataset dataloader mydecoder MSEloss搞定，循环+遍历数据集已开始，jxt史上第一个手搓NN正式开始训练！
#2024年4月17日20:43:23 多gpu(但是有loss全0 NAN inf的bug) 训练日志epoch平均loss搞定 但是loss为什么训练起来好像没什么效果呢
#2024年4月20日19:37:17 能实现loss训练了，但是训练的速度太慢了，一个21000大小的数据集epoch要10h，训练完到loss变小的得一个月多。感觉还是得实现多卡，下次试试DistributedDataParallel，增大batchsize到4用P40跑试试看。
#2024年4月23日15:36:54 实现了torch.load，能从以前训练的checkpoint接着训练了，这样就能解决数据集不能一次读完的问题