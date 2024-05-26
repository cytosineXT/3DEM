#C:/ProgramData/Anaconda3/envs/jxtnet/python.exe -u d:/workspace/jxtnet/NNtrain_dataset.py > log0423san.txt
# python -u NNtrain_upconv_piloss.py > ./log0508pilosstest.txt
#SCREEN ctrl+D删除 ctrl+AD关闭 screen -S name创建 screen -r name回复 screen -ls查看list
#tmux attach -t name恢复 
import torch
import time
from tqdm import tqdm
from net.jxtnet_upConv2_piloss import MeshAutoencoder
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DataParallel as DP
import os
import sys
import re
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path

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
        meshdata = torch.Tensor(self.meshdata[index])
        RCSmap = torch.Tensor(self.RCSmap[index])
        return meshdata, RCSmap

start_time0 = time.time()
print('代码开始时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))  

batchsize = 2
epoch = 600
use_preweight = True
cudadevice = 'cuda:0'
threshold = 20
learning_rate = 0.0001  # 初始学习率
shuffle = False
multigpu = False

bestloss = 100000
epoch_loss1 = 0.
in_ems = []
rcss = []
cnt = 0
losses = []  # 用于保存每个epoch的loss值

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# rcsdir = r'/mnt/Disk/jiangxiaotian/datasets/b827_xiezhen' #T7920 
rcsdir = r'/mnt/Disk/jiangxiaotian/datasets/RCS_mapsmall'#T7920 test
pretrainweight = r'/home/ljm/workspace/jxt/jxtnet/output/test/0508upconv_b827_piloss3/last.pt' #T7920
loadobj = r'./datasets/b82731071bd39b66e4c15ad8a2edd2e.obj'
loadpt = r'./datasets/b827_edges.pt'

save_dir = str(increment_path(Path(ROOT / "output" / "test" /'0508upconv_b827_piloss'), exist_ok=False))##日期-NN结构-飞机-训练数据-改动
lastsavedir = os.path.join(save_dir,'last.pt')
bestsavedir = os.path.join(save_dir,'best.pt')
lossessavedir = os.path.join(save_dir,'loss.png')

print('保存到',lastsavedir)
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

device = torch.device(cudadevice if torch.cuda.is_available() else "cpu")
print(f'device:{device}')

mesh = trimesh.load_mesh(loadobj)
planesur_face = torch.tensor(mesh.faces,dtype=int).unsqueeze(0).to(device)
planesur_vert = torch.tensor(mesh.vertices,dtype=torch.float32).unsqueeze(0).to(device)
planesur_faceedge = torch.load(loadpt).to(device)
if batchsize > 1 :
    planesur_face = planesur_face.repeat(batchsize, 1, 1)
    planesur_vert = planesur_vert.repeat(batchsize, 1, 1)
    planesur_faceedge = planesur_faceedge.repeat(batchsize, 1, 1)
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
# optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-4)

for i in range(epoch):
    epoch_loss = 0.
    # tqdm.write(f'epoch:{i+1}')
    timeepoch = time.time()
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
            # tqdm.write(f'loss:{loss.tolist()}')
            loss=loss.mean()
            loss.backward() #这一步很花时间，但是没加optimizer是白给的
        else:
            outem = [int(in_emm[0][0]), int(in_emm[0][1]), float(f'{in_emm[0][2]:.3f}')]
            # tqdm.write(f'em:{outem},loss:{loss.item():.4f}')
            loss.backward()
        epoch_loss=epoch_loss + loss.item()
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=threshold)
        optimizer.step()
        torch.cuda.empty_cache()

    epoch_loss1 = epoch_loss
    epoch_mean_loss = epoch_loss1/dataset.__len__()
    losses.append(epoch_mean_loss)  # 保存当前epoch的loss以备绘图

    if bestloss > epoch_mean_loss:
        bestloss = epoch_mean_loss
    #     torch.save(autoencoder.state_dict(), bestsavedir)
    torch.save(autoencoder.state_dict(), lastsavedir) #T7920
    # if i % 20 == 0: #存指定倍数轮的checkpoint
    #     checkpointsavedir = os.path.join(save_dir,f'epoch{i}.pt')
    #     torch.save(autoencoder.state_dict(), checkpointsavedir) #T7920

    ## 手动调学习率用
    # if epoch_mean_loss < 50000 and cnt != 0:
    #     learning_rate *= 0.1  # 减小学习率的因子
    #     print(f'减小学习率为{learning_rate}')
    #     # threshold *= 0.9
    #     cnt = cnt - 1
    # optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f'\n↓------------本epoch用时：{time.strftime("%H:%M:%S", time.gmtime(time.time()-timeepoch))}------------↓')
    print(f'↑--------epoch:{i+1},loss_mean:{epoch_mean_loss:.4f}--------↑')
    
    # 绘制loss曲线图
    plt.plot(range(0, i + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig(lossessavedir)
    # plt.show()


print('训练结束时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
end_time0 = time.time()
print('训练用时：', time.strftime("%H:%M:%S", time.gmtime(end_time0-start_time0)))
#2024年4月2日22:25:07 终于从头到尾跟着跑完了一轮 明天开始魔改！
#2024年4月6日17:24:56 encoder和decoder加入了EM因素，NN魔改完成，接下来研究如何训练。
#2024年4月6日18:13:55 loss.backward optimizer .to(device)搞定，循环已开始，接下来研究如何dataloader
#2024年4月15日22:16:57 dataset dataloader mydecoder MSEloss搞定，循环+遍历数据集已开始，jxt史上第一个手搓NN正式开始训练！
#2024年4月17日20:43:23 多gpu DP(但是有loss全0 NAN inf的bug) 训练日志epoch平均loss搞定 但是loss为什么训练起来好像没什么效果呢
#2024年4月20日19:37:17 能实现loss训练了，但是训练的速度太慢了，一个21000大小的数据集epoch要10h，训练完到loss变小的得一个月多。感觉还是得实现多卡，下次试试DistributedDataParallel，增大batchsize到4用P40跑试试看。
#2024年4月23日15:36:54 实现了torch.load，能从以前训练的checkpoint接着训练了，这样就能解决数据集不能一次读完的问题。DDP不大行，手搓并发好多毛病，虽然离完美只一步之遥，但是应只用一张3090跑，没必要，等要的时候再说。
#2024年4月30日22:06:57 实现了upconv的NN Decoder设计，重新梳理了RCS和obj数据，实现更换飞机obj的代码和3Drcs结果绘图的代码，并且按计划用第一架飞机的谐振区14000数据训练。下一步看看encoder是哪里耗时间，能否优化。
#2024年5月1日18:57:04 实现了upconvde的NN Encoder加速，替换掉了爱因斯坦求和规则等，加速了十倍左右！新的代码位于jxtnet_upconv_deencoder.py中。还尝试debug了多batchsize训练，应该好了，但是因为显存问题和zx冲突，等他跑完了我再试试：如果最后loss不报大小不一的错，且值不会比为1时小一倍，说明没问题！我草刚好一个月
#2024年5月5日23:56:53 首先完成了多batch训练；其次修改了decoder，去掉了其中的BN和Relu，然后将loss改为了sum
#2024年5月6日13:03:08 修改了主体训练代码，实现了best权重存储，实现了losses结果图保存；修改了decoder，在梯度裁剪的操作下把loss成功用sum训练了！虽然是过拟合但是还是得到了很不错的效果！loss从数十万降至三万，结果图/home/ljm/workspace/jxt/jxtnet/output/inference/0506_b827_theta90phi330freq0.9_1w4W_nobn.png也证明其实是能学到的！看了一下大概400轮一直在慢慢降。看看如何把loss优化，加入相邻角度连续变化的先验，让生成的图是光滑的
#2024年5月7日17:35:23 实现了losses图随轮更新的效果。实现了datasets的削减和整理，新的data目前在任服务器puredatasets文件夹中。