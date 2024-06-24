#C:/ProgramData/Anaconda3/envs/jxtnet/python.exe -u d:/workspace/jxtnet/NNtrain_dataset.py > log0423san.txt
# python -u NNtrain_upconv.py > ./log0430upconvtest.txt
#SCREEN ctrl+D删除 ctrl+AD关闭 screen -S name创建 screen -r name回复 screen -ls查看list
#tmux attach -t name恢复 
import torch
import time
from tqdm import tqdm
# from net.jxtnet_upConv5 import MeshAutoencoder
from net.jxtnet_upConv4_InsNorm import MeshAutoencoder
# from net.jxtnet_upConv4_relu import MeshAutoencoder
# from net.jxtnet_upConv4 import MeshAutoencoder
import torch.utils.data.dataloader as DataLoader
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DataParallel as DP
import os
import sys
import re
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from pathlib import Path
from net.utils import increment_path, meshRCSDataset, get_logger, get_model_memory, psnr, ssim, find_matching_files, process_files #, get_tensor_memory, transform_to_log_coordinates
from NNvalfast import plotRCS2, plot2DRCS, valmain
# from pytorch_memlab import profile, set_target_gpu


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
start_time0 = time.time()
print('代码开始时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))  

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

batchsize = 10 #1卡12是极限了 0卡10是极限
# epoch = 1000
epoch = 1000
use_preweight = True
use_preweight = False
cudadevice = 'cuda:0'
lgrcs = True
lgrcs = False

threshold = 20
learning_rate = 0.001  # 初始学习率
lr_time = 20

shuffle = True
# shuffle = False
multigpu = False 

bestloss = 100000
epoch_loss1 = 0.
in_ems = []
rcss = []
cnt = 0
losses = []  # 用于保存每个epoch的loss值
psnr_list = []
ssim_list = []
mse_list = []
psnrs = []
ssims = []
mses = []
corrupted_files = []

# rcsdir = r'/home/jiangxiaotian/datasets/mul2347_pretrain' #T7920 Liang
# valdir = r'/home/jiangxiaotian/datasets/mul2347_6val'
rcsdir = r'/home/jiangxiaotian/datasets/traintest' #T7920 Liang
valdir = r'/home/jiangxiaotian/datasets/traintest' #T7920 Liang

# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul2347_pretrain' #T7920 
# valdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul2347_6val'
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul2347_train' #T7920 
# valdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul2347_6val_small'
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul26_MieOpt' #T7920 
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul26_MieOpt_test100' #T7920 
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul_test10' #T7920 
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul_MieOpt' #T7920 
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul_MieOptpretrain' #T7920 
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_MieOpt' #T7920 
# rcsdir = r"/mnt/Disk/jiangxiaotian/puredatasets/b82731071bd39b66e4c15ad8a2edd2e" #T7920 
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_xiezhen' #T7920 
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_xiezhen_val' #T7920 1300个
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_xiezhen_ctrl9090_test10' #T7920 
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_test10'#T7920 test
# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_xiezhen_pretrain'#T7920 pretrain
# rcsdir = r'/mnt/f/datasets/b827_test10' #305winwsl
# rcsdir = r'/mnt/f/datasets/mul_test10' #305winwsl
# rcsdir = r'D:\datasets\mul2347_pretrain' #T640
# valdir = r'D:\datasets\mul2347_6val' #T640
# pretrainweight = r'./output/train/0618upconv4_mul2347pretrain_/last.pt' #T7920
pretrainweight = r'./output/train/0615upconv4fckan_mul2347pretrain_000/last.pt' #T7920

save_dir = str(increment_path(Path(ROOT / "output" / "test" /'0624upconv4plus_mul2347pretrain_'), exist_ok=False))##日期-NN结构-飞机-训练数据-改动
# save_dir = str(increment_path(Path(ROOT / "output" / "train" /'0518upconv3L1_b827_MieOpt'), exist_ok=False))##日期-NN结构-飞机-训练数据-改动
lastsavedir = os.path.join(save_dir,'last.pt')
bestsavedir = os.path.join(save_dir,'best.pt')
lossessavedir = os.path.join(save_dir,'loss.png')
psnrsavedir = os.path.join(save_dir,'psnr.png')
ssimsavedir = os.path.join(save_dir,'ssim.png')
msesavedir = os.path.join(save_dir,'mse.png')
logdir = os.path.join(save_dir,'log.txt')
logger = get_logger(logdir)

logger.info(f'参数设置：batchsize={batchsize}, epoch={epoch}, use_preweight={use_preweight}, cudadevice={cudadevice}, threshold={threshold}, learning_rate={learning_rate}, lr_time={lr_time}, shuffle={shuffle}, multigpu={multigpu}, lgrcs={lgrcs}')
logger.info(f'数据集用{rcsdir}训练')
logger.info(f'保存到{lastsavedir}')

for file in tqdm(os.listdir(rcsdir),desc=f'数据集加载进度',ncols=100,postfix='后缀'):
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
        corrupted_files.append(os.path.join(rcsdir,file))
        logger.info(f"Error loading file {os.path.join(rcsdir,file)}: {e}")
    in_ems.append(in_em)
    rcss.append(rcs)

total_size = 0
for root, dirs, files in os.walk(rcsdir):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        total_size += os.path.getsize(file_path)
total_size_mb = total_size / (1024 * 1024)  # 转换为 MB
total_size_gb = total_size / (1024 * 1024 * 1024)  # 转换为 GB
logger.info(f"数据集文件夹大小(内存占用)：{total_size_mb:.2f} MB 或 {total_size_gb:.2f} GB")

dataset = meshRCSDataset(in_ems, rcss)
dataloader = DataLoader.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=0) #创建DataLoader迭代器
# end_timedata = time.time()
# print('数据集加载完成，耗时:',time.strftime("%H:%M:%S", time.gmtime(end_timedata-start_timedata)))

device = torch.device(cudadevice if torch.cuda.is_available() else "cpu")
# device = 'cpu'
logger.info(f'device:{device}')

autoencoder = MeshAutoencoder( #这里实例化，是进去跑了init
    num_discrete_coors = 128,
    device= device
)
get_model_memory(autoencoder,logger)

if use_preweight == True:
    autoencoder.load_state_dict(torch.load(pretrainweight), strict=False)
    logger.info(f'成功加载预训练权重{pretrainweight}')
else:
    logger.info('未使用预训练权重，为从头训练')

# if device.type=='cuda' and torch.cuda.device_count() > 1 and multigpu == True:
#     print(f"use {torch.cuda.device_count()} GPUs!")
#     autoencoder = DP(autoencoder)
autoencoder = autoencoder.to(device)
# optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_time)# CosineAnnealingLR使用余弦函数调整学习率，可以更平滑地调整学习率

flag = 1
GTflag = 1
for i in range(epoch):
    logger.info('\n')
    epoch_loss = 0.
    # tqdm.write(f'epoch:{i+1}')
    timeepoch = time.time()
    for in_em1,rcs1 in tqdm(dataloader,desc=f'epoch:{i+1},train进度,lr={scheduler.get_last_lr()[0]:.5f}',ncols=130,postfix=f'上一轮的epoch:{i},loss_mean:{(epoch_loss1/dataset.__len__()):.4f}'):
        in_em0 = in_em1.copy()
        optimizer.zero_grad()
        objlist , ptlist = find_matching_files(in_em1[0], "./planes")
        planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device)
        # logger.info(f"物体：{objlist}， verts={planesur_verts.shape}， faces={planesur_faces.shape}， edge={planesur_faceedges.shape}")

        loss, outrcs, psnr_mean, _, ssim_mean, _, mse_mean = autoencoder( #这里使用网络，是进去跑了forward 
            vertices = planesur_verts,
            faces = planesur_faces, #torch.Size([batchsize, 33564, 3])
            face_edges = planesur_faceedges,
            geoinfo = geoinfo, #[area, volume, scale]
            in_em = in_em1,#.to(device)
            GT = rcs1.to(device), #这里放真值
            logger = logger,
            device = device,
            lgrcs = lgrcs
        )
        if lgrcs == True:
            outrcslg = outrcs
            outrcs = torch.pow(10, outrcs)
        if batchsize > 1:
            # tqdm.write(f'loss:{loss.tolist()}')
            loss=loss.sum()
            loss.backward() #这一步很花时间，但是没加optimizer是白给的
        else:
            # outem = [int(in_em1[0][0]), int(in_em1[0][1]), float(f'{in_em1[0][2]:.3f}')]
            # tqdm.write(f'em:{outem},loss:{loss.item():.4f}')
            loss.backward()
        epoch_loss=epoch_loss + loss.item()
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=threshold)
        optimizer.step()
        torch.cuda.empty_cache()

        psnr_list.append(psnr_mean)
        ssim_list.append(ssim_mean)
        mse_list.append(mse_mean)
        
    #-----------------------------------定期作图看效果小模块-------------------------------------------
        in_em0[1:] = [tensor.to(device) for tensor in in_em0[1:]]
        if flag == 1:
            # print(f'rcs:{outrcs.shape},em:{in_em1}in{in_em1.shape}')
            # print(f'rcs0{outrcs[0]==outrcs[0,:]}')
            drawrcs = outrcs[0]
            # drawem = torch.stack(in_em1[1:]).t()[0]
            drawem = torch.stack(in_em0[1:]).t()[0]
            drawGT = rcs1[0]
            drawplane = in_em0[0][0]
            flag = 0
        for j in range(torch.stack(in_em0[1:]).t().shape[0]):
            # print(f'em:{in_em1[0]},drawem:{drawem}')
            if flag == 0 and torch.equal(torch.stack(in_em0[1:]).t()[j], drawem):
                drawrcs = outrcs[j]
                break
                # print(drawrcs.shape)
    p = psnr(drawrcs.to(device), drawGT.to(device))
    s = ssim(drawrcs.to(device), drawGT.to(device))
    m = torch.nn.functional.mse_loss(drawrcs.to(device), drawGT.to(device))
    if GTflag == 1:
        outGTpngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_GT.png')
        out2DGTpngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_2DGT.png')
        # plotRCS2(rcs=drawGT, savedir=outGTpngpath, logger=logger)
        plot2DRCS(rcs=drawGT, savedir=out2DGTpngpath, logger=logger,cutmax=None)
        GTflag = 0
        logger.info('已画GT图')
    if i == 0 or i % 50 == 0: #存指定倍数轮时画某张图看训练效果
        outrcspngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_epoch{i}.png')
        out2Drcspngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_epoch{i}_psnr{p.item():.2f}_ssim{s.item():.4f}_mse{m:.4f}_2D.png')
        # plotRCS2(rcs=drawrcs, savedir=outrcspngpath, logger=logger)
        plot2DRCS(rcs=drawrcs, savedir=out2Drcspngpath, logger=logger,cutmax=None)
        logger.info(f'已画{i}轮图')

    epoch_loss1 = epoch_loss
    epoch_mean_loss = epoch_loss1/dataset.__len__()
    losses.append(epoch_mean_loss)  # 保存当前epoch的loss以备绘图

    epoch_psnr = sum(psnr_list)/len(psnr_list)
    epoch_ssim = sum(ssim_list)/len(ssim_list)
    epoch_mse = sum(mse_list)/len(mse_list)
    psnrs.append(epoch_psnr)
    ssims.append(epoch_ssim)
    mses.append(epoch_mse)

    if bestloss > epoch_mean_loss:
        bestloss = epoch_mean_loss
        torch.save(autoencoder.state_dict(), bestsavedir)
    torch.save(autoencoder.state_dict(), lastsavedir) #T7920

    scheduler.step()
    # if i % 20 == 0: #存指定倍数轮的checkpoint
    #     checkpointsavedir = os.path.join(save_dir,f'epoch{i}.pt')
    #     torch.save(autoencoder.state_dict(), checkpointsavedir) #T7920

    logger.info(f'↓-----------------本epoch用时：{time.strftime("%H:%M:%S", time.gmtime(time.time()-timeepoch))}-----------------↓')
    logger.info(f'↑----epoch:{i+1},loss:{epoch_mean_loss:.4f},psnr:{epoch_psnr:.2f},ssim:{epoch_ssim:.4f},mse:{epoch_mse:.4f}----↑')
    
    # 绘制loss曲线图
    plt.clf()
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(0, i+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig(lossessavedir)
    
    # 绘制psnr曲线图
    plt.clf()
    plt.plot(range(0, i+1), psnrs)
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Training PSNR Curve')
    plt.savefig(psnrsavedir)
    
    # 绘制ssim曲线图
    plt.clf()
    plt.plot(range(0, i+1), ssims)
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Training SSIM Curve')
    plt.savefig(ssimsavedir)

    # 绘制mse曲线图
    plt.clf()
    plt.plot(range(0, i+1), mses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training MSE Curve')
    plt.savefig(msesavedir)
    plt.close()
    # plt.show()
    # if i % 50 == 0 or i == -1: #存指定倍数轮的checkpoint
    #     valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, batchsize=batchsize, trainval=True, draw3d=False, lgrcs=lgrcs)

logger.info(f"损坏的文件：{corrupted_files}")
logger.info(f'训练结束时间：{time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))}')
end_time0 = time.time()
logger.info(f'训练用时： {time.strftime("%H:%M:%S", time.gmtime(end_time0-start_time0))}')
