# python NNtrain_arg.py --seed 777 --gama 0.0005 --cuda 'cuda:0'
#0802会议精神：1.decoder换成Transformer试试，swin-Transformer可能引入不合理的归纳偏置(可能不行 因为361*720对于Transformer来说太长了。。)；2.incident conditioning也要归一化 否则不合理(已完成)；3.AutoEncoder用纯3D训练，decoder再单独训练一个带skip connection的unet decoder(重头戏)；4.学习率调度不要一直起伏，而是用半个cos先大后小这样做；5.定长pooling实现，这样就不用padding了，否则有0还是不合理。
import torch
import time
from tqdm import tqdm
from net.jxtnet_Transupconv import MeshEncoderDecoder
# from net.jxtnet_Transupconv_fan import MeshEncoderDecoder
# from net.jxtnet_pureTrans import MeshEncoderDecoder
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
from net.utils import increment_path, meshRCSDataset, get_logger, get_model_memory, psnr, ssim, find_matching_files, process_files, get_x_memory#, get_tensor_memory, toc, checksize#, transform_to_log_coordinates
from NNvalfast import  plot2DRCS, valmain#, plotRCS2
from pytictoc import TicToc
t = TicToc()
t.tic()
import random
import numpy as np
import argparse

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
    parser.add_argument('--epoch', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--use_preweight', type=bool, default=False, help='Whether to use pretrained weights')
    parser.add_argument('--draw', type=bool, default=True, help='Whether to enable drawing')

    parser.add_argument('--trainname', type=str, default='mul2347', help='logname')
    parser.add_argument('--rcsdir', type=str, default='/home/jiangxiaotian/datasets/mul2347_mie_pretrain', help='Path to rcs directory') #liang
    parser.add_argument('--valdir', type=str, default='/home/jiangxiaotian/datasets/mul2347_mie_6smallval', help='Path to validation directory') #liang
    # parser.add_argument('--rcsdir', type=str, default='/home/ljm/workspace/datasets/mulbb7c_mie_pretrain', help='Path to rcs directory')
    # parser.add_argument('--valdir', type=str, default='/home/ljm/workspace/datasets/mulbb7c_mie_val', help='Path to validation directory')
    parser.add_argument('--pretrainweight', type=str, default='/mnt/SrvUserDisk/JiangXiaotian/workspace/3DEM/output/train/1129_TransConv_pretrain_b7fd_nofilter/last.pt', help='Path to pretrained weights')

    parser.add_argument('--seed', type=int, default=77, help='Random seed for reproducibility')
    parser.add_argument('--gama', type=float, default=0.0005, help='Loss threshold or gamma parameter')
    parser.add_argument('--cuda', type=str, default='cuda:1', help='CUDA device to use')
    return parser.parse_args()

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
tic0 = time.time()
tic = time.time()
print('代码开始时间：',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))  

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

args = parse_args()

# 使用命令行参数
epoch = args.epoch
use_preweight = args.use_preweight
draw = args.draw
rcsdir = args.rcsdir
valdir = args.valdir
pretrainweight = args.pretrainweight
seed = args.seed
gama = args.gama
cudadevice = args.cuda
name = args.trainname

if 'pretrain' in rcsdir:
    mode = 'pretrain'
else:
    if use_preweight == True:
        mode = 'finetune'
    else:
        mode = 'train'
# 设置随机种子
setup_seed(seed)

# 其他固定参数
accumulation_step = 8
threshold = 20
bestloss = 1
maxpsnr = 1
epoch_loss1 = 0.0
in_ems = []
rcss = []
cnt = 0
losses = []  # 用于保存每个epoch的loss值
psnrs = []
ssims = []
mses = []
corrupted_files = []
lgrcs = False
shuffle = True
multigpu = False
alpha = 0.0
learning_rate = 0.001  # 初始学习率
lr_time = epoch
batchsize = 10  # TransUpconv能10 PureTrans只能4
encoder_layer = 6
decoder_outdim = 12  # 3S 6M 12L
paddingsize = 18000

from datetime import datetime
date = datetime.today().strftime("%m%d")
save_dir = str(increment_path(Path(ROOT / "output" / "train" /f'{date}_{mode}_{name}_seed{seed}_maxloss{gama}_{cudadevice}_'), exist_ok=False))##
lastsavedir = os.path.join(save_dir,'last.pt')
bestsavedir = os.path.join(save_dir,'best.pt')
maxsavedir = os.path.join(save_dir,'maxp.pt')
lossessavedir = os.path.join(save_dir,'loss.png')
psnrsavedir = os.path.join(save_dir,'psnr.png')
ssimsavedir = os.path.join(save_dir,'ssim.png')
msesavedir = os.path.join(save_dir,'mse.png')
logdir = os.path.join(save_dir,'log.txt')
logger = get_logger(logdir)
# logger.info(f'使用net.jxtnet_pureTrans')
# logger.info(f'使用net.jxtnet_Transupconv')

# logger.info(f'使用jxtnet_transformerEncoder.py')
logger.info(f'参数设置：batchsize={batchsize}, epoch={epoch}, use_preweight={use_preweight}, cudadevice={cudadevice}, learning_rate={learning_rate}, lr_time={lr_time}, shuffle={shuffle}, gama={gama}, seed={seed}, rcsdir = {rcsdir}, valdir = {valdir}, pretrainweight = {pretrainweight}')
logger.info(f'数据集用{rcsdir}训练')
logger.info(f'保存到{lastsavedir}')

for file in tqdm(os.listdir(rcsdir),desc=f'数据集加载进度',ncols=100,postfix='后缀'):
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
            corrupted_files.append(os.path.join(rcsdir,file))
            logger.info(f"Error loading file {os.path.join(rcsdir,file)}: {e}")
        in_ems.append(in_em)
        rcss.append(rcs)
get_x_memory(rcss,logger)
# get_tensor_memory(torch.Tensor(rcss),logger)

dataset = meshRCSDataset(in_ems, rcss)
# get_x_memory(dataset,logger)

dataloader = DataLoader.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=0) #创建DataLoader迭代器

device = torch.device(cudadevice if torch.cuda.is_available() else "cpu")
# device = 'cpu'
logger.info(f'device:{device}')

autoencoder = MeshEncoderDecoder( #这里实例化，是进去跑了init 草 但是这里还是用的paddingsize
    num_discrete_coors = 128,
    device= device,
    paddingsize = paddingsize,
    decoder_outdim = decoder_outdim, #决定了decoder的size 12L 6M 3S
    encoder_layer = encoder_layer, #决定了encoder的层数
    alpha = alpha
)
get_model_memory(autoencoder,logger)

if use_preweight == True:
    autoencoder.load_state_dict(torch.load(pretrainweight), strict=False)
    logger.info(f'成功加载预训练权重{pretrainweight}')
else:
    logger.info('未使用预训练权重，为从头训练')

autoencoder = autoencoder.to(device)
# optimizer = torch.optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_time)# CosineAnnealingLR使用余弦函数调整学习率，可以更平滑地调整学习率

# t.toc('代码准备时间',restart=True)

flag = 1
GTflag = 1
for i in range(epoch):
    psnr_list = []
    ssim_list = []
    mse_list = []
    jj=0
    logger.info('\n')
    epoch_loss = 0.
    # tqdm.write(f'epoch:{i+1}')
    timeepoch = time.time()
    # for in_em1,rcs1 in tqdm(dataloader,desc=f'epoch:{i+1},train进度',ncols=130,postfix=f'上一轮的epoch:{i},loss_mean:{(epoch_loss1/dataset.__len__()):.4f}'):
    for in_em1,rcs1 in tqdm(dataloader,desc=f'epoch:{i+1},train进度,lr={scheduler.get_last_lr()[0]:.5f}',ncols=130,postfix=f'上一轮的epoch:{i},loss_mean:{(epoch_loss1/dataset.__len__()):.4f}'):
        # print('-->')
        # t.toc('刚进循环',restart=True)
        
        jj=jj+1
        in_em0 = in_em1.copy()
        # optimizer.zero_grad()
        objlist , ptlist = find_matching_files(in_em1[0], "./planes")
        planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device) #为了解决多batch变量长度不一样的问题 在这一步就已经padding到等长了
        # logger.info(f"物体：{objlist}， verts={planesur_verts.shape}， faces={planesur_faces.shape}， edge={planesur_faceedges.shape}")
        # print('\n--循环准备总时长')
        # tic=toc(tic)
        
        # t.toc('循环内准备时间',restart=True)
        loss, outrcs, psnr_mean, _, ssim_mean, _, mse_mean,_ = autoencoder( #这里使用网络，是进去跑了forward 
            vertices = planesur_verts,
            faces = planesur_faces, #torch.Size([batchsize, 33564, 3])
            geoinfo = geoinfo, #[area, volume, scale]
            in_em = in_em1,#.to(device)
            GT = rcs1.to(device), #这里放真值
            logger = logger,
            device = device,
            lgrcs = lgrcs,
            gama=gama,
        )
        # print('--推理总时长:')
        # tic=toc(tic)
        # t.toc('推理时间',restart=True)
        
        if lgrcs == True:
            outrcslg = outrcs
            outrcs = torch.pow(10, outrcs)
        if batchsize > 1:
            # tqdm.write(f'loss:{loss.tolist()}')
            loss=loss.mean() / accumulation_step #loss.sum()改成loss.mean()
            loss.backward() #这一步很花时间，但是没加optimizer是白给的
            # print('--loss.backward：')
            # tic=toc(tic)
        else:
            outem = [int(in_em1[1]), int(in_em1[2]), float(f'{in_em1[3].item():.3f}')]
            tqdm.write(f'em:{outem},loss:{loss.item():.4f}')
            loss=loss / accumulation_step
            loss.backward()
        # t.toc('loss反传时间',restart=True)
        epoch_loss=epoch_loss + loss.item()
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=threshold)
        # optimizer.step()
        if (jj) % accumulation_step == 0 or (jj) == len(dataloader):
            optimizer.step() #结果发现这一步也不花时间。。
            optimizer.zero_grad()
            # print('--优化器:')
            # tic=toc(tic)
        # torch.cuda.empty_cache()
        psnr_list.append(psnr_mean)
        ssim_list.append(ssim_mean)
        mse_list.append(mse_mean)
        
    #-----------------------------------定期作图看效果小模块-------------------------------------------
        in_em0[1:] = [tensor.to(device) for tensor in in_em0[1:]]
        if flag == 1:
            # print(f'rcs:{outrcs.shape},em:{in_em1}in{in_em1.shape}')
            # print(f'rcs0{outrcs[0]==outrcs[0,:]}')
            drawrcs = outrcs[0].unsqueeze(0)
            # drawem = torch.stack(in_em1[1:]).t()[0]
            drawem = torch.stack(in_em0[1:]).t()[0]
            drawGT = rcs1[0][:-1,:].unsqueeze(0)
            drawplane = in_em0[0][0]
            flag = 0
        for j in range(torch.stack(in_em0[1:]).t().shape[0]):
            # print(f'em:{in_em1[0]},drawem:{drawem}')
            if flag == 0 and torch.equal(torch.stack(in_em0[1:]).t()[j], drawem):
                drawrcs = outrcs[j].unsqueeze(0)
                break
                # print(drawrcs.shape)
        # print('--loss到本轮batch结束时长：')
        # tic = toc(tic)
        # t.toc('循环尾巴时间',restart=True)
    logger.info(save_dir)

    p = psnr(drawrcs.to(device), drawGT.to(device))
    s = ssim(drawrcs.to(device), drawGT.to(device))
    m = torch.nn.functional.mse_loss(drawrcs.to(device), drawGT.to(device))
    if GTflag == 1:
        outGTpngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_GT.png')
        out2DGTpngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_2DGT.png')
        # plotRCS2(rcs=drawGT, savedir=outGTpngpath, logger=logger)
        plot2DRCS(rcs=drawGT.squeeze(), savedir=out2DGTpngpath, logger=logger,cutmax=None)
        GTflag = 0
        logger.info('已画GT图')
    if i == 0 or i % 10 == 0: #存指定倍数轮时画某张图看训练效果
        outrcspngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_epoch{i}.png')
        out2Drcspngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_epoch{i}_psnr{p.item():.2f}_ssim{s.item():.4f}_mse{m:.4f}_2D.png')
        # plotRCS2(rcs=drawrcs, savedir=outrcspngpath, logger=logger)
        plot2DRCS(rcs=drawrcs.squeeze(), savedir=out2Drcspngpath, logger=logger,cutmax=None)
        logger.info(f'已画{i}轮图')

    epoch_loss1 = epoch_loss
    epoch_mean_loss = epoch_loss1/dataset.__len__()
    losses.append(epoch_mean_loss)  # 保存当前epoch的loss以备绘图

    epoch_psnr = sum(psnr_list)/len(psnr_list) #这个应该每轮清零的
    epoch_ssim = sum(ssim_list)/len(ssim_list)
    epoch_mse = sum(mse_list)/len(mse_list)
    psnrs.append(epoch_psnr) #这个不是每轮清零，是和轮数长度一样的用于作图的
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

    logger.info(f'↓-----------------------本epoch用时：{time.strftime("%H:%M:%S", time.gmtime(time.time()-timeepoch))}-----------------------↓')
    logger.info(f'↑----epoch:{i+1}(lr:{scheduler.get_last_lr()[0]:.4f}),loss:{epoch_mean_loss:.4f},psnr:{epoch_psnr:.2f},ssim:{epoch_ssim:.4f},mse:{epoch_mse:.4f}----↑')
    
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
    if mode == "pretrain":
        if (i+1) % 20 == 0 or i == -1: #存指定倍数轮的checkpoint
        # if (i+1) % 1 == 0 or i == -1: #存指定倍数轮的checkpoint
            valpsnr=valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, batchsize=batchsize, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize)
    else :
        if (i+1) % 1 == 0 or i == -1: #存指定倍数轮的checkpoint
            valpsnr=valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, batchsize=batchsize, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize)
    if maxpsnr < valpsnr:
        maxpsnr = valpsnr
        torch.save(autoencoder.state_dict(), maxsavedir)

logger.info(f"损坏的文件：{corrupted_files}")
logger.info(f'训练结束时间：{time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))}')
logger.info(f'训练用时： {time.strftime("%H:%M:%S", time.gmtime(time.time()-tic0))}')
