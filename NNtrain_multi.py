#C:/ProgramData/Anaconda3/envs/jxtnet/python.exe -u d:/workspace/jxtnet/NNtrain_dataset.py > log0423san.txt
# python -u NNtrain_upconv.py > ./log0430upconvtest.txt
#SCREEN ctrl+D删除 ctrl+AD关闭 screen -S name创建 screen -r name回复 screen -ls查看list
#tmux attach -t name恢复 
import torch
import time
from tqdm import tqdm
# from net.jxtnet_upConv5 import MeshAutoencoder
from net.jxtnet_upConv4_silu import MeshAutoencoder
# from net.jxtnet_upConv4 import MeshAutoencoder
import torch.utils.data.dataloader as DataLoader
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DataParallel as DP
import os
import sys
import re
import matplotlib.pyplot as plt
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
epoch = 400
use_preweight = True
# use_preweight = False
cudadevice = 'cuda:1'
lgrcs = True
# lgrcs = False

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

# rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul2347_train' #T7920 
rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul2347_pretrain' #T7920 
valdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul2347_6val'
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
# pretrainweight = r'./output/train/0618upconv4_mul2347pretrain_/last.pt' #T7920
pretrainweight = r'./output/train/0615upconv4fckan_mul2347pretrain_000/last.pt' #T7920

save_dir = str(increment_path(Path(ROOT / "output" / "train" /'0619upconv4_mul2347pretrain_'), exist_ok=False))##日期-NN结构-飞机-训练数据-改动
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
    # plt.show()
    # if i % 50 == 0 or i == -1: #存指定倍数轮的checkpoint
    #     valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, batchsize=batchsize, trainval=True, draw3d=False, lgrcs=lgrcs)

logger.info(f"损坏的文件：{corrupted_files}")
logger.info(f'训练结束时间：{time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))}')
end_time0 = time.time()
logger.info(f'训练用时： {time.strftime("%H:%M:%S", time.gmtime(end_time0-start_time0))}')
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
#2024年5月8日18:33:16 采用了新的库画3D图，单张图由1分钟降为10秒左右。在jxtnet_upConv_piloss.py中尝试往loss中加入平滑惩罚，然后加上了relu非线性层（发现不会影响网络权重），并且尝试改梯度裁剪的thershold为20试图实现更精细的梯度下降权重优化（效果应该等效于调小lr，只不过是特别针对RCS值大的突出部分），但是loss平滑惩罚还未成功？？？虽然不报错但是好像也没啥效果。把数据集整理了一下，补漏+对称翻倍+裁剪成RCSnp保存，等最后一批处理完，就可以搬到T7920上用新的数据训练了！
#2024年5月9日22:04:27 今天把新数据用上，开始跑新的；然后把网络结构又改了一下，参考segnet维度留多点 给1D和2D都加了BN和Relu 然后最后1x1卷积；还加上了学习率调度器，用的指数/余弦衰减调度。把批量推理写了，司马了，大数据训30轮效果根本不行，10个数据训800轮还只是勉强，司马了，这样真不行，还得1优化训练速度！！2用小样本能loss训练成0了再来上大数据！！可以先帮网络控制变量，并直接把结果矩阵x2，看样子好像能好点，明天试试。
#2024年5月10日10:42:49 控制变量训练、结果矩阵x2;实现了第i整数倍轮画图看效果。plan：shapenet飞机训练，分成两个网络（3个模块）；把角度信息再卷一次强调一次，可能是进了decoder后角度信息都卷的不成样子了，所以再用eg3d的conditioning强调一次感觉会很不错:最后把encoder里的em_embedding拿过来在decoder里又和x concat了一下，让入射角度和频率信息在decoder里强调了一次。确实loss就从之前48万死活下不来下探到二十几万了，还在下降，看看能不能掌握方向性。是不是测试样本太少的原因，我要先不等训完800轮用这个val一遍本数据集，然后训中数据集ctrl9090，然后验ctrl9090_val。
#2024年5月11日15:13:06 采用logger记录日志；de了emfreq离散的bug；条件控制修改：angle conditioning+freq conditioning。
#2024年5月15日14:01:46 重大突破！！1.加了线性层，结果显著，新称conv3 2.改进了smoothloss 并添加了中值滤波+高斯滤波后处理
#2024年5月18日11:00:14 实现了SSIM和PSNR指标 解决了多batchsize的bug，现用batchsize=8训练！ 修改了L1loss 加了mse指标 de了一下午晚上的bug我草真逆天 不能同时用nn.mseloss和nn.l1loss、不全是grad的影响，太逆天了，最后采用手搓mse。 准备实现TV全变分正则化