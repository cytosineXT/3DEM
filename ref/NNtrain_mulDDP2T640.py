# torchrun --nproc_per_node=1 NNtrain_mulDDP2.py
#SCREEN ctrl+D删除 ctrl+AD关闭 screen -S name创建 screen -r name回复 screen -ls查看list
#tmux attach -t name恢复 
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from tqdm import tqdm
from net.jxtnet_upConv4 import MeshAutoencoder
import torch.utils.data.dataloader as DataLoader
import os
import sys
import re
import matplotlib.pyplot as plt
from pathlib import Path
from net.utils import increment_path, meshRCSDataset, get_logger, get_model_memory, psnr, ssim, find_matching_files, process_files
from NNvalfast import plotRCS2, plot2DRCS
import signal
import datetime

def setup(rank, world_size):
    # nprocs=world_size
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12356'
    print(f"Initializing process group for rank {rank}, worldsize{world_size}")
    dist.init_process_group(backend="gloo",init_method="tcp://localhost:12355" ,rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=1)) #T640
    # dist.init_process_group(backend="gloo",init_method="file://D:/tmp/torch_sharedfile" ,rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=1)) #T640
    # dist.init_process_group(backend="nccl",init_method="file:///tmp/torch_sharedfile" ,rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=1)) #草 卡在这步了
    # dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=20)) #草 卡在这步了
    # dist.init_process_group(backend="mpi", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=20)) #草 卡在这步了
    print(f"Process group initialized for rank {rank}")
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    print("Cleaning up process group")

def signal_handler(sig, frame):
    print(f"Received signal {sig}, exiting...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main(rank, world_size):
    print(f"Starting main for rank {rank}")
# def main():
    # rank = int(os.environ['RANK'])
    # world_size = int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)
    
    start_time0 = time.time()
    print('代码开始时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))  

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    batchsize = 1
    epoch = 200
    use_preweight = True
    use_preweight = False
    cudadevice = f'cuda:{rank}'
    
    threshold = 20
    learning_rate = 0.001  # 初始学习率
    lr_time = 20

    shuffle = False
    multigpu = True

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

    # rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul26_MieOpt_test100'
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
    rcsdir = r'D:\datasets\mul26_MieOpt_test100' #T640
    pretrainweight = r'./output/train/0529upconv4ffc_MieOptpretrain3/last.pt' #T7920
    
    save_dir = str(increment_path(Path(ROOT / "output" / "test" / '0530upconv4ffc_DDP'), exist_ok=False))
    lastsavedir = os.path.join(save_dir, 'last.pt')
    bestsavedir = os.path.join(save_dir, 'best.pt')
    lossessavedir = os.path.join(save_dir, 'loss.png')
    psnrsavedir = os.path.join(save_dir, 'psnr.png')
    ssimsavedir = os.path.join(save_dir, 'ssim.png')
    msesavedir = os.path.join(save_dir, 'mse.png')
    logdir = os.path.join(save_dir, 'log.txt')
    logger = get_logger(logdir)

    if rank == 0:
        logger.info(f'参数设置：batchsize={batchsize}, epoch={epoch}, use_preweight={use_preweight}, cudadevice={cudadevice}, threshold={threshold}, learning_rate={learning_rate}, lr_time={lr_time}, shuffle={shuffle}, multigpu={multigpu}')
        logger.info(f'数据集用{rcsdir}训练')
        logger.info(f'保存到{lastsavedir}')

    for file in tqdm(os.listdir(rcsdir), desc=f'数据集加载进度', ncols=100, postfix='后缀'):
        plane, theta, phi, freq = re.search(r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d.+).pt", file).groups()
        theta = int(theta)
        phi = int(phi)
        freq = float(freq)
        in_em = [plane, theta, phi, freq]
        rcs = torch.load(os.path.join(rcsdir, file))

        in_ems.append(in_em)
        rcss.append(rcs)

    total_size = 0
    for root, dirs, files in os.walk(rcsdir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            total_size += os.path.getsize(file_path)
    total_size_mb = total_size / (1024 * 1024)  # 转换为 MB
    total_size_gb = total_size / (1024 * 1024 * 1024)  # 转换为 GB

    if rank == 0:
        logger.info(f"数据集文件夹大小(内存占用)：{total_size_mb:.2f} MB 或 {total_size_gb:.2f} GB")

    dataset = meshRCSDataset(in_ems, rcss)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=0, sampler=sampler)

    device = torch.device(cudadevice if torch.cuda.is_available() else "cpu")
    logger.info(f'device:{device}')

    autoencoder = MeshAutoencoder(
        num_discrete_coors = 128,
        device= device
    ).to(device)
    autoencoder = DDP(autoencoder, device_ids=[rank])

    get_model_memory(autoencoder, logger)

    if use_preweight:
        autoencoder.load_state_dict(torch.load(pretrainweight), strict=False)
        if rank == 0:
            logger.info(f'成功加载预训练权重{pretrainweight}')
    else:
        if rank == 0:
            logger.info('未使用预训练权重，为从头训练')

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_time)

    flag = 1
    GTflag = 1
    for i in range(epoch):
        epoch_loss = 0.
        timeepoch = time.time()
        for in_em1, rcs1 in tqdm(dataloader, desc=f'epoch:{i+1},datasets进度,lr={scheduler.get_last_lr()[0]:.5f}', ncols=130, postfix=f'上一轮的epoch:{i},loss_mean:{(epoch_loss1/dataset.__len__()):.4f}'):
            optimizer.zero_grad()
            objlist , ptlist = find_matching_files(in_em1[0], "./planes")
            planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device)

            loss, outrcs, psnr_mean, _, ssim_mean, _, mse_mean = autoencoder(
                vertices = planesur_verts,
                faces = planesur_faces,
                face_edges = planesur_faceedges,
                geoinfo = geoinfo,
                in_em = in_em1,
                GT = rcs1.to(device),
                logger = logger,
                device = device
            )
            if batchsize > 1:
                loss=loss.sum()
                loss.backward()
            else:
                loss.backward()
            epoch_loss = epoch_loss + loss.item()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=threshold)
            optimizer.step()
            torch.cuda.empty_cache()

            psnr_list.append(psnr_mean)
            ssim_list.append(ssim_mean)
            mse_list.append(mse_mean)
            
        in_em1[1:] = [tensor.to(device) for tensor in in_em1[1:]]
        if flag == 1:
            drawrcs = outrcs[0]
            drawem = torch.stack(in_em1[1:]).t()[0]
            drawGT = rcs1[0]
            flag = 0
        for j in range(torch.stack(in_em1[1:]).t().shape[0]):
            if flag == 0 and torch.equal(torch.stack(in_em1[1:]).t()[j], drawem):
                drawrcs = outrcs[j]
                break

        p = psnr(drawrcs.to(device), drawGT.to(device))
        s = ssim(drawrcs.to(device), drawGT.to(device))
        m = torch.nn.functional.mse_loss(drawrcs.to(device), drawGT.to(device))
        if GTflag == 1 and rank == 0:
            outGTpngpath = os.path.join(save_dir, f'theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_GT.png')
            out2DGTpngpath = os.path.join(save_dir, f'theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_2DGT.png')
            plotRCS2(rcs=drawGT, savedir=outGTpngpath, logger=logger)
            plot2DRCS(rcs=drawGT, savedir=out2DGTpngpath, logger=logger)
            GTflag = 0
            logger.info('已画GT图')
        if i == 0 or i % 20 == 0 and rank == 0:
            outrcspngpath = os.path.join(save_dir, f'theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_epoch{i}.png')
            out2Drcspngpath = os.path.join(save_dir, f'theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_epoch{i}_psnr{p.item():.2f}_ssim{s.item():.4f}_mse{m:.4f}_2D.png')
            plotRCS2(rcs=drawrcs, savedir=outrcspngpath, logger=logger)
            plot2DRCS(rcs=drawrcs, savedir=out2Drcspngpath, logger=logger)
            logger.info(f'已画{i}轮图')

        epoch_loss1 = epoch_loss
        epoch_mean_loss = epoch_loss1 / dataset.__len__()
        losses.append(epoch_mean_loss)

        epoch_psnr = sum(psnr_list) / len(psnr_list)
        epoch_ssim = sum(ssim_list) / len(ssim_list)
        epoch_mse = sum(mse_list) / len(mse_list)
        psnrs.append(epoch_psnr)
        ssims.append(epoch_ssim)
        mses.append(epoch_mse)

        if bestloss > epoch_mean_loss:
            bestloss = epoch_mean_loss
            if rank == 0:
                torch.save(autoencoder.state_dict(), bestsavedir)
        if rank == 0:
            torch.save(autoencoder.state_dict(), lastsavedir)

        scheduler.step()
        
        logger.info(f'↓-----------------本epoch用时：{time.strftime("%H:%M:%S", time.gmtime(time.time()-timeepoch))}-----------------↓')
        logger.info(f'↑----epoch:{i+1},loss:{epoch_mean_loss:.4f},psnr:{epoch_psnr:.2f},ssim:{epoch_ssim:.4f},mse:{epoch_mse:.4f}----↑\n')

        # 绘制loss曲线图
        plt.clf()
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

    if rank == 0:
        logger.info(f'训练结束时间：{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}')
        end_time0 = time.time()
        logger.info(f'训练用时： {time.strftime("%H:%M:%S", time.gmtime(end_time0-start_time0))}')
    
    print(f"Process {rank} completed training")
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # world_size = 1
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
