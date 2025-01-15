import os
import sys
import re
import time
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as DataLoader

from net.jxtnet_upConv4 import MeshAutoencoder
from net.utils import increment_path, meshRCSDataset, get_logger, get_model_memory, psnr, ssim, find_matching_files, process_files
from NNvalfast import plotRCS2, plot2DRCS

# 分布式设置
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    setup(rank, world_size)

    # 设置设备
    device = torch.device(f'cuda:{rank}')
    start_time0 = time.time()
    print('代码开始时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

    # 参数设置
    batchsize = 10
    epoch = 200
    use_preweight = True
    cudadevice = 'cuda:1'
    threshold = 20
    learning_rate = 0.001
    lr_time = 20
    shuffle = False
    multigpu = False

    bestloss = 100000
    epoch_loss1 = 0.
    in_ems = []
    rcss = []
    losses = []
    psnr_list = []
    ssim_list = []
    mse_list = []
    psnrs = []
    ssims = []
    mses = []

    rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul26_MieOpt_test100'
    pretrainweight = r'./output/train/0529upconv4ffc_MieOptpretrain3/last.pt'

    save_dir = str(increment_path(Path(ROOT / "output" / "train" / '0530upconv4ffc_mul26test100_'), exist_ok=False))
    lastsavedir = os.path.join(save_dir, 'last.pt')
    bestsavedir = os.path.join(save_dir, 'best.pt')
    lossessavedir = os.path.join(save_dir, 'loss.png')
    psnrsavedir = os.path.join(save_dir, 'psnr.png')
    ssimsavedir = os.path.join(save_dir, 'ssim.png')
    msesavedir = os.path.join(save_dir, 'mse.png')
    logdir = os.path.join(save_dir, 'log.txt')
    logger = get_logger(logdir)

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
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size / (1024 * 1024 * 1024)
    logger.info(f"数据集文件夹大小(内存占用)：{total_size_mb:.2f} MB 或 {total_size_gb:.2f} GB")

    dataset = meshRCSDataset(in_ems, rcss)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=0, sampler=train_sampler)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else "cpu")
    logger.info(f'device:{device}')

    autoencoder = MeshAutoencoder(
        num_discrete_coors = 128,
        device = device
    ).to(device)
    autoencoder = DDP(autoencoder, device_ids=[rank])

    get_model_memory(autoencoder, logger)

    if use_preweight:
        autoencoder.load_state_dict(torch.load(pretrainweight), strict=False)
        logger.info(f'成功加载预训练权重{pretrainweight}')
    else:
        logger.info('未使用预训练权重，为从头训练')

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_time)

    flag = 1
    GTflag = 1
    for i in range(epoch):
        epoch_loss = 0.
        timeepoch = time.time()
        train_sampler.set_epoch(i)
        for in_em1, rcs1 in tqdm(dataloader, desc=f'epoch:{i+1},datasets进度,lr={scheduler.get_last_lr()[0]:.5f}', ncols=130, postfix=f'上一轮的epoch:{i},loss_mean:{(epoch_loss1/dataset.__len__()):.4f}'):
            optimizer.zero_grad()
            objlist, ptlist = find_matching_files(in_em1[0], "./planes")
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
                loss = loss.sum()
                loss.backward()
            else:
                loss.backward()
            epoch_loss += loss.item()
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
        if GTflag == 1:
            outGTpngpath = os.path.join(save_dir, f'theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_GT.png')
            out2DGTpngpath = os.path.join(save_dir, f'theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_2DGT.png')
            plotRCS2(rcs=drawGT, savedir=outGTpngpath, logger=logger)
            plot2DRCS(rcs=drawGT, savedir=out2DGTpngpath, logger=logger)
            GTflag = 0
            logger.info('已画GT图')