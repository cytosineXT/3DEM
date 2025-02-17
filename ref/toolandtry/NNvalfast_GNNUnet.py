import torch
import time
from net.jxtnet_GNNUnet import MeshEncoderDecoder
# from net.jxtnet_Transupconv import MeshEncoderDecoder
# from net.jxtnet_pureTrans import MeshEncoderDecoder
from net.utils import increment_path, meshRCSDataset, get_logger, find_matching_files, process_files
import torch.utils.data.dataloader as DataLoader
# import trimesh
from pathlib import Path
import sys
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def plotRCS2(rcs,savedir,logger):
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    tic = time.time()
    # rcs = torch.load('/mnt/Disk/jiangxiaotian/datasets/RCS_mapsmall/RCSmap_theta90phi330f0.9.pt')[:,:,0]
    # print(rcs.shape)
    rcs_np = rcs.detach().cpu().numpy()
    npmax = np.max(rcs_np)
    npmin = np.min(rcs_np)
    theta = np.linspace(0, 2 * np.pi, rcs_np.shape[1])
    phi = np.linspace(0, np.pi, rcs_np.shape[0])
    theta, phi = np.meshgrid(theta, phi)

    x = rcs_np * np.sin(phi) * np.cos(theta)
    y = rcs_np * np.sin(phi) * np.sin(theta)
    z = rcs_np * np.cos(phi)

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, cmin = 0, cmax = npmax,  surfacecolor=rcs_np, colorscale='Jet', colorbar=dict(exponentformat='E',title=dict(side='top',text="RCS/m²"), showexponent='all', showticklabels=True, thickness = 30,tick0 = 0, dtick = npmax))])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectratio=dict(x=1, y=1, z=0.8),
            aspectmode="manual",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        )
    )
    # pio.show(fig)
    pio.write_image(fig, savedir)
    # logger.info(f'画图用时：{time.time()-tic:.4f}s')

def plot2DRCS(rcs, savedir,logger,cutmax):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    tic = time.time()
    # print(rcs.shape)
    vmin = torch.min(rcs)
    vmax = torch.max(rcs)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.jet

    plt.figure()
    plt.imshow(rcs, cmap=cmap, norm=norm, origin='lower')
    plt.colorbar(label='RCS/m²')
    if cutmax != None:# 设置图例的上下限
        plt.clim(0, cutmax)
    plt.xlabel("Theta")
    plt.ylabel("Phi")
    plt.savefig(savedir)
    plt.close()
    if logger!=None:
        # logger.info(f'画图用时：{time.time()-tic:.4f}s')
        1
    else:
        print(f'画图用时：{time.time()-tic:.4f}s')

def plotstatistic(psnr_list, ssim_list, mse_list, statisticdir):
    # 绘制统计图
    def to_percent(y,position):
        return str(int((100*y))) #+"%"#这里可以用round（）函数设置取几位小数
    binss = 40

    plt.clf()
    # 设置图像大小和子图
    plt.figure(figsize=(12, 6))

    mse_threshold = 7
    mse_list = [m for m in mse_list if m <= mse_threshold]

    # MSE 直方图和正态分布曲线
    plt.subplot(3, 3, 1)
    # counts, bins, patches = plt.hist(mse_list, bins=binss, edgecolor='black', density=True, stacked=True)
    counts, bins, patches = plt.hist(mse_list, bins=binss, edgecolor='black', density=True)
    fomatter=FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    mu, std = norm.fit(mse_list)
    # x = np.linspace(-5, 15, 1000)
    x = np.linspace(min(mse_list)-2, max(mse_list)+2, 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    # plt.xlim(-5, 15)  # 限制横坐标范围
    plt.xlabel('MSE')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('MSE Histogram and Normal Fit')
    plt.legend()

    # PSNR 直方图和正态分布曲线
    plt.subplot(3, 3, 2)
    # counts, bins, patches = plt.hist(psnr_list, bins=binss, edgecolor='black', density=True, stacked=True)
    counts, bins, patches = plt.hist(psnr_list, bins=binss, edgecolor='black', density=True)
    fomatter=FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    mu, std = norm.fit(psnr_list)
    # x = np.linspace(15,45, 1000)
    x = np.linspace(min(psnr_list)-2, max(psnr_list)+2, 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    # plt.xlim(-5, 15)  # 限制横坐标范围
    plt.xlabel('PSNR')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('PSNR Histogram and Normal Fit')
    plt.legend()

    # SSIM 直方图和正态分布曲线
    plt.subplot(3, 3, 3)
    # counts, bins, patches = plt.hist(ssim_list, bins=binss, edgecolor='black', density=True, stacked=True)
    counts, bins, patches = plt.hist(ssim_list, bins=binss, edgecolor='black', density=True)
    # fomatter=FuncFormatter(to_percent)
    # plt.gca().yaxis.set_major_formatter(fomatter)
    mu, std = norm.fit(ssim_list)
    # x = np.linspace(0.6,1.1, 1000)
    x = np.linspace(min(ssim_list)-0.05, max(ssim_list)+0.05, 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    # plt.xlim(0.55, 1.1)  # 限制横坐标范围
    plt.xlabel('SSIM')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('SSIM Histogram and Normal Fit')
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(statisticdir)
    plt.close()

def plotstatistic2(psnr_list, ssim_list, mse_list, statisticdir):
    # 绘制统计图
    def to_percent(y,position):
        return str(int((100*y))) #+"%"#这里可以用round（）函数设置取几位小数
    # binss0 = 40
    binss = 20

    plt.clf()
    plt.figure(figsize=(12, 6))

    #-----------------------------------mse-------------------------------------------
    mse_threshold = 2
    mse_list = [m for m in mse_list if m <= mse_threshold]
    print(len(mse_list))
    # MSE 直方图和正态分布曲线
    plt.subplot(3, 3, 1)
    # counts, bins, patches = plt.hist(mse_list, bins=binss, edgecolor='black', density=True, stacked=True)
    counts, bins, patches = plt.hist(mse_list, bins=binss, edgecolor='black', range=(0,0.75), density=True)
    # print(f'counts{counts},bins{bins},patches{patches}')

    # fomatter=FuncFormatter(to_percent)#这里把刻度乘了100，为了得到百分比纵轴
    # plt.gca().yaxis.set_major_formatter(fomatter)

    mu, std = norm.fit(mse_list)
    # x = np.linspace(-5, 15, 1000)
    x = np.linspace(min(mse_list)-0.5, max(mse_list)+0.5, 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    plt.xlim(-0.5, 1)  # 限制横坐标范围
    plt.xlabel('MSE')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('MSE Histogram and Normal Fit')
    plt.legend()


    #-----------------------------------PSNR-------------------------------------------
    # PSNR 直方图和正态分布曲线
    plt.subplot(3, 3, 2)
    # counts, bins, patches = plt.hist(psnr_list, bins=binss, edgecolor='black', density=True, stacked=True)
    counts, bins, patches = plt.hist(psnr_list, bins=binss, edgecolor='black', density=True)
    fomatter=FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    mu, std = norm.fit(psnr_list)
    # x = np.linspace(15,45, 1000)
    x = np.linspace(min(psnr_list)-2, max(psnr_list)+2, 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    # plt.xlim(-5, 15)  # 限制横坐标范围
    plt.xlabel('PSNR')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('PSNR Histogram and Normal Fit')
    plt.legend()

    #-----------------------------------SSIM-------------------------------------------
    # SSIM 直方图和正态分布曲线
    plt.subplot(3, 3, 3)
    # counts, bins, patches = plt.hist(ssim_list, bins=binss, edgecolor='black', density=True, stacked=True)
    counts, bins, patches = plt.hist(ssim_list, bins=binss, edgecolor='black', density=True)
    # fomatter=FuncFormatter(to_percent)
    # plt.gca().yaxis.set_major_formatter(fomatter)
    mu, std = norm.fit(ssim_list)
    # x = np.linspace(0.6,1.1, 1000)
    x = np.linspace(min(ssim_list)-0.05, max(ssim_list)+0.05, 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    # plt.xlim(0.55, 1.1)  # 限制横坐标范围
    plt.xlabel('SSIM')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('SSIM Histogram and Normal Fit')
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(statisticdir)
    plt.close()


def valmain(draw, device, weight, rcsdir, save_dir, logger, epoch, batchsize, trainval=False, draw3d=False,lgrcs=True,decoder_outdim=3,encoder_layer=6,paddingsize=18000, n=4, middim=64):
    tic = time.time()
    # pngsavedir = os.path.join(save_dir,'0508_b827_theta90phi330freq0.9_4w_sm.png')
    if trainval == False:
        logger.info(f'正在用{weight}验证推理{rcsdir}及画图')

    in_ems = []
    rcss = []
    psnrs = []
    ssims = []
    mses = []
    losses = []
    corrupted_files = []
    for file in tqdm(os.listdir(rcsdir),desc=f'加载验证数据集',ncols=60,postfix=''):
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
            # rcss.append(rcs[:,:,0])

    dataset = meshRCSDataset(in_ems, rcss)
    dataloader = DataLoader.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)#嗷 这个batchsize只能是1.。不知道啥时候写成batchsize的。。
    #-------------------------------------------------------------------------------------
    if trainval == False:
        logger.info(f'device:{device}')

    autoencoder = MeshEncoderDecoder(num_discrete_coors = 128,decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,n=n).to(device) #这里实例化，是进去跑了init
    autoencoder.load_state_dict(torch.load(weight), strict=False)
    # autoencoder = autoencoder.to(device)
    #-------------------------------------------------------------------------------------
    with torch.no_grad():
        for in_em1,rcs1 in tqdm(dataloader,desc=f'val进度',ncols=70,postfix=f''):
            in_em0 = in_em1.copy()
            objlist , _ = find_matching_files(in_em1[0], "./planes")
            planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device)

            start_time0 = time.time()
            loss, outrcs, _, psnrlist, _, ssimlist, mse = autoencoder( #这里使用网络，是进去跑了forward
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
            # torch.cuda.empty_cache()
            if lgrcs == True:
                outrcs = torch.pow(10, outrcs)
            eminfo = [int(in_em0[1]), int(in_em0[2]), float(in_em0[3])]
            plane = in_em0[0][0]
            outrcs = outrcs.squeeze()
            rcs1 = rcs1.squeeze()
            if trainval == False:
                logger.info(f'推理用时：{time.time()-start_time0:.4f}s')
                logger.info(f'{plane}, em={eminfo}, loss={loss:.4f}')
            # torch.cuda.empty_cache()
            if draw == True:
                save_dir2 = os.path.join(save_dir,f'epoch{epoch}')
                Path(save_dir2).mkdir(exist_ok=True)
                outrcspngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}.png')
                out2Drcspngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_psnr{psnrlist.item():.2f}_ssim{ssimlist.item():.4f}_mse{mse:.4f}_2D.png')
                out2Drcspngpath2 = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_psnr{psnrlist.item():.2f}_ssim{ssimlist.item():.4f}_mse{mse:.4f}_2Dcut.png')
                outGTpngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_GT.png')
                out2DGTpngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_2DGT.png')
                # logger.info(out2Drcspngpath) #查看输出的图片叫啥在哪儿
                plot2DRCS(rcs=outrcs, savedir=out2Drcspngpath, logger=logger,cutmax=None) #预测2D
                plot2DRCS(rcs=outrcs, savedir=out2Drcspngpath2, logger=logger,cutmax=torch.max(rcs1).item()) #预测2D但是带cut
                plot2DRCS(rcs=rcs1, savedir=out2DGTpngpath, logger=logger,cutmax=None) #GT2D
                if draw3d == True:
                    plotRCS2(rcs=rcs1, savedir=outGTpngpath, logger=logger) #GT
                    plotRCS2(rcs=outrcs, savedir=outrcspngpath, logger=logger) #预测
            
            torch.cuda.empty_cache()
            losses.append(loss)
            psnrs.append(psnrlist.mean())
            ssims.append(ssimlist.mean())
            mses.append(mse.mean())
            # psnrs.append(psnrlist.item())
            # ssims.append(ssimlist.item())
            # mses.append(mse.item())
        ave_loss = sum(losses)/len(losses)
        ave_psnr = sum(psnrs)/len(psnrs)
        ave_ssim = sum(ssims)/len(ssims)
        ave_mse = sum(mses)/len(mses)
        if trainval == False:
            logger.info(f"已用{weight}验证{len(losses)}个数据, Mean Loss: {ave_loss:.4f}, Mean PSNR: {ave_psnr:.2f}dB, Mean SSIM: {ave_ssim:.4f}, Mean MSE:{ave_mse:.4f}")
            logger.info(f'val数据集地址:{rcsdir}, 总耗时:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')
            logger.info(f"损坏的文件：{corrupted_files}")
        logger.info(f'val数据集地址:{rcsdir}, 总耗时:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')
        logger.info(f'↑----val loss:{ave_loss:.4f},psnr:{ave_psnr:.2f},ssim:{ave_ssim:.4f},mse:{ave_mse:.4f}----↑')
        # if epoch % 20 == 0 or epoch == -1: #存指定倍数轮的
        #     statisdir = os.path.join(save_dir,f'statistic_epoch{epoch}.png')
        #     plotstatistic(psnrs,ssims,mses,statisdir)

        statisdir = os.path.join(save_dir,f'statistic_epoch{epoch}_PSNR{ave_psnr:.2f}dB_SSIM{ave_ssim:.4f}_MSE:{ave_mse:.4f}_Loss{ave_loss:.4f}.png')
        plotstatistic2(psnrs,ssims,mses,statisdir)
        # plotstatistic(psnrs,ssims,mses,statisdir,ave_loss,ave_psnr,ave_ssim,ave_mse)


if __name__ == '__main__':

    trainval = False
    cuda = 'cuda:0'
    draw = True
    # draw = False
    draw3d = False
    lgrcs = False
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    batchsize = 1
    decoder_outdim = 3
    encoder_layer = 6

    # weight = r'./output/test/0509upconv2_b827_001lr6/best2w.pt'
    # weight = r'./output/test/0514upconv2_b827_10/last.pt'
    # weight = r'./output/train/0605upconv4fckan_mul2347_pretrain3/last.pt'
    weight = r'./output/train/0615upconv4fckan_mul2347pretrain_000/best.pt'
    # rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_xiezhen_ctrl9090_val'
    # rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_test10'
    # rcsdir = r'/mnt/Disk/jiangxiaotian/datasets/b827_xiezhen_small'
    # rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_xiezhen_val'
    rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul2347_6val'
    # rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul2347_6val2'
    # rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/mul2347_train'

    save_dir = str(increment_path(Path(ROOT / "output" / "inference" /'0619_upconv4_mul2347pretrain_val6_'), exist_ok=False))
    logdir = os.path.join(save_dir,'alog.txt')
    logger = get_logger(logdir)
    epoch = -1

    valmain(draw, device, weight, rcsdir, save_dir, logger, epoch, batchsize ,trainval, draw3d,lgrcs,decoder_outdim,encoder_layer)