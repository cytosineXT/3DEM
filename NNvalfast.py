import torch
import time
from net.jxtnet_upConv3_KAN_TV import MeshAutoencoder
# from net.jxtnet_upConv3 import MeshAutoencoder
# from net.jxtnet_upConv2_piloss import MeshAutoencoder
# from net.jxtnet_upConv_deendocer import MeshAutoencoder
from net.utils import increment_path, meshRCSDataset, get_logger
import torch.utils.data.dataloader as DataLoader
import trimesh
from pathlib import Path
import sys
import os
from tqdm import tqdm
import re

cuda = 'cuda:1'
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def inference(weight, in_obj, in_em, GT, logger,device):
    start_time0 = time.time()
    loss, outrcs, _, psnrlist, _, ssimlist, mse = autoencoder( #这里使用网络，是进去跑了forward
        vertices = planesur_vert,
        faces = planesur_face, #torch.Size([batchsize, 33564, 3])
        face_edges = planesur_faceedge,
        in_em = in_em.to(device),
        GT = GT.to(device),
        logger = logger,
        device = device
    )
    # torch.cuda.empty_cache()
    logger.info(f'\n推理用时：{time.time()-start_time0:.4f}s')
    return loss, outrcs, psnrlist, ssimlist, mse

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
    logger.info(f'画图用时：{time.time()-tic:.4f}s')

def plot2DRCS(rcs, savedir,logger):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    tic = time.time()
    print(rcs.shape)
    vmin = torch.min(rcs)
    vmax = torch.max(rcs)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.jet

    plt.figure()
    plt.imshow(rcs, cmap=cmap, norm=norm, origin='lower')
    plt.colorbar(label='RCS/m²')
    plt.xlabel("Theta")
    plt.ylabel("Phi")
    plt.savefig(savedir)
    logger.info(f'画图用时：{time.time()-tic:.4f}s')

if __name__ == '__main__':
    tic = time.time()

    # weight = r'./output/test/0509upconv2_b827_001lr6/best2w.pt'
    # weight = r'./output/test/0514upconv2_b827_10/last.pt'
    weight = r'./output/train/0521upconv3kan_b827_xiezhen2/best.pt'
    # rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_xiezhen_ctrl9090_val'
    # rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_test10'
    # rcsdir = r'/mnt/Disk/jiangxiaotian/datasets/b827_xiezhen_small'
    rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_xiezhen_val'
    in_obj = 'b82731071bd39b66e4c15ad8a2edd2e'

    save_dir = str(increment_path(Path(ROOT / "output" / "inference" /'0522_upconv3kan_xiezhenval'), exist_ok=False))
    # pngsavedir = os.path.join(save_dir,'0508_b827_theta90phi330freq0.9_4w_sm.png')
    logdir = os.path.join(save_dir,'alog.txt')
    logger = get_logger(logdir)

    logger.info(f'正在用{weight}验证推理{rcsdir}及画图')

    in_ems = []
    rcss = []
    psnrs = []
    ssims = []
    mses = []
    losses = []
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
        # rcss.append(rcs[:,:,0])

    dataset = meshRCSDataset(in_ems, rcss)
    dataloader = DataLoader.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    #-------------------------------------------------------------------------------------
    logger.info(f'device:{device}')
    loadobj = os.path.join(ROOT/'planes'/f'{in_obj}.obj')
    loadpt = os.path.join(ROOT/'planes'/f'{in_obj}.pt')
    mesh = trimesh.load_mesh(loadobj)
    planesur_face = torch.tensor(mesh.faces,dtype=int).unsqueeze(0).to(device)
    planesur_vert = torch.tensor(mesh.vertices,dtype=torch.float32).unsqueeze(0).to(device)
    planesur_faceedge = torch.load(loadpt).to(device)
    logger.info(f"物体：{loadobj}， verts={planesur_vert.shape}， faces={planesur_face.shape}， edge={planesur_faceedge.shape}")
    autoencoder = MeshAutoencoder(num_discrete_coors = 128) #这里实例化，是进去跑了init
    autoencoder.load_state_dict(torch.load(weight), strict=False)
    autoencoder = autoencoder.to(device)
    #-------------------------------------------------------------------------------------
    with torch.no_grad():
        for in_em1,rcs1 in tqdm(dataloader,desc=f'datasets进度',ncols=130,postfix=f''):
            loss, outrcs, psnrlist, ssimlist, mse = inference( #这里使用网络，是进去跑了forward
                weight = weight,
                in_obj= in_obj,
                in_em = in_em1,
                GT = rcs1.to(device), #这里放真值
                logger=logger,
                device=device
            )
            eminfo = [int(in_em1[0][0]), int(in_em1[0][1]), float(f'{in_em1[0][2]:.3f}')]
            logger.info(f'em={eminfo}, loss={loss:.4f}')
            # torch.cuda.empty_cache()
            outrcs = outrcs.squeeze()
            rcs1 = rcs1.squeeze()
            outrcspngpath = os.path.join(save_dir,f'{in_obj[:4]}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]}.png')
            out2Drcspngpath = os.path.join(save_dir,f'{in_obj[:4]}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]}_psnr{psnrlist.item():.2f}_ssim{ssimlist.item():.4f}_mse{mse:.4f}_2D.png')
            outGTpngpath = os.path.join(save_dir,f'{in_obj[:4]}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]}_GT.png')
            out2DGTpngpath = os.path.join(save_dir,f'{in_obj[:4]}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]}_2DGT.png')
            logger.info(out2Drcspngpath)
            # plotRCS2(rcs=outrcs, savedir=outrcspngpath, logger=logger) #ValueError: operands could not be broadcast together with shapes (1,361,720) (1,361)
            # plot2DRCS(rcs=outrcs, gesavedir=out2Drcspngpath, logger=logger) #ValueError: operands could not be broadcast together with shapes (1,361,720) (1,361)
            # plotRCS2(rcs=rcs1, savedir=outGTpngpath, logger=logger) #r'./output/inference/b827_theta90phi330freq0.9GT_1w4weight.png'
            # plot2DRCS(rcs=rcs1, savedir=out2DGTpngpath, logger=logger) #r'./output/inference/b827_theta90phi330freq0.9GT_1w4weight.png'
            torch.cuda.empty_cache()
            losses.append(loss)
            psnrs.append(psnrlist.item())
            ssims.append(ssimlist.item())
            mses.append(mse.item())
        ave_loss = sum(losses)/len(losses)
        ave_psnr = sum(psnrs)/len(psnrs)
        ave_ssim = sum(ssims)/len(ssims)
        ave_mse = sum(mses)/len(mses)
        logger.info(f"已用{weight}验证{len(losses)}个数据, Mean Loss: {ave_loss:.4f}, Mean PSNR: {ave_psnr:.2f}dB, Mean SSIM: {ave_ssim:.4f}, Mean MSE:{ave_mse:.4f}")
        logger.info(f'val数据集地址:{rcsdir}, 总耗时:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')