import torch
import time
from net.jxtnet_upConv4_silu import MeshAutoencoder
from net.utils import increment_path, meshRCSDataset, get_logger, find_matching_files, process_files
import torch.utils.data.dataloader as DataLoader
# import trimesh
from pathlib import Path
import sys
import os
from tqdm import tqdm
import re

cuda = 'cuda:1'
draw = True
plane = 'b943'
f = 0.1
weight = r'./output/train/0604upconv4fckan_mul2347_pretrain/last.pt'
date = '0605'
NN = 'upconv4fckan'
weightused = 'mul2347'

device = torch.device(cuda if torch.cuda.is_available() else "cpu")

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

def plot1DRCS(data,savedir_1drcs,logger,f):
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    tic = time.time()

    # 转换数据格式用于绘图
    theta_values = sorted(set(theta for theta, phi in data.keys()))
    alpha_values_0 = np.deg2rad(np.array(theta_values) - 0)  # phi=0, theta 0-180 -> alpha -90 to 90
    alpha_values_180 = np.deg2rad(360 - np.array(theta_values))  # phi=180, theta 0-180 -> alpha 90 to 270

    rcs_values_0 = [data[(theta, 0)] for theta in theta_values if (theta, 0) in data]
    rcs_values_180 = [data[(theta, 180)] for theta in theta_values if (theta, 180) in data]

    # 合并alpha和RCS值
    alpha_values = np.concatenate((alpha_values_0, alpha_values_180))
    rcs_values = np.concatenate((rcs_values_0, rcs_values_180))

    # 对 alpha_values 和 rcs_values 进行排序
    sorted_idx = np.argsort(alpha_values)
    sorted_alpha = alpha_values[sorted_idx]
    sorted_rcs = rcs_values[sorted_idx]

    # 在末尾添加第一个数据点,使曲线闭合
    sorted_alpha = np.concatenate((sorted_alpha, [sorted_alpha[0]]))
    sorted_rcs = np.concatenate((sorted_rcs, [sorted_rcs[0]]))

    # 绘制散点图和包围曲线
    plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, projection='polar')
    ax.scatter(alpha_values, rcs_values, marker='o', color='g')
    ax.plot(sorted_alpha, sorted_rcs, color='g', linestyle='-')

    ax.set_title(f'mean total RCS in XoZ plane vs Alpha (f = {f} GHz)')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.grid(True, which="both", ls="--")
    plt.savefig(savedir_1drcs)
    plt.show()

    toc = time.time()
    logger.info(f"绘图用时: {toc - tic:.2f}s")


if __name__ == '__main__':
    tic = time.time()
    save_dir = str(increment_path(Path(ROOT / "output" / "inference" / f'{date}_{NN}_{weightused}_'), exist_ok=False))
    logdir = os.path.join(save_dir,'alog.txt')
    
    logger = get_logger(logdir)

    

    in_ems = []
    rcss = []
    psnrs = []
    ssims = []
    mses = []
    losses = []
    data = {}

    autoencoder = MeshAutoencoder(num_discrete_coors = 128).to(device) #这里实例化，是进去跑了init
    autoencoder.load_state_dict(torch.load(weight), strict=False)

    for f in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        logger.info(f'正在用{weight}验证推理{plane}在{f}GHz以step0.2°XOZ平面polarRCS图')
        with torch.no_grad():
            for phi in [0,180]:
                for theta in tqdm(range(0,180,1)):
                # for theta in tqdm(range(0,180,1)):
                # for theta in tqdm(range(0,180*5,1)):
                    # theta = theta / 5
                    #[('baa9',), tensor([180]), tensor([270]), tensor([5.], dtype=torch.float64)]
                    in_em = [plane,torch.tensor([theta]),torch.tensor([phi]),torch.tensor([f])]
                    objlist , _ = find_matching_files([in_em[0]], "./planes")
                    planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device)
                    start_time0 = time.time()
                    tensor = autoencoder( #这里使用网络，是进去跑了forward
                        vertices = planesur_verts,
                        faces = planesur_faces, #torch.Size([batchsize, 33564, 3])
                        face_edges = planesur_faceedges,
                        geoinfo = geoinfo, #[area, volume, scale]
                        in_em = in_em,#.to(device)
                        GT = None, #这里放真值
                        logger = logger,
                        device = device
                    )
                    # torch.cuda.empty_cache()
                    logger.info(f'\n推理用时：{time.time()-start_time0:.4f}s')

                    tensor[:, 0][tensor[:, 0] == 0] = tensor[:, -1][tensor[:, 0] == 0]
                    rcs_value = tensor.mean().item()  # 假设张量包含一个单一的RCS值

                    key = (theta, phi)
                    data[key] = rcs_value
        savedir_1drcs = os.path.join(save_dir,f'{plane}in{f}GHzpolar1drcs.png')
        plot1DRCS(data,savedir_1drcs,logger,f)

        logger.info(f'推理{plane}在{f}GHz以step0.2°XOZ平面polarRCS图, 总耗时:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')