import torch
import time
from net.jxtnet_upConv2_piloss import MeshAutoencoder #36000的MLP
from net.utils import increment_path, meshRCSDataset
import torch.utils.data.dataloader as DataLoader
import trimesh
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.colors import Normalize
from pathlib import Path
import sys
import os
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm
import re

cuda = 'cuda:1'
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def inference(weight, in_obj, in_em, GT):
    start_time0 = time.time()
    print(f'device:{device}')

    loadobj = os.path.join(ROOT/'planes'/f'{in_obj}.obj')
    loadpt = os.path.join(ROOT/'planes'/f'{in_obj}.pt')

    mesh = trimesh.load_mesh(loadobj)
    planesur_face = torch.tensor(mesh.faces,dtype=int).unsqueeze(0).to(device)
    planesur_vert = torch.tensor(mesh.vertices,dtype=torch.float32).unsqueeze(0).to(device)
    planesur_faceedge = torch.load(loadpt).to(device)
    print(f"物体：{loadobj}， verts={planesur_vert.shape}， faces={planesur_face.shape}， edge={planesur_faceedge.shape}， 入射角{in_em}")

    autoencoder = MeshAutoencoder(num_discrete_coors = 128) #这里实例化，是进去跑了init
    autoencoder.load_state_dict(torch.load(weight), strict=False)
    autoencoder = autoencoder.to(device)

    loss, outrcs = autoencoder( #这里使用网络，是进去跑了forward
        vertices = planesur_vert,
        faces = planesur_face, #torch.Size([batchsize, 33564, 3])
        face_edges = planesur_faceedge,
        in_em = in_em.to(device),
        GT = GT.to(device)
    )
    # torch.cuda.empty_cache()
    print(f'推理用时：{time.time()-start_time0:.4f}s')
    return loss, outrcs

# def plotRCS(rcs,savedir):
#     start_time0 = time.time()
#     # rcs = torch.load(r'/mnt/Disk/jiangxiaotian/datasets/RCS_mapsmall/RCSmap_theta90phi330f0.9.pt')[:,:,0]
#     print(rcs.shape)

#     theta = np.linspace(0, 2 * np.pi, rcs.shape[1])  # 调整 shape[1]
#     phi = np.linspace(0, np.pi, rcs.shape[0])  # 调整 shape[0]
#     theta, phi = np.meshgrid(theta, phi)

#     x = rcs * np.sin(phi) * np.cos(theta)
#     y = rcs * np.sin(phi) * np.sin(theta)
#     z = rcs * np.cos(phi)

#     vmin = torch.min(rcs)
#     vmax = torch.max(rcs)
#     norm = Normalize(vmin=vmin, vmax=vmax)
#     cmap = cm.jet
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cmap(norm(rcs)), linewidth=0, antialiased=False)
#     ax.view_init(elev=30, azim=45)
#     fig.subplots_adjust(left=0, right=0.9, bottom=0.1, top=0.9)

#     cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # 调整位置和大小
#     cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.5, aspect=10, cax=cax)
#     cb.set_ticks([vmin, vmax])  # 设置colorbar显示的刻度值为真实值范围
#     cb.set_label('RCS/m²')  # 设置colorbar标签

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     plt.savefig(savedir)
#     # plt.show()
#     end_time0 = time.time()
#     print('画图用时：', time.strftime("%H:%M:%S", time.gmtime(end_time0-start_time0)))

def plotRCS2(rcs,savedir):
    tic = time.time()
    # rcs = torch.load('/mnt/Disk/jiangxiaotian/datasets/RCS_mapsmall/RCSmap_theta90phi330f0.9.pt')[:,:,0]
    print(rcs.shape)
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
    print(f'画图用时{time.time()-tic:.4f}s')

if __name__ == '__main__':
    tic = time.time()

    weight = r'./output/test/0509upconv2_b827_001lr6/best2w.pt'
    # rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_xiezhen_val'
    rcsdir = r'/mnt/Disk/jiangxiaotian/puredatasets/b827_test10'
    in_obj = 'b82731071bd39b66e4c15ad8a2edd2e'

    print(f'正在用{weight}验证推理{rcsdir}及画图')
    save_dir = str(increment_path(Path(ROOT / "output" / "inference" /'0509out'), exist_ok=False))
    # pngsavedir = os.path.join(save_dir,'0508_b827_theta90phi330freq0.9_4w_sm.png')

    in_ems = []
    rcss = []
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
    dataloader = DataLoader.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for in_em1,rcs1 in tqdm(dataloader,desc=f'datasets进度',ncols=130,postfix=f''):
        loss, outrcs = inference( #这里使用网络，是进去跑了forward
            weight = weight,
            in_obj= in_obj,
            in_em = in_em1,
            GT = rcs1.to(device) #这里放真值
        )
        eminfo = [int(in_em1[0][0]), int(in_em1[0][1]), float(f'{in_em1[0][2]:.3f}')]
        # torch.cuda.empty_cache()
        outrcs = outrcs.squeeze()
        rcs1 = rcs1.squeeze()
        outrcspngpath = os.path.join(save_dir,f'{in_obj[:4]}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]}_weight{weight[-5:]}.png')
        outGTpngpath = os.path.join(save_dir,f'{in_obj[:4]}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]}_weight{weight[-5:]}_GT.png')
        plotRCS2(rcs=outrcs, savedir=outrcspngpath) #ValueError: operands could not be broadcast together with shapes (1,361,720) (1,361)
        plotRCS2(rcs=rcs1, savedir=outGTpngpath) #r'./output/inference/b827_theta90phi330freq0.9GT_1w4weight.png'

    print(f'val:{rcsdir},总耗时:{time.strftime(time.gmtime(time.time()-tic))}s')