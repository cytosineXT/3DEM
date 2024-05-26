import torch
import time
from net.jxtnet_upConv_deendocer import MeshAutoencoder #36000的MLP
# from net.jxtnet_padding2w import MeshAutoencoder #21000的MLP
# from net.jxtnet_upConv import MeshAutoencoder #upConv解码器
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path
import sys
import os
import plotly.graph_objects as go
import plotly.io as pio


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def increment_path(path, exist_ok=False, sep="", mkdir=True):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def inference(weight,loadobj,loadpt,in_emm):
    start_time0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')

    mesh = trimesh.load_mesh(loadobj)
    planesur_face = torch.tensor(mesh.faces,dtype=int).unsqueeze(0).to(device)
    planesur_vert = torch.tensor(mesh.vertices,dtype=torch.float32).unsqueeze(0).to(device)
    planesur_faceedge = torch.load(loadpt).to(device)
    print(f"物体：{loadobj}， verts={planesur_vert.shape}， faces={planesur_face.shape}， edge={planesur_faceedge.shape}， 入射角{in_emm}")

    autoencoder = MeshAutoencoder(num_discrete_coors = 128) #这里实例化，是进去跑了init
    autoencoder.load_state_dict(torch.load(weight), strict=False)
    autoencoder = autoencoder.to(device)

    rcs = autoencoder( #这里使用网络，是进去跑了forward
        vertices = planesur_vert,
        faces = planesur_face, #torch.Size([batchsize, 33564, 3])
        face_edges = planesur_faceedge,
        in_em = in_emm.to(device),
        GT = None
    )
    torch.cuda.empty_cache()
    end_time0 = time.time()
    print('推理用时：', time.strftime("%H:%M:%S", time.gmtime(end_time0-start_time0)))
    return rcs

def plotRCS(rcs,savedir):
    start_time0 = time.time()
    # rcs = torch.load(r'/mnt/Disk/jiangxiaotian/datasets/RCS_mapsmall/RCSmap_theta90phi330f0.9.pt')[:,:,0]
    print(rcs.shape)

    theta = np.linspace(0, 2 * np.pi, rcs.shape[1])  # 调整 shape[1]
    phi = np.linspace(0, np.pi, rcs.shape[0])  # 调整 shape[0]
    theta, phi = np.meshgrid(theta, phi)

    x = rcs * np.sin(phi) * np.cos(theta)
    y = rcs * np.sin(phi) * np.sin(theta)
    z = rcs * np.cos(phi)

    vmin = torch.min(rcs)
    vmax = torch.max(rcs)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.jet
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cmap(norm(rcs)), linewidth=0, antialiased=False)
    ax.view_init(elev=30, azim=45)
    fig.subplots_adjust(left=0, right=0.9, bottom=0.1, top=0.9)

    cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # 调整位置和大小
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.5, aspect=10, cax=cax)
    cb.set_ticks([vmin, vmax])  # 设置colorbar显示的刻度值为真实值范围
    cb.set_label('RCS/m²')  # 设置colorbar标签

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(savedir)
    # plt.show()
    end_time0 = time.time()
    print('画图用时：', time.strftime("%H:%M:%S", time.gmtime(end_time0-start_time0)))

def plotRCS2(rcs,savedir):
    tic = time.time()
    rcs = torch.load('/mnt/Disk/jiangxiaotian/datasets/RCS_mapsmall/RCSmap_theta90phi330f0.9.pt')[:,:,0]
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
    print(f'画图用时{time.strftime("%H:%M:%S",time.gmtime(time.time()-tic))}')
    tic = time.time()

if __name__ == '__main__':
    
    save_dir = str(increment_path(Path(ROOT / "output" / "inference" /'0508out'), exist_ok=False))
    pngsavedir = os.path.join(save_dir,'0508_b827_theta90phi330freq0.9_4w_sm.png')
    weight = r'/home/ljm/workspace/jxt/jxtnet/output/test/0508upconv_b827_piloss3/last.pt'
    loadobj = r'./datasets/b82731071bd39b66e4c15ad8a2edd2e.obj'
    loadpt = r'./datasets/b827_edges.pt'
    in_emm = torch.tensor([[90,330,0.9]]) 
    GTdir = r'/mnt/Disk/jiangxiaotian/datasets/RCS_mapsmall/RCSmap_theta90phi330f0.9.pt'

    rcss = inference(
        weight = weight,
        loadobj = loadobj,
        loadpt = loadpt,
        in_emm = in_emm
    )

    rcss = rcss.squeeze().to('cpu')
    plotRCS2(rcs=rcss, savedir=pngsavedir)

    rcsGT=torch.load(GTdir)[:,:,0]
    # plotRCS(rcs=rcsGT, savedir=r'./output/inference/b827_theta90phi330freq0.9GT_1w4weight.png')

    mse_loss = torch.nn.MSELoss(reduction='sum')
    print(f'loss:{mse_loss(rcss, rcsGT):.4f}')