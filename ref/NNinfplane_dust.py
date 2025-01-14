import torch
import time
from net.jxtnet_Transupconv import MeshEncoderDecoder
from net.utils import increment_path, meshRCSDataset, get_logger, find_matching_files, process_files
import torch.utils.data.dataloader as DataLoader
from pathlib import Path
import sys
import os
from tqdm import tqdm
import re

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def valmain(draw, device, weight, rcsdir, save_dir, logger, epoch, batchsize, trainval=False, draw3d=False,lgrcs=True,decoder_outdim=3,encoder_layer=6,paddingsize=18000):
    if trainval == False:
        logger.info(f'正在用{weight}验证推理{rcsdir}及画图')
    in_ems = []
    rcss = []
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

    autoencoder = MeshEncoderDecoder(num_discrete_coors = 128,decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize).to(device) #这里实例化，是进去跑了init
    autoencoder.load_state_dict(torch.load(weight), strict=False)
    # autoencoder = autoencoder.to(device)
    #-------------------------------------------------------------------------------------
    with torch.no_grad():
        for in_em1 in tqdm(dataloader,desc=f'val进度',ncols=70,postfix=f''):
            objlist , _ = find_matching_files(in_em1[0], "./planes")
            planesur_faces, planesur_verts, _, geoinfo = process_files(objlist, device)

            outrcs = autoencoder( #这里使用网络，是进去跑了forward
                vertices = planesur_verts,
                faces = planesur_faces, #torch.Size([batchsize, 33564, 3])
                # face_edges = planesur_faceedges,
                geoinfo = geoinfo, #[area, volume, scale]
                in_em = in_em1,#.to(device)
                # GT = rcs1.to(device), #这里放真值
                logger = logger,
                device = device,
                lgrcs = lgrcs
            )

            outrcs#这是输出的该入射角和频率下的rcs，然后用于绘制Characteristic curve of RCS


if __name__ == '__main__':

    trainval = False
    cuda = 'cuda:1'
    draw = True
    # draw = False
    draw3d = False
    # draw3d = True
    lgrcs = False
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    batchsize = 1
    encoder_layer = 6
    decoder_outdim = 12

    weight = r'/mnt/SrvUserDisk/JiangXiaotian/workspace/3DEM/2346p2004.pt'

    rcsdir = r'/mnt/SrvDataDisk/Datasets_3DEM/NewPlane6/Pba0f_mie_val'

    from datetime import datetime
    date = datetime.today().strftime("%m%d")
    save_dir = str(increment_path(Path(ROOT / "output" / "inference" /f'{date}_mul2346train50p'), exist_ok=False))
    logdir = os.path.join(save_dir,'alog.txt')
    logger = get_logger(logdir)
    epoch = -1

    valmain(draw, device, weight, rcsdir, save_dir, logger, epoch, batchsize ,trainval, draw3d,lgrcs,decoder_outdim,encoder_layer)

    # baa9 ba0f bb7d bbc6 
    # b827 bb26