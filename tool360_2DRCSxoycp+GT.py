import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from net.jxtnet_Transupconv0 import MeshEncoderDecoder
from net.utils import find_matching_files, process_files
import re

def generate_rcs_curve(device, plane1, weight, f, save_path, batch_size=1):
    """
    生成 RCS 特性曲线。

    Args:
        device (torch.device): PyTorch 设备。
        plane_model (str): 飞机型号。
        weight (str): 神经网络权重路径。
        f (float): 入射频率 (GHz)。
        save_path (str): 保存绘图结果的路径。
        batch_size (int): 批量大小。
    """
    # 固定飞机模型目录
    plane_obj_dir = "planes"

    # 初始化网络模型
    autoencoder = MeshEncoderDecoder(
        num_discrete_coors=128, decoder_outdim=12, encoder_layer=6, paddingsize=18000
    ).to(device)
    autoencoder.load_state_dict(torch.load(weight), strict=False)
    autoencoder.eval()

    # 加载飞机模型
    objlist, _ = find_matching_files([plane1], plane_obj_dir)  # 指定飞机型号
    if not objlist:
        raise FileNotFoundError("未找到指定飞机的 obj 文件")
    planesur_faces, planesur_verts, _, geoinfo = process_files(objlist, device)

    # XOY平面
    theta = 90  # theta 固定为 90°
    # phi_values = range(0, 361, 1)  # phi 从 0° 到 360°
    phi_values = range(0, 181, 1)  # phi 从 0° 到 180°

    # 加载真值和验证值
    # trainGTdir = '/home/jiangxiaotian/datasets/pbb7c_mie_train'
    trainGTdir = '/home/jiangxiaotian/datasets/pbb7c_mie_pretrain'
    valGTdir = '/home/jiangxiaotian/datasets/pbb7c_mie_val'
    GTtrain = [None] * 361  # 初始化训练真值列表
    GTval = [None] * 361    # 初始化验证真值列表

    # 提取训练真值
    for file in os.listdir(trainGTdir):
        if '.pt' in file:
            match = re.search(r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d.+).pt", file)
            if match:
                plane1_gt, theta1, phi1, freq1 = match.groups()
                theta1 = int(theta1)
                phi1 = int(phi1)
                freq1 = float(freq1)
                # if file == ''
                if freq1 == f and theta1 == theta and phi1 in phi_values:
                    rcs1 = torch.load(os.path.join(trainGTdir, file))
                    GTtrain[phi1] = rcs1.mean().item()

    # 提取验证真值
    for file in os.listdir(valGTdir):
        if '.pt' in file:
            match = re.search(r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d.+).pt", file)
            if match:
                plane1_gt, theta1, phi1, freq1 = match.groups()
                theta1 = int(theta1)
                phi1 = int(phi1)
                freq1 = float(freq1)
                if freq1 == f and theta1 == theta and phi1 in phi_values:
                    rcs1 = torch.load(os.path.join(valGTdir, file))
                    GTval[phi1] = rcs1.mean().item()

    # 推理 RCS 值
    rcs_data = []
    for phi in tqdm(phi_values, desc="推理 RCS 值", ncols=70):
        # 构造输入
        phi = int(phi)
        freq = float(f)

        # 创建输入列表
        in_em = [[plane1], torch.tensor([theta]), torch.tensor([phi-90]), torch.tensor([f])]

        with torch.no_grad():
            rcs_value = autoencoder(
                vertices=planesur_verts,
                faces=planesur_faces,
                geoinfo=geoinfo,
                in_em=in_em,  # 使用 plane_model 作为字符串
                device=device,
                lgrcs=False,
            )

        rcs_value = rcs_value.mean()  # 提取 RCS 值
        # rcs_value = rcs_value.squeeze(0)[theta*2,phi*2]
        rcs_data.append(rcs_value.item())

    # 转换为 dBsm
    rcs_data_db = 10 * np.log10(np.maximum(rcs_data, 1e-10))  # 避免 log(0)
    # # 准备绘图数据
    full_rcs_data_db = np.zeros(361)  # 初始化完整的 RCS 数据数组
    full_rcs_data_db[:181] = rcs_data_db  # 前 180 度的 RCS 数据
    full_rcs_data_db[180:] = rcs_data_db[::-1]  # 后 180 度的 RCS 数据对称

    # 准备绘图数据
    # alpha_values = np.deg2rad(np.array(phi_values))  # 将 phi 转换为弧度 在这儿+-90
    alpha_values = np.deg2rad(np.arange(0, 361))  # 0 到 360 度的角度 这里-90？

    # 绘制极坐标图
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')
    ax.plot(alpha_values, full_rcs_data_db, label=f'inference output', color='skyblue')

    # 绘制训练真值
    # [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    # [0.35339200496673584, 0.43261367082595825, 0.1607571542263031, 0.43312886357307434, 0.3538358509540558, 0.27493783831596375, 0.4307756721973419, 0.5590197443962097, 0.22259481251239777, 0.22227345407009125, 0.4302466809749603]
    # [-4.5174328  -3.63899761 -7.9382969  -3.63382874 -4.51198166 -5.60765486
    # -3.65748831 -2.52572853 -6.52484961 -6.53112402 -3.66282471]
    GTtrain_phi = [phi for phi in phi_values if GTtrain[phi] is not None]
    GTtrain_values = [GTtrain[phi] for phi in GTtrain_phi]
    GTtrain_values_db = 10 * np.log10(np.maximum(GTtrain_values, 1e-10))
    print(GTtrain_phi)
    print(GTtrain_values)
    print(GTtrain_values_db)
    GTtrain_alpha = np.deg2rad(np.array(GTtrain_phi))
    ax.scatter(GTtrain_alpha, GTtrain_values_db, label='train Groundtruth', color='green', s=10)

    # 绘制验证真值
    # [0]
    # [0.2745873034000397]
    # [-5.61319548]
    GTval_phi = [phi for phi in phi_values if GTval[phi] is not None]
    GTval_values = [GTval[phi] for phi in GTval_phi]
    GTval_values_db = 10 * np.log10(np.maximum(GTval_values, 1e-10))
    print(GTval_phi)
    print(GTval_values)
    print(GTval_values_db)
    GTval_alpha = np.deg2rad(np.array(GTval_phi))
    ax.scatter(GTval_alpha, GTval_values_db, label='val Groundtruth', color='red', s=10)

    # 设置图形属性
    ax.set_theta_zero_location('N')  # 0° 在顶部
    ax.set_theta_direction(-1)       # 顺时针方向
    ax.set_rlabel_position(225)      # 角度标签位置
    ax.set_ylabel(r"$\sigma$ (dBsm)", fontsize=12, labelpad=30)  # 调整标签位置
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.12))  # 调整图例位置
    plt.savefig(save_path)
    plt.show()
    plt.clf()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    from datetime import datetime
    date = datetime.today().strftime("%m%d")
    # weight = "output/1218bb7c50train_m0989.pt"  # 网络权重路径
    weight = "output/bb7cm1753.pt"  # 网络权重路径
    lsp = ['bb7c']
    # lsf = [0.11,0.12,0.13,0.14,0.16,0.17]  # 指定频率列表 (GHz)
    lsf = [0.105,0.11,0.4,0.5,0.6,0.9,1,]
    # lsf = np.arange(0.1, 0.3, 0.05)
    for plane1 in lsp:
        for f in lsf:
            f = float(round(f, 3))
            save_path = f"./output/rcs360/0113sweep/{date}_{plane1}_m1753_xoyGTcp_f{f:.3f}.png"
            generate_rcs_curve(device, plane1, weight, f, save_path)