import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from net.jxtnet_Transupconv0 import MeshEncoderDecoder
from net.utils import find_matching_files, process_files

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
    phi_values = range(0, 181, 1)  # phi 从 0° 到 180°

    # 推理 RCS 值
    rcs_data = []
    for phi in tqdm(phi_values, desc="推理 RCS 值", ncols=70):
        # 构造输入
        in_em = [[plane1], torch.tensor([theta]), torch.tensor([phi]), torch.tensor([f])] #发现在这里把phi-90也能转，但是我都没有负值输入，说明真学到东西了！

        with torch.no_grad():
            rcs_value = autoencoder(
                vertices=planesur_verts,
                faces=planesur_faces,
                geoinfo=geoinfo,
                in_em=in_em,  # 使用 plane_model 作为字符串
                device=device,
                lgrcs=False,
            )
        rcs_value = rcs_value.mean() # 提取平均 RCS 值
        # rcs_value = rcs_value.sum() # 提取和 RCS 值
        # rcs_value = rcs_value.squeeze(0)[theta*2,phi*2]  # 提取该方向 RCS 值（单站）
        rcs_data.append(rcs_value)
    rcs_data_db = 10 * np.log10(np.maximum(rcs_data, 1e-10))  # 避免 log(0)

    # # 准备绘图数据
    full_rcs_data_db = np.zeros(361)  # 初始化完整的 RCS 数据数组
    full_rcs_data_db[:181] = rcs_data_db  # 前 180 度的 RCS 数据
    full_rcs_data_db[180:] = rcs_data_db[::-1]  # 后 180 度的 RCS 数据对称

    alpha_values = np.deg2rad(np.arange(0, 361)+90)  # 0 到 360 度的角度 这里-90？

    # 绘制极坐标图
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')
    ax.plot(alpha_values, full_rcs_data_db, label=f'f = {f} GHz', color='skyblue')

    # ax.set_title("Characteristic Curve of RCS", fontsize=14)
    ax.set_theta_zero_location('N')  # 0° 在顶部
    ax.set_theta_direction(-1)       # 顺时针方向
    ax.set_rlabel_position(270)      # 角度标签位置
    ax.set_ylabel(r"$\sigma$ (dBsm)", fontsize=12, labelpad=30)  # 调整标签位置
    ticks = np.arange(0, 360, 22.5)
    ax.set_xticks(np.deg2rad(ticks))  # 将角度转换为弧度
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1))  # 调整图例位置
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    # weight = "output/1218bb7c50train_m0989.pt"  # 网络权重路径
    weight = "output/bb7cm1753.pt"  # 网络权重路径
    lsp = ['bb7c']
    # lsp = ['bb7c','b943','b979','baa9','b7fd']
    # lsf = [0.15]  # 指定频率列表 (GHz)
    # lsf = [0.001, 0.01, 0.1, 0.15,  1]  # 指定频率列表 (GHz)
    for plane1 in lsp:
        for f in lsf:
            save_path = f"./output/rcs360/{plane1}_m1753_xoycp_oldnet_alpha+90_f{f}.png"
            generate_rcs_curve(device, plane1, weight, f, save_path)