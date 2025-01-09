import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from net.jxtnet_Transupconv import MeshEncoderDecoder
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
    plane_obj_dir = "/mnt/SrvUserDisk/JiangXiaotian/workspace/3DEM/planes"

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

    # XOZ平面
    # theta_range = range(0, 45, 1)  # theta 从 0° 到 180°，步长为 1°
    theta_range = range(0, 181, 1)  # theta 从 0° 到 180°，步长为 1°
    phi_values = [0, 180]  # phi 固定为 0 和 180

    # 推理 RCS 值
    rcs_data = {phi: [] for phi in phi_values}
    for phi in phi_values:
        for theta in tqdm(theta_range, desc=f"推理 phi={phi}", ncols=70):
            # 构造输入
            in_em = [[plane1], torch.tensor([theta]), torch.tensor([phi]), torch.tensor([f])]
            with torch.no_grad():
                rcs_value = autoencoder(
                    vertices=planesur_verts,
                    faces=planesur_faces,
                    geoinfo=geoinfo,
                    in_em=in_em,
                    device=device,
                    lgrcs=False,
                )

            rcs_value = rcs_value.mean()  # 提取 RCS 值
            rcs_data[phi].append(rcs_value)

    # 转换为 dBsm
    rcs_data_db = {
        phi: 10 * np.log10(np.maximum(rcs, 1e-10)) for phi, rcs in rcs_data.items()
    }  # 避免 log(0)

    # 准备绘图数据
    alpha_values_0 = np.deg2rad(theta_range)  # phi = 0
    alpha_values_180 = np.deg2rad(360 - np.array(theta_range))  # phi = 180
    rcs_values_0 = rcs_data_db[0]
    rcs_values_180 = rcs_data_db[180]

    alpha_values = np.concatenate((alpha_values_0, alpha_values_180))
    rcs_values = np.concatenate((rcs_values_0, rcs_values_180))

    # 排序数据
    sorted_idx = np.argsort(alpha_values)
    alpha_values = alpha_values[sorted_idx]
    rcs_values = rcs_values[sorted_idx]

    # 闭合曲线
    alpha_values = np.append(alpha_values, alpha_values[0])
    rcs_values = np.append(rcs_values, rcs_values[0])

    # 绘制极坐标图
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')
    ax.plot(alpha_values, rcs_values, label=f'f = {f} GHz', color='skyblue')

    ax.set_title("Characteristic Curve of RCS", fontsize=14)
    ax.set_theta_zero_location('N')  # 0° 在顶部
    ax.set_theta_direction(-1)       # 顺时针方向
    ax.set_rlabel_position(225)      # 角度标签位置
    ax.set_ylabel(r"$\sigma$ (dBsm)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # 参数配置
    device = torch.device("cuda:0")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = "/mnt/SrvUserDisk/JiangXiaotian/workspace/3DEM/output/train/1222_pretrain_bb7c_seed77_maxloss0.0005_cuda:0_p1818s6923/last.pt"  # 网络权重路径
    f = 0.15  # 指定频率 (GHz)

    save_path = "./rcs_characteristic_xozcurve.png"  # 保存路径

    # 指定飞机型号
    plane1 = "bb7c"  # 在此指定飞机型号

    # 生成 RCS 特性曲线
    generate_rcs_curve(device, plane1, weight, f, save_path)