import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from net.jxtnet_Transupconv import MeshEncoderDecoder
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
    phi_values = range(0, 361, 1)  # phi 从 0° 到 360°
    # phi_values = range(0, 10, 1)  # 测试用
    # # /home/ljm/workspace/datasets/mulbb7c_mie_train
    # # /home/ljm/workspace/datasets/mulbb7c_mie_pretrain
    # # /home/ljm/workspace/datasets/mulbb7c_mie_val

    # # /home/jiangxiaotian/datasets/pbb7c_mie_train
    # # /home/jiangxiaotian/datasets/pbb7c_mie_pretrain 用这个作为真值
    # # /home/jiangxiaotian/datasets/pbb7c_mie_val 用这个作为验证值
    # # 要提取出(theta,phi)符合的所有数据，然后处理成对应点的RCS值rcsGT_value = rcsGT_value.mean()，然后再在对应phi处记录该点的真值（可以以列表的形式，有GT值的点把GT值填入，没有GT值的点用空列表代替），最后再画到同一幅图上，训练真值和验证真值用两种颜色标出
    # trainGTdir = '/home/jiangxiaotian/datasets/pbb7c_mie_pretrain'
    # valGTdir = '/home/jiangxiaotian/datasets/pbb7c_mie_val'
    # GTtrain = list(360)
    # GTval = list(360)
    # for file in os.listdir(trainGTdir):
    #     if '.pt' in file:
    #         # print(file)
    #         plane1, theta1, phi1, freq1= re.search(r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d.+).pt", file).groups()
    #         theta1 = int(theta)
    #         phi1 = int(phi)
    #         freq1 = float(freq)
    #         if freq1==f and theta1==theta and phi1 in phi_values:
    #             rcs1 = torch.load(os.path.join(trainGTdir,file))
    #             GTtrain[phi1]=rcs1.mean()
    # for file in os.listdir(valGTdir):
    #     if '.pt' in file:
    #         # print(file)
    #         plane1, theta1, phi1, freq1= re.search(r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d.+).pt", file).groups()
    #         theta1 = int(theta)
    #         phi1 = int(phi)
    #         freq1 = float(freq)
    #         if freq1==f and theta1==theta and phi1 in phi_values:
    #             rcs1 = torch.load(os.path.join(valGTdir,file))
    #             GTval[phi1]=rcs1.mean()

    # 推理 RCS 值
    rcs_data = []
    for phi in tqdm(phi_values, desc="推理 RCS 值", ncols=70):
        # 构造输入
        phi = int(phi)
        freq = float(f)

        # 创建输入列表
        in_em = [[plane1], torch.tensor([theta]), torch.tensor([phi]), torch.tensor([f])]

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
        rcs_data.append(rcs_value)

    # 转换为 dBsm
    rcs_data_db = 10 * np.log10(np.maximum(rcs_data, 1e-10))  # 避免 log(0)

    # 准备绘图数据
    alpha_values = np.deg2rad(np.array(phi_values)-90)  # 将 phi 转换为弧度
    # alpha_values = np.deg2rad(np.array(phi_values) + 90)  # 将 phi 转换为弧度并加90度

    # 绘制极坐标图
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')
    ax.plot(alpha_values, rcs_data_db, label=f'f = {f} GHz', color='skyblue')

    # ax.set_title("Characteristic Curve of RCS", fontsize=14)
    ax.set_theta_zero_location('N')  # 0° 在顶部
    ax.set_theta_direction(-1)       # 顺时针方向
    ax.set_rlabel_position(225)      # 角度标签位置
    # ax.set_ylabel(r"$\sigma$ (dBsm)", fontsize=12)
    ax.set_ylabel(r"$\sigma$ (dBsm)", fontsize=12, labelpad=30)  # 调整标签位置
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.legend(loc='upper right', fontsize=10)
    plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1,1))  # 调整图例位置
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    weight = "output/bb7cm1753.pt"  # 网络权重路径
    lsp = ['bb7c']
    # lsp = ['bb7c','b943','b979','baa9','b7fd']
    lsf = [0.15]  # 指定频率列表 (GHz)
    # lsf = [0.001, 0.01, 0.1, 0.15,  1]  # 指定频率列表 (GHz)
    for plane1 in lsp:
        for f in lsf:
            save_path = f"./output/rcs360/{plane1}_m1753_xoy_f{f}.png"
            generate_rcs_curve(device, plane1, weight, f, save_path)

    # lsf = np.arange(0.01, 0.3, 0.01)
    # lsf = [0.11,0.15,0.16,0.18,0.25,0.3]  # 指定频率列表 (GHz)
    # lsf = [0.001, 0.01, 0.1, 0.15,  1]  # 指定频率列表 (GHz)