import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from net.jxtnet_Transupconv0 import MeshEncoderDecoder
from net.utils import find_matching_files, process_files
# import re

def generate_rcs_curve(device, lsp, weight, lsf, dirname='./output/rcs360/', date="0000", batch_size=1):
    # 固定飞机模型目录
    plane_obj_dir = "planes"
    os.makedirs(dirname, exist_ok=True)

    # 初始化网络模型
    autoencoder = MeshEncoderDecoder(
        num_discrete_coors=128, decoder_outdim=12, encoder_layer=6, paddingsize=18000
    ).to(device)
    autoencoder.load_state_dict(torch.load(weight), strict=False)
    autoencoder.eval()
    color=['skyblue', 'deepskyblue', 'mediumpurple', 'dodgerblue', 'mediumslateblue', ]
    color=[ 'mediumpurple', 'deepskyblue', 'skyblue', 'dodgerblue', 'mediumslateblue', ]
    for plane1 in lsp:
        for i,f in enumerate(lsf):
            f = float(round(f, 3))
            save_path = dirname + f"{date}_{plane1}_m1753_xoycp_f{f:.3f}.png"

            # 加载飞机模型
            objlist, _ = find_matching_files([plane1], plane_obj_dir)  # 指定飞机型号
            if not objlist:
                raise FileNotFoundError("未找到指定飞机的 obj 文件")
            planesur_faces, planesur_verts, _, geoinfo = process_files(objlist, device)

            # XOY平面
            theta = 90  # theta 固定为 90°
            # phi_values = range(0, 361, 1)  # phi 从 0° 到 360°
            phi_values = range(0, 181, 1)  # phi 从 0° 到 180°


            # 推理 RCS 值
            rcs_data = []
            for phi in tqdm(phi_values, desc="推理 RCS 值", ncols=70):
                # 构造输入
                phi = int(phi)
                # freq = float(f)
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
            ax.plot(alpha_values, full_rcs_data_db, color=color[i])
            # ax.plot(alpha_values, full_rcs_data_db, label=f'inference output', color='skyblue')

            # 设置图形属性
            ax.set_theta_zero_location('N')  # 0° 在顶部
            ax.set_theta_direction(-1)       # 顺时针方向
            ax.set_rlabel_position(0)      # 角度标签位置
            ax.set_ylabel(r"$\sigma$ (dBsm)", fontsize=12, labelpad=30)  # 调整标签位置
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            # plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.12))  # 调整图例位置
            plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.clf()
            torch.cuda.empty_cache()
        

if __name__ == "__main__":
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")
    # device = torch.device("cpu")
    from datetime import datetime
    date = datetime.today().strftime("%m%d")
    # weight = "output/1218bb7c50train_m0989.pt"  # 网络权重路径
    weight = "output/bb7cm1753.pt"  # 网络权重路径
    lsp = ['bb7c']
    # lsf = [0.11,0.12,0.13,0.14,0.16,0.17]  # 指定频率列表 (GHz)
    # lsf = [0.1,0.105,0.11,0.4,0.5,0.6,0.9,1,]
    # lsf = [0.1,0.130,0.160]
    lsf = [0.1,0.3,0.5]
    # lsf = np.arange(0.1, 0.3, 0.005)
    dirname = f'./output/rcs360/{date}sweep7/'
    # for plane1 in lsp:
    #     for f in lsf:
    #         f = float(round(f, 3))
            # save_path = f"./output/rcs360/0113sweep/{date}_{plane1}_m1753_xoyGTcp_f{f:.3f}.png"
    generate_rcs_curve(device, lsp, weight, lsf, dirname, date)