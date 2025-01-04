import numpy as np
import matplotlib.pyplot as plt

def generate_empty_polar_plot(save_path):
    # 创建极坐标图
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')

    # 设置 22.5° 的刻度
    ticks = np.arange(0, 360, 22.5)
    ax.set_xticks(np.deg2rad(ticks))  # 将角度转换为弧度

    # 设置网格
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 设置角度方向和位置
    ax.set_theta_zero_location('N')  # 0° 在顶部
    ax.set_theta_direction(-1)       # 顺时针方向
    # ax.set_rlabel_position(225)      # 角度标签位置
    ax.set_yticklabels([])
    ax.set_ylabel(r"$\sigma$ (dBsm)", fontsize=12, labelpad=30)  # 调整标签位置

    # 保存图像
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    save_path = "./output/empty_polar_plot.png"  # 保存路径
    generate_empty_polar_plot(save_path)