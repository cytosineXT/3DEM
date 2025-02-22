import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# 随机生成数据
planes_mse = ['b871', 'bb7d', 'b827', 'b905', 'bbc6', 'b80b', 'ba0f', 'b7c1', 'b9e6', 'bb7c']
mse_values = np.random.rand(len(planes_mse)) * 0.5  # 随机生成 MSE 值
allavemse_train = np.random.rand() * 0.5  # 随机生成训练集平均 MSE
allavemse_val = np.random.rand() * 0.5  # 随机生成验证集平均 MSE

trainplanes = ['b7fd', 'b80b', 'b7c1']  # 随机选择一些训练集飞机
valplanes = ['bb7c', 'b827', 'b905']  # 随机选择一些验证集飞机

# 绘制 MSE 的柱状图
plt.clf()

# 为柱状图排序
sorted_mse = sorted(zip(planes_mse, mse_values), key=lambda x: x[1])  # 从小到大排序
planes_mse_sorted, mse_values_sorted = zip(*sorted_mse)

# 绘制柱状图
bars = plt.bar(planes_mse_sorted, mse_values_sorted, label='MSE per Plane')
plt.axhline(allavemse_train, color='g', linestyle='--', label=f'Average Train MSE ({allavemse_train:.4f})')
plt.axhline(allavemse_val, color='b', linestyle='--', label=f'Average Val MSE ({allavemse_val:.4f})')
plt.xlabel('Plane')
plt.ylabel('MSE')
plt.title('Validation MSE per Plane')

# 图例的设置
legend_elements = [
    Line2D([0], [0], color='g', lw=2, linestyle='--', label=f'Average Train MSE ({allavemse_train:.4f})'),
    Line2D([0], [0], color='b', lw=2, linestyle='--', label=f'Average Val MSE ({allavemse_val:.4f})'),
    Rectangle((0, 0), 0.9, 0.5, color='g', label='Train Plane'),  # 长方形表示训练集，宽1，高0.5
    Rectangle((0, 0), 0.9, 0.5, color='b', label='Val Plane')  # 长方形表示验证集，宽1，高0.5
]

plt.legend(handles=legend_elements, loc='best')

# 为训练集和验证集的柱子分别着色
for i, bar in enumerate(bars):
    if planes_mse_sorted[i] in trainplanes:
        bar.set_color('g')  # 训练集柱子为绿色
    else:
        bar.set_color('b')  # 验证集柱子为蓝色

    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom', rotation=45)

# 设置 y 轴范围，避免文本被裁剪
plt.ylim(0, max(mse_values_sorted) * 1.2)

# 布局调整，避免图例超出图像范围
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# 显示图表
plt.savefig('/home/jiangxiaotian/workspace/3DEM/try.png')
