# from NNval_GNN4fold1toAllvaltrain import plot_scatter_separate
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

# 方案 1: 使用curve_fit进行拟合（光滑递增曲线）
def smooth_increasing_curve(x, a, b, c):
    # 用一个简单的二次方程来拟合递增的曲线
    return a * x**2 + b * x + c

def plot_scatter_separate(freq_list, mse_list, psnr_list, ssim_list, savedir):
    # 绘制MSE图，添加95%上界的水平线
    plt.figure(figsize=(10, 6))
    plt.scatter(freq_list, mse_list, alpha=0.5)
    plt.xlabel('Frequency (GHz)', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('MSE vs Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0.1, 1.0)

    # 计算95%上界的MSE值
    mse_sorted = np.sort(mse_list)
    upper_5 = np.percentile(mse_sorted, 95)

    # 画水平线表示95%上界
    plt.axhline(y=upper_5, color='darkseagreen', linestyle='--', label=f'95% Upper Bound (MSE: {upper_5:.4f})')

    plt.legend(loc='best')
    plt.savefig(os.path.join(savedir, f'mse_vs_freq.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制PSNR图，拟合曲线并绘制95%和5%的上下界
    plt.figure(figsize=(10, 6))
    plt.scatter(freq_list, psnr_list, alpha=0.5)
    plt.xlabel('Frequency (GHz)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('PSNR vs Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0.1, 1.0)

    # 使用curve_fit进行拟合
    popt, _ = curve_fit(smooth_increasing_curve, freq_list, psnr_list, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))  # 保证拟合曲线是递增的
    psnr_smooth = smooth_increasing_curve(np.array(freq_list), *popt)  # 生成拟合曲线的值

    # 绘制拟合曲线
    plt.plot(freq_list, psnr_smooth, label='Smooth Fit', color='r', lw=2)

    # # 计算95%上下界
    # psnr_sorted = np.sort(psnr_list)
    # lower_95_psnr = np.percentile(psnr_sorted, 5)
    # upper_5_psnr = np.percentile(psnr_sorted, 95)

    # # 绘制95%上下界
    # plt.fill_between(freq_list, lower_95_psnr, upper_5_psnr, color='grey', alpha=0.3, label=f'95% Bound (PSNR: {lower_95_psnr:.2f} - {upper_5_psnr:.2f})')

    plt.legend(loc='best')
    plt.savefig(os.path.join(savedir, f'psnr_vs_freq.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制SSIM图，拟合曲线并绘制95%和5%的上下界
    plt.figure(figsize=(10, 6))
    plt.scatter(freq_list, ssim_list, alpha=0.5)
    plt.xlabel('Frequency (GHz)', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title('SSIM vs Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0.1, 1.0)

    # 使用curve_fit进行拟合
    popt, _ = curve_fit(smooth_increasing_curve, freq_list, ssim_list, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))  # 保证拟合曲线是递增的
    ssim_smooth = smooth_increasing_curve(np.array(freq_list), *popt)  # 生成拟合曲线的值

    # 绘制拟合曲线
    plt.plot(freq_list, ssim_smooth, label='Smooth Fit', color='r', lw=2)

    # # 计算95%上下界
    # ssim_sorted = np.sort(ssim_list)
    # lower_95_ssim = np.percentile(ssim_sorted, 5)
    # upper_5_ssim = np.percentile(ssim_sorted, 95)

    # # 绘制95%上下界
    # plt.fill_between(freq_list, lower_95_ssim, upper_5_ssim, color='grey', alpha=0.3, label=f'95% Bound (SSIM: {lower_95_ssim:.4f} - {upper_5_ssim:.4f})')

    plt.legend(loc='best')
    plt.savefig(os.path.join(savedir, f'ssim_vs_freq.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'散点图已保存至 {savedir}')



freq_list = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_vs_freq.npy')
mse_samples = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_mse_vs_freq.npy')
psnr_samples = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_psnr_vs_freq.npy')
ssim_samples = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_ssim_vs_freq.npy')
save_dir = 'tryfreq4'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
plot_scatter_separate(freq_list, mse_samples, psnr_samples, ssim_samples, save_dir)
