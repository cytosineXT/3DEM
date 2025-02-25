import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_scatter_separate(freq_list, mse_list, psnr_list, ssim_list, savedir):
    # 创建保存目录
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    # 统一绘图风格
    # plt.style.use('seaborn')
    
    # MSE绘图（保持不变）
    plt.figure(figsize=(10, 6))
    plt.scatter(freq_list, mse_list, alpha=0.7, color='#2c7bb6')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('MSE')
    plt.title('MSE vs Frequency')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.1, 1.0)
    
    # 计算并绘制95%上界
    upper_5_mse = np.percentile(mse_list, 95)
    plt.axhline(upper_5_mse, color='#d7191c', linestyle='--', 
                linewidth=2, label=f'95% Upper Bound ({upper_5_mse:.4f})')
    plt.legend()
    plt.savefig(os.path.join(savedir, 'mse_vs_freq.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 改进的PSNR拟合（使用LOESS算法）
    plt.figure(figsize=(10, 6))
    plt.scatter(freq_list, psnr_list, alpha=0.7, color='#2c7bb6')
    
    # 使用LOESS进行非参数拟合
    psnr_smoothed = lowess(psnr_list, freq_list, 
                          frac=0.3,  # 控制平滑度的关键参数（0-1之间）
                          it=3)      # 迭代次数
    
    # 排序数据以绘制连续曲线
    sort_idx = np.argsort(psnr_smoothed[:, 0])
    x_sorted = psnr_smoothed[sort_idx, 0]
    y_sorted = psnr_smoothed[sort_idx, 1]
    
    plt.plot(x_sorted, y_sorted, color='#fdae61', linewidth=3, 
             label='LOESS Fit')
    
    # 绘制动态置信区间
    for alpha in [0.3, 0.2, 0.1]:  # 多层级透明度
        plt.fill_between(x_sorted, 
                        y_sorted - (1-alpha)*2, 
                        y_sorted + (1-alpha)*2,
                        color='#abd9e9', 
                        alpha=alpha*0.7)
    
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs Frequency with LOESS Fit')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.1, 1.0)
    plt.legend()
    plt.savefig(os.path.join(savedir, 'psnr_vs_freq.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 改进的SSIM拟合（使用多项式回归）
    plt.figure(figsize=(10, 6))
    plt.scatter(freq_list, ssim_list, alpha=0.7, color='#2c7bb6')
    
    # 使用3次多项式拟合
    coeffs = np.polyfit(freq_list, ssim_list, 3)
    poly = np.poly1d(coeffs)
    
    # 生成密集采样点
    x_fit = np.linspace(min(freq_list), max(freq_list), 200)
    y_fit = poly(x_fit)
    
    plt.plot(x_fit, y_fit, color='#d7191c', linewidth=3, 
             label='Cubic Fit')
    
    # 绘制置信带
    y_std = np.std(ssim_list) * 0.8  # 调整标准差缩放系数
    plt.fill_between(x_fit, y_fit - y_std, y_fit + y_std,
                    color='#fdae61', alpha=0.3,
                    label='Confidence Band')
    
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('SSIM')
    plt.title('SSIM vs Frequency with Polynomial Fit')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.1, 1.0)
    plt.legend()
    plt.savefig(os.path.join(savedir, 'ssim_vs_freq.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'可视化结果已保存至: {os.path.abspath(savedir)}')

# 数据加载和函数调用（保持原样）
freq_list = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_vs_freq.npy')
mse_samples = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_mse_vs_freq.npy')
psnr_samples = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_psnr_vs_freq.npy')
ssim_samples = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_ssim_vs_freq.npy')

save_dir = 'try2freq'
plot_scatter_separate(freq_list, mse_samples, psnr_samples, ssim_samples, save_dir)