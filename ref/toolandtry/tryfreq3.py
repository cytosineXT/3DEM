import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import sem

def polynomial_fit(x, y, degree=3, alpha=0.05):
    """执行多项式拟合并返回拟合曲线和置信区间"""
    # 多项式拟合
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    
    # 生成拟合曲线
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = poly(x_fit)
    
    # 计算残差
    residuals = y - poly(x)
    
    # 计算分位数
    lower = np.percentile(residuals, 5)
    upper = np.percentile(residuals, 95)
    
    return x_fit, y_fit, y_fit + lower, y_fit + upper

def plot_metric(ax, x, y, title, ylabel,  color_fit='#D55E00'): #color_point='#4C72B0',
    """通用绘图函数"""
    # 绘制散点
    ax.scatter(x, y, alpha=0.6, edgecolor='white', linewidth=0.5) # color=color_point,
    
    # 执行多项式拟合
    x_fit, y_fit, y_lower, y_upper = polynomial_fit(x, y)
    
    # 绘制拟合曲线
    ax.plot(x_fit, y_fit, color=color_fit, lw=2.5, label='Cubic Fit')
    
    # 绘制置信区间
    ax.fill_between(x_fit, y_lower, y_upper, color='#999999', alpha=0.2, 
                    label='5%-95% Confidence Band')
    
    ax.set_title(title)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 1.0)
    ax.legend()

def plot_scatter_separate(freq_list, mse_list, psnr_list, ssim_list, savedir):
    # 创建保存目录
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    # 设置全局样式
    # plt.style.use('seaborn')
    plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
    
    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # # 绘制MSE
    # axes[0].scatter(freq_list, mse_list, alpha=0.6, edgecolor='white', linewidth=0.5)#color='#4C72B0', 
    # upper_5_mse = np.percentile(mse_list, 95)
    # axes[0].axhline(upper_5_mse, color='#CC0000', linestyle='--', 
    #                linewidth=2, label=f'95% Upper Bound\n({upper_5_mse:.4f})')
    # axes[0].set_title('MSE vs Frequency')
    # axes[0].set_xlabel('Frequency (GHz)')
    # axes[0].set_ylabel('MSE')
    # axes[0].grid(True, alpha=0.3)
    # axes[0].set_xlim(0.1, 1.0)
    # axes[0].legend()

    # 绘制PSNR
    plot_metric(axes[0], freq_list, mse_list, 
               'MSE vs Frequency', 'MSE')
    # 绘制PSNR
    plot_metric(axes[1], freq_list, psnr_list, 
               'PSNR vs Frequency', 'PSNR (dB)')
    
    # 绘制SSIM
    plot_metric(axes[2], freq_list, ssim_list, 
               'SSIM vs Frequency', 'SSIM')
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'combined_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'可视化结果已保存至: {os.path.abspath(savedir)}')

# 数据加载和函数调用
freq_list = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_vs_freq.npy')
mse_samples = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_mse_vs_freq.npy')
psnr_samples = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_psnr_vs_freq.npy')
ssim_samples = np.load('outputGNN/inference/0225_b943fine50test_valall_perfreq10/b943/data/scatter_ssim_vs_freq.npy')

save_dir = 'tryfreq7'
plot_scatter_separate(freq_list, mse_samples, psnr_samples, ssim_samples, save_dir)