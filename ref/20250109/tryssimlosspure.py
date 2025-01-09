import torch
from pytorch_msssim import ms_ssim, ssim, SSIM, MS_SSIM
from net.utils import ssim as ss
a = torch.load('ref/decoded.pt')  # 你可以根据实际情况初始化矩阵 a
b = torch.load('ref/gt.pt')  # 你可以根据实际情况初始化矩阵 b
print(a.shape)

decoded=a.unsqueeze(1) #需要输入(B, C, W, H)
GT=b.unsqueeze(1)
ssim_val = ssim( decoded, GT, data_range=torch.max(torch.max(decoded),torch.max(GT)), size_average=False) # return (N,)
print(f'ssim={ssim_val}')

ssim_val2 = torch.stack([ssim(decoded[i].unsqueeze(0), GT[i].unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze()
print(f'ssim2={ssim_val2}')

# torch.Size([2, 361, 720])
# ssim=tensor([0.8294, 0.9081], device='cuda:0', grad_fn=<MeanBackward1>) #确实 又是栽倒在最大值问题上
# ssim2=tensor([0.5952, 0.9081], device='cuda:0', grad_fn=<SqueezeBackward0>)
