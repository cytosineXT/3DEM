import torch
from pytorch_msssim import ms_ssim, ssim, SSIM, MS_SSIM
from net.utils import ssim as ss
a = torch.load('ref/decoded.pt')  # 你可以根据实际情况初始化矩阵 a
b = torch.load('ref/gt.pt')  # 你可以根据实际情况初始化矩阵 b
print(a.shape)
c = a - b
d = c ** 2
mean_value = d.sum() / d.numel()

print(mean_value)

X=a.unsqueeze(1) #需要输入(B, C, W, H)
Y=b.unsqueeze(1)
ssim_val = ssim( X, Y, data_range=torch.max(torch.max(X),torch.max(Y)), size_average=False) # return (N,)
ms_ssim_val = ms_ssim( X, Y, data_range=torch.max(torch.max(X),torch.max(Y)), size_average=False ) #(N,) 多分辨率，做了几次下采样
myssim = ss(a,b)
# set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details 自动做了平均方便当loss
ssim_loss = 1 - ssim( X, Y, data_range=torch.max(torch.max(X),torch.max(Y)), size_average=True) # return a scalar
ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=torch.max(torch.max(X),torch.max(Y)), size_average=True )

# reuse the gaussian kernel with SSIM & MS_SSIM. 复用高斯核，这样就可训练了！
ssim_module = SSIM(data_range=torch.max(torch.max(X),torch.max(Y)), size_average=True, channel=1) # channel=1 for grayscale images
ms_ssim_module = MS_SSIM(data_range=torch.max(torch.max(X),torch.max(Y)), size_average=True, channel=1)
ssim_loss = 1 - ssim_module(X, Y)
ms_ssim_loss = 1 - ms_ssim_module(X, Y)

print(f'ssim={ssim_val}')
print(f'msssim={ms_ssim_val}')
print(f'myssim={myssim}')
print(1-ssim_loss)
print(1-ms_ssim_loss)

# torch.Size([2, 361, 720])
# tensor(0.0212, device='cuda:0', grad_fn=<DivBackward0>)
# ssimtensor([0.8294, 0.9081], device='cuda:0', grad_fn=<MeanBackward1>)
# msssimtensor([0.8576, 0.9330], device='cuda:0', grad_fn=<MeanBackward1>)
# myssimtensor([0.5960, 0.9094], device='cuda:0', grad_fn=<MeanBackward1>)
# tensor(0.8687, device='cuda:0', grad_fn=<RsubBackward1>)
# tensor(0.8953, device='cuda:0', grad_fn=<RsubBackward1>)

# torch.Size([2, 361, 720])
# ssim=tensor([0.8294, 0.9081], device='cuda:0', grad_fn=<MeanBackward1>)
# ssim2=tensor([0.5952, 0.9081], device='cuda:0', grad_fn=<SqueezeBackward0>)