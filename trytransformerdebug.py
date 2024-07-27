from myswinunet import SwinTransformerSys
import torch

def checksize(x):
    print(x.shape, x.shape[0] * x.shape[1] * x.shape[2])
    return 1

x = torch.randn(2, 1, 45*90*96)
x = x.reshape(x.shape[0],45*90,-1)
checksize(x)

swinunet = SwinTransformerSys(embed_dim=12,window_size=9)
x = swinunet(x)
checksize(x)