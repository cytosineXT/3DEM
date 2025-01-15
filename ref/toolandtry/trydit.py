import torch
from ditmodel import DiT
from diffusion import create_diffusion
import time
from net.utils import get_model_memory_nolog
tic = time.time()

length = 2500
cudadevice = 'cuda:0'
hidden_size = 576
device = torch.device(cudadevice if torch.cuda.is_available() else "cpu")
# in_em = [('b943'),
#          torch.tensor([150],device=device), 
#          torch.tensor([60],device=device), 
#          torch.tensor([0.9142],device=device, dtype=torch.float64)]
# in_em = [('b943', 'b943', 'bb7c', 'b7fd', 'bb7c', 'b7fd', 'b943', 'b979', 'b979', 'b943'),
#          torch.tensor([150, 180, 180,  90, 120,  30, 150,  90, 180,  90]), 
#          torch.tensor([ 60, 210,  30, 150,  30, 210,  60, 330, 150,  30]), 
#          torch.tensor([0.9142, 0.7557, 0.9015, 0.6822, 0.5867, 0.5681, 0.7617, 0.5124, 0.5747, 0.9309], dtype=torch.float64)]
in_em = [('b7fd'), 
        torch.tensor([[-8.6603e-01,  5.0000e-01, -4.3711e-08]], device='cuda:0'), 
        torch.tensor([0.5178], device='cuda:0', dtype=torch.float64)]
# in_em = [('b7fd'), 
#         torch.tensor([[-8.6603e-01,  5.0000e-01, -4.3711e-08],[-4.3301e-01, -7.5000e-01,  5.0000e-01]], device='cuda:0'), 
#         torch.tensor([0.5178, 0.7452], device='cuda:0', dtype=torch.float64)]
# in_em = [('b7fd', 'b943', 'b979', 'b979', 'b943', 'b943', 'b7fd', 'b7fd', 'b943', 'bb7c'), 
#         torch.tensor([[-8.6603e-01,  5.0000e-01, -4.3711e-08],
#         [-4.3301e-01, -7.5000e-01,  5.0000e-01],
#         [ 4.3301e-01,  7.5000e-01, -5.0000e-01],
#         [ 4.3301e-01, -2.5000e-01,  8.6603e-01],
#         [-5.0000e-01, -8.6603e-01,  1.1925e-08],
#         [ 7.5710e-08, -4.3711e-08, -1.0000e+00],
#         [-8.6603e-01, -5.0000e-01,  1.1925e-08],
#         [ 5.0000e-01,  8.6603e-01, -4.3711e-08],
#         [-4.3301e-01, -2.5000e-01, -8.6603e-01],
#         [ 8.6603e-01,  0.0000e+00,  5.0000e-01]], device='cuda:0'), 
#         torch.tensor([0.5178, 0.7452, 0.6798, 0.7041, 0.9835, 0.9201, 0.6229, 0.6098, 0.5372,
#         0.8512], device='cuda:0', dtype=torch.float64)]
input_matrix = torch.randn(1, hidden_size, length).to(device)  # batchsize channel 长(N, T, D)(1,2500,784)
diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
print(input_matrix.shape, hidden_size * length) #(784 * 2500)
dit = DiT(hidden_size=hidden_size, num_heads=16, depth=28, length=length).to(device)
get_model_memory_nolog(dit)
print(f'耗时{time.time() - tic:.4f}s')
tic = time.time()

x = input_matrix.reshape(1,-1,hidden_size) # Reshape to (N, T, D) torch.Size([1, 2500, 784])
t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device) #torch.Size([1]) 在embedding的时候会变成hidden_size的
print(x.shape, x.shape[0] * x.shape[1] * x.shape[2])
print(f'耗时{time.time() - tic:.4f}s')
tic = time.time()

x = dit(x=x,t=t,in_em=in_em)
print(x.shape, x.shape[0] * x.shape[1] * x.shape[2])
print(f'耗时{time.time() - tic:.4f}s')