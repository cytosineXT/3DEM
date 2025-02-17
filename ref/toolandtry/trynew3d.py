import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import torch
import time

tic = time.time()

rcs = torch.load('/mnt/Disk/jiangxiaotian/datasets/RCS_mapsmall/RCSmap_theta90phi330f0.9.pt')[:,:,0]
print(rcs.shape)
rcs_np = rcs.detach().cpu().numpy()
npmax = np.max(rcs_np)
npmin = np.min(rcs_np)
print(f'max:{npmax},min:{npmin}')
theta = np.linspace(0, 2 * np.pi, rcs_np.shape[1])
phi = np.linspace(0, np.pi, rcs_np.shape[0])
theta, phi = np.meshgrid(theta, phi)

x = rcs_np * np.sin(phi) * np.cos(theta)
y = rcs_np * np.sin(phi) * np.sin(theta)
z = rcs_np * np.cos(phi)

fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, cmin = 0, cmax = npmax,  surfacecolor=rcs_np, colorscale='Jet', colorbar=dict(exponentformat='E',title=dict(side='top',text="RCS/m²"), showexponent='all', showticklabels=True, thickness = 30,tick0 = 0, dtick = npmax))]) # 

fig.update_layout(
    scene=dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
        aspectratio=dict(x=1, y=1, z=0.8),
        aspectmode="manual",
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.7))
    ),
)
# pio.show(fig)
pio.write_image(fig, 'newRCS.png')
print(f'用时{time.strftime("%H:%M:%S",time.gmtime(time.time()-tic))}')
tic = time.time()