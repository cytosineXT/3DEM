import kaolin
import torch
import os
# import numpy as np
# from torch.utils.data import DataLoader
from tqdm import tqdm
from data.datautils import MakeSurfaceMesh, SamplePointsFromMesh, SDFPoints, save_mesh, calculate_normal_and_centroid#, collate_fn
from net.jxtnet_autoencoderpure import MeshAutoencoder
from math import pi

'''存取tensor数据
torch.save(face_edges, 'face_edges.pt')
face_edge_loaded = torch.load('face_edges.pt')
'''

in_em = [45,90,2] #入射波\theta \phi freq
planesur_face = torch.load('planesur_face.pt')
planesur_vert = torch.load('planesur_vert.pt')
planesur_faceedge = torch.load('face_edges.pt') #这个face_edges是图论边，不是物理边，那这个生成边的代码不用动。


autoencoder = MeshAutoencoder( #这里实例化，是进去跑了init
    num_discrete_coors = 128
)

faces, loss = autoencoder( #这里使用网络，是进去跑了forward
    vertices = planesur_vert,
    faces = planesur_face,
    face_edges = planesur_faceedge,
    in_em = in_em
)

print('loss:',loss)

loss.backward() #这一步很花时间，但是没加optimizer是不是白给的

print('驯龙结束')
#2024年4月2日22:25:07 终于从头到尾跟着跑完了一轮 明天开始魔改！
#2024年4月6日17:24:56 encoder和decoder加入了EM因素，NN魔改完成，接下来研究如何训练。

# loss = transformer(
#     vertices = vertices,
#     faces = faces,
#     texts = ['a high chair', 'a small teapot']
# )

# loss.backward()

# # after much training of transformer, you can now sample novel 3d assets

# faces_coordinates, face_mask = transformer.generate(texts = ['a long table'])

# print('face_coordinates:',faces_coordinates)
# print('face_mask:',face_mask)