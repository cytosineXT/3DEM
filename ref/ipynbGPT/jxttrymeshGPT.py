import torch
# from plyfile import PlyData,PlyElement
import numpy as np
from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
)
def write_ply(save_path,pointin,facein,text=True):
    vertex = np.array(pointin, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])#解释的是tuple中每一项的含义或者属性
    face = np.array(facein, dtype=[('vertex_indices', 'i4', (3,))])    
    # face = np.array(facein, dtype=[('vertex_indices', 'i4', (3,)),('Efiled','f4')]) #等加入了电场表达，再多这一项    
    # print(vertex)
    pointp = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    facep = PlyElement.describe(face, 'face', comments=['faces'])
    PlyData([pointp,facep], text=text).write(save_path)

def jxtmesh2ply(faces):
    size = np.array(faces).shape
    print(size)
    plyfaces = []
    plydots = []
    #需要利用list.index()方法，得到目标索引
    for batch in faces:
        for face in batch:
            tface = []
            #[[-0.9141,  0.1328,  0.8672],
            #[ 0.7266,  0.8203,  0.6484],
            #[-0.9453,  0.4922,  0.4297]]一个face长这样
            #提出了一个face，要生成dots列表和dots索引产生的face列表
            for dot in face:
                tdot = tuple(dot)
                plydots.append(tdot)
                tface.append(plydots.index(tdot)) #生成一个含三个整数的list，为对应点的index
            tface = (tface,)
            plyfaces.append(tface)
    return plydots, plyfaces

def jxtdot2ply(dots):
    # size = np.array(dots).shape
    # print(size)
    plydots = []
    for batch in dots:
        for dot in batch:
            tdot = tuple(dot)
            plydots.append(tdot)
    return plydots

def jxtface2ply(faces):
    # size = np.array(faces).shape
    # print(size)
    plyfaces = []
    for batch in faces:
        for face in batch:
            tface = (face,)
            plyfaces.append(tface)
    return plyfaces


## 代码主体部分
# autoencoder
autoencoder = MeshAutoencoder(
    num_discrete_coors = 128
)

# mock inputs
vertices = torch.randn((2, 121, 3))            # (batch, num vertices, coor (3)) 正态随机 2批，121个顶点，每个顶点3个xyz坐标值
faces = torch.randint(0, 121, (2, 64, 3))      # (batch, num faces, vertices (3)) 0-121随机整数 2批，64个面，每面3个顶点索引
initplydots, initplyfaces = jxtdot2ply(vertices), jxtface2ply(faces)
write_ply(r'F:\workspace\mwGPT\meshgpt-pytorch\output\inmesh4.ply', initplydots, initplyfaces, text=True)
# make sure faces are padded with `-1` for variable lengthed meshes

# forward in the faces
loss = autoencoder(
    vertices = vertices,
    faces = faces
)
loss.backward()

# after much training...
# you can pass in the raw face data above to train a transformer to model this sequence of face vertices 当进行了足够的训练后，您可以传入上面的原始面数据来训练transformer 来对这个面顶点序列进行建模
transformer = MeshTransformer(
    autoencoder,
    dim = 512,
    max_seq_len = 768,
    condition_on_text = True
)

loss = transformer(
    vertices = vertices,
    faces = faces,
    texts = ['a high chair', 'a small teapot']
)
loss.backward()

# after much training of transformer, you can now sample novel 3d assets经过对 Transformer 的大量训练后，您现在可以对新的 3D 资源进行采样
faces_coordinates, face_mask = transformer.generate(texts=['a long table']) #回头探究一下这个怎么generate的，一轮是什么，结果是在哪

# plydots, plyfaces = jxtmesh2ply(faces_coordinates)
# write_ply(r'F:\workspace\mwGPT\meshgpt-pytorch\output\mesh4.ply', plydots, plyfaces, text=True)

# (batch, num faces, vertices (3), coordinates (3)), (batch, num faces)
# now post process for the generated 3d asset