import kaolin
import torch
import os
# import numpy as np
# from torch.utils.data import DataLoader
from tqdm import tqdm
from data.datautils import MakeSurfaceMesh, SamplePointsFromMesh, SDFPoints, save_mesh, calculate_normal_and_centroid#, collate_fn
from net.jxtnet_autoencoder import MeshAutoencoder

#---------------------------preprocess ShapeNet data------------------------------------
#from DefTet
shapenet_source='/mnt/f/datasets/ShapeNet1'
# shapenet_source='/home/jxt/datasets/ShapeNet1'
save_cache_root = '/mnt/f/datasets/ShapeNetCache'
# save_cache_root='/home/jxt/datasets/Shapenet1/cache'
train=False
batch_size=8
add_occupancy=False
# only_planes=True
train_cat = ['02691156',]
save_1obj = False
save_allobj = False
get_pcd = False
get_sdf = False

ds = kaolin.io.shapenet.ShapeNetV1(root=shapenet_source, categories=train_cat,
                                    with_materials=False, train=train) #读取shapenetv1数据

###################################################
print('==> preprocess watertight mesh')
sv_dir = os.path.join(save_cache_root, 'watertight')
if not os.path.exists(sv_dir):
    os.makedirs(sv_dir)
watertight_mesh = kaolin.io.dataset.ProcessedDataset( 
    ds, MakeSurfaceMesh(100, 3, save_preprocess=True), num_workers=0,
    cache_dir=sv_dir)

#################################################
if get_pcd:
    print('==> preprocess point cloud')
    sv_dir = os.path.join(save_cache_root, 'pcd')
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    processed_ds = kaolin.io.dataset.ProcessedDataset(
        watertight_mesh, SamplePointsFromMesh(100000, with_normals=False, save_preprocess=True),
        num_workers=0,
        cache_dir=sv_dir)
###################################################
if get_sdf:
    print('==> preprocess sdf')
    sv_dir = os.path.join(save_cache_root, 'sdf')
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    occ_dataset = kaolin.io.dataset.ProcessedDataset(
        watertight_mesh, SDFPoints(100000, save_preprocess=True),
        num_workers=0,
        cache_dir=sv_dir)
#################################################

#第0个物体的
planename = watertight_mesh[0].attributes['name'] #物体名
planesynset = watertight_mesh[0].attributes['synset'] #物体大类号
planesur_vert = watertight_mesh[0].data[0] #水密表面顶点数据
planesur_face = watertight_mesh[0].data[1] #水密表面面数据
# planepcd = processed_ds[0].data #采样点云数据
# planesdfpcd = occ_dataset[0].data[0] #sdf点云数据
# planesdf = occ_dataset[0].data[1] #点云的sdf值
# planeface_norm = [] #face的end法向量
# planeface_centroid = [] #face的质心
# planeface_norm0 = [] #face的原点法向量

#---------------------------calculate normal and centroid------------------------------------
# for face in tqdm(planesur_face):
#     p1 = planesur_vert[face[0]]
#     p2 = planesur_vert[face[1]]
#     p3 = planesur_vert[face[2]]
#     normal, centroid = calculate_normal_and_centroid(p1, p2, p3)
#     planeface_norm0.append(normal)
#     planeface_centroid.append(centroid)
#     end_norm = centroid + normal
#     planeface_norm.append(end_norm)

# print('normal:', planeface_norm[0])
# print('centroid:', planeface_centroid[0])
# print('norm0:', planeface_norm0[0])
# print('face', planesur_face[0])
# for fa in planesur_face[0]:
#     print('point{}:{}'.format(fa, planesur_vert[fa]))

#-------------------------------------save obj-----------------------------------------------
if save_1obj == True:
    save_folder = '/mnt/f/datasets/ShapeNet1plane_watertight'
    # save_folder = '/home/jxt/datasets/shapenetplane_watertight'
    os.makedirs(os.path.join(save_folder, planesynset), exist_ok=True)
    print('正在保存一个模型')
    saveobj = os.path.join(save_folder, planename  + '.obj')
    save_mesh(planesur_vert.data.cpu().numpy(),  planesur_face.data.cpu().numpy(), saveobj)
    print('模型{}.obj已保存'.format(planename))

#------------------------------------NN Encoder----------------------------------------------
autoencoder = MeshAutoencoder( #这里实例化，是进去跑了init
    num_discrete_coors = 128
)

planesur_face = planesur_face.unsqueeze(0)#.to(device)
planesur_vert = planesur_vert.unsqueeze(0)#.to(device)
# torch.save(planesur_face, 'planesur_face.pt')
# torch.save(planesur_vert, 'planesur_vert.pt')
'''存取tensor数据
torch.save(face_edges, 'face_edges.pt')
face_edge_loaded = torch.load('face_edges.pt')
'''
planesur_faceedge = torch.load('face_edges.pt') #这个face_edges是图论边，不是物理边，那这个生成边的代码不用动。
faces, loss = autoencoder( #这里使用网络，是进去跑了forward
    vertices = planesur_vert,
    faces = planesur_face,
    face_edges = planesur_faceedge
)

print('loss:',loss)

loss.backward() #这一步很花时间，但是没加optimizer是不是白给的

print('驯龙结束')#终于从头到尾跟着跑完了一轮2024年4月2日22:25:07 明天开始魔改！

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