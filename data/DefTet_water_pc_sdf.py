import kaolin
# import torch
import os
# import numpy as np
# from torch.utils.data import DataLoader
from tqdm import tqdm
from datautils import MakeSurfaceMesh, SamplePointsFromMesh, SDFPoints, save_mesh, calculate_normal_and_centroid#, collate_fn

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
save_allobj = True

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
print('==> preprocess point cloud')
sv_dir = os.path.join(save_cache_root, 'pcd')
if not os.path.exists(sv_dir):
    os.makedirs(sv_dir)
processed_ds = kaolin.io.dataset.ProcessedDataset(
    watertight_mesh, SamplePointsFromMesh(100000, with_normals=False, save_preprocess=True),
    num_workers=0,
    cache_dir=sv_dir)
###################################################
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
planepcd = processed_ds[0].data #采样点云数据
planesdfpcd = occ_dataset[0].data[0] #sdf点云数据 torch.Size([100000, 3])  tensor([[ 0.2769,  0.1747,  0.1532],
planesdf = occ_dataset[0].data[1] #点云的sdf值 torch.Size([100000])   tensor([-0.0360, -0.0420, -0.2393,  ..., -0.2449, -0.1963, -0.1159])
planeface_norm = [] #face的end法向量
planeface_centroid = [] #face的质心
planeface_norm0 = [] #face的原点法向量

#---------------------------calculate normal and centroid------------------------------------
for face in tqdm(planesur_face):
    p1 = planesur_vert[face[0]]
    p2 = planesur_vert[face[1]]
    p3 = planesur_vert[face[2]]
    normal, centroid = calculate_normal_and_centroid(p1, p2, p3)
    planeface_norm0.append(normal)
    planeface_centroid.append(centroid)
    end_norm = centroid + normal
    planeface_norm.append(end_norm)
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

if save_allobj == True:
    save_folder = '/mnt/f/datasets/ShapeNet1plane_watertight'
    # save_folder = '/home/jxt/datasets/shapenetplane_watertight'
    os.makedirs(os.path.join(save_folder, planesynset), exist_ok=True)
    for i in tqdm(range(len(ds))):
        planenamei = watertight_mesh[i].attributes['name'] #物体名
        planesynseti = watertight_mesh[i].attributes['synset'] #物体大类号
        planesur_verti = watertight_mesh[i].data[0] #水密表面顶点数据
        planesur_facei = watertight_mesh[i].data[1] #水密表面面数据
        saveobj = os.path.join(save_folder, planenamei  + '.obj')
        save_mesh(planesur_verti.data.cpu().numpy(),  planesur_facei.data.cpu().numpy(), saveobj)
        print('模型{}.obj已保存'.format(planenamei))
#------------------------------------NN Encoder----------------------------------------------
