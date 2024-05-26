import kaolin
import os
from tqdm import tqdm
from data.datautils import MakeSurfaceMesh, save_mesh

#---------------------------preprocess ShapeNet data------------------------------------
#from DefTet
shapenet_source='/mnt/f/datasets/ShapeNetCore.v1'
save_cache_root = '/mnt/f/datasets/ShapeNetCache'
train=False
batch_size=8
add_occupancy=False
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

planesynset = watertight_mesh[0].attributes['synset'] #物体大类号，反正都是同一类

#-------------------------------------save obj-----------------------------------------------
if save_1obj == True:
    save_folder = '/mnt/f/datasets/ShapeNet1plane_watertight'
    # save_folder = '/home/jxt/datasets/shapenetplane_watertight'
    os.makedirs(os.path.join(save_folder, planesynset), exist_ok=True)
    #第0个物体的
    planename = watertight_mesh[0].attributes['name'] #物体名
    # planesynset = watertight_mesh[0].attributes['synset'] #物体大类号
    planesur_vert = watertight_mesh[0].data[0] #水密表面顶点数据
    planesur_face = watertight_mesh[0].data[1] #水密表面面数据
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
        # planesynseti = watertight_mesh[i].attributes['synset'] #物体大类号
        planesur_verti = watertight_mesh[i].data[0] #水密表面顶点数据
        planesur_facei = watertight_mesh[i].data[1] #水密表面面数据
        saveobj = os.path.join(save_folder, planenamei  + '.obj')
        save_mesh(planesur_verti.data.cpu().numpy(),  planesur_facei.data.cpu().numpy(), saveobj)
        print('模型{}.obj已保存'.format(planenamei))
#------------------------------------NN Encoder----------------------------------------------
