import kaolin as kaolin
import torch
from datetime import datetime
import numpy as np

def calculate_normal_and_centroid(p1, p2, p3):
    # Create vectors from points
    v1 = p2 - p1
    v2 = p3 - p1
    # Compute the cross product of vectors
    cross_product = np.cross(v1, v2)
    # Normalize the cross product to get the unit normal vector
    normal = cross_product / np.linalg.norm(cross_product)
    # Calculate the centroid of the triangle
    centroid = (p1 + p2 + p3) / 3
    return normal, centroid

class MakeSurfaceMesh:
    def __init__(self, resolution=100, smoothing_iterations=3, save_preprocess=False, max_length=0.9):
        self.resolution = resolution
        self.smoothing_iterations = smoothing_iterations
        self.save_preprocess = save_preprocess
        self.max_length = max_length
        self.error_idx = []

    def __call__(self, mesh):
        vertices = mesh.vertices.cuda()
        faces = mesh.faces.cuda()
        max_l = max(vertices[..., 0].max() - vertices[..., 0].min(),
                    vertices[..., 1].max() - vertices[..., 1].min(),
                    vertices[..., 2].max() - vertices[..., 2].min())
        vertices = (vertices / max_l) * self.max_length
        mid_p = (vertices.max(dim=0)[0] + vertices.min(dim=0)[0]) / 2
        vertices = vertices - mid_p.unsqueeze(dim=0)
        voxelgrid = kaolin.ops.conversions.trianglemeshes_to_voxelgrids(
                vertices.unsqueeze(0), faces,
                resolution=self.resolution)

        odms = kaolin.ops.voxelgrid.extract_odms(voxelgrid)
        voxelgrid = kaolin.ops.voxelgrid.project_odms(odms)
        # convert back to voxelgrids
        new_vertices, new_faces = kaolin.ops.conversions.voxelgrids_to_trianglemeshes(
            voxelgrid,
        )
        new_vertices = new_vertices[0]
        new_faces = new_faces[0]
        # laplacian smoothing
        adj_mat = kaolin.ops.mesh.adjacency_matrix(
            new_vertices.shape[0],
            new_faces)
        num_neighbors = torch.sparse.sum(
            adj_mat, dim=1).to_dense().view(-1, 1)
        for i in range(self.smoothing_iterations):
            neighbor_sum = torch.sparse.mm(adj_mat, new_vertices)
            new_vertices = neighbor_sum / num_neighbors
        # normalize
        orig_min = vertices.min(dim=0)[0]
        orig_max = vertices.max(dim=0)[0]
        new_min = new_vertices.min(dim=0)[0]
        new_max = new_vertices.max(dim=0)[0]
        new_vertices = (new_vertices - new_min) / (new_max - new_min)
        new_vertices = new_vertices * (orig_max - orig_min) + orig_min
        return new_vertices.cpu(), new_faces.cpu()

    def __repr__(self):
        if not self.save_preprocess:
            return 'watertight_%s'%(str(datetime.now()))
        return 'watertight'

class SamplePointsFromMesh:
    def __init__(self, num_points, with_normals=True, save_preprocess=False):
        self.num_points = num_points
        self.with_normals = with_normals
        self.save_preprocess = save_preprocess
    def __call__(self, mesh):
        vertices = mesh[0].unsqueeze(dim=0).float().cuda()
        faces = mesh[1].long().cuda()
        points, face_choices = kaolin.ops.mesh.sample_points(
            vertices, faces, self.num_points) #kaolin.ops.mesh.sample_point(vertices, faces, num_points)用来从mesh中采样点
        if self.with_normals:
            face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces)
            face_normals = kaolin.ops.mesh.face_normals(
                face_vertices, unit=True)
            normals = face_normals[face_choices]
            return points.squeeze(0), normals.squeeze(0)
        return points.squeeze(0).cpu()

    def __repr__(self):
        if not self.save_preprocess:
            return 'point_cloud_%s'%(str(datetime.now()))
        return 'point_cloud'
    
def kaolin_mesh_to_sdf(verts_bxnx3, face_fx3, points_bxnx3):
    sign = kaolin.ops.mesh.check_sign(verts_bxnx3, face_fx3, points_bxnx3, hash_resolution=512)
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(verts_bxnx3, face_fx3)
    distance, index, dist_type = kaolin.metrics.trianglemesh.point_to_mesh_distance(points_bxnx3, face_vertices)
    sign = sign.float() * 2.0 - 1.0  # (1: inside; -1: outside)
    sdf = sign * distance
    return sdf

class SDFPoints:
    def __init__(self, num_points, with_normals=True, save_preprocess=False):
        self.num_points = num_points
        self.with_normals = with_normals
        self.save_preprocess = save_preprocess
    def __call__(self, mesh):
        vertices = mesh[0].unsqueeze(dim=0).float().cuda()
        faces = mesh[1].long().cuda()
        points = 1.05 * (torch.rand(1, self.num_points, 3).cuda() - .5)
        sdf = kaolin_mesh_to_sdf(vertices, faces, points)
        return points[0].cpu(), sdf[0].cpu()

    def __repr__(self):
        if not self.save_preprocess:
            return 'sdf_%s'%(str(datetime.now()))
        return 'sdf'
    
def laplace_regularizer_const(mesh_verts, mesh_faces): #输入mesh的顶点和面
    term = torch.zeros_like(mesh_verts) #torch.zeros_lise()函数生成一个和入参一样大小的Tensor 0 矩阵！
    norm = torch.zeros_like(mesh_verts[..., 0:1]) #行数和mesh_verts一样，列数只有前两列

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v0 - v1))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0) #torch.clamp()函数把第一个入参限制在第二个入参之内，可以是max也可以是min

    return torch.mean(term**2)


def loss_f(mesh_verts, mesh_faces, points, it, iterations=5000, laplacian_weight=0.1, device='cuda'): #输入mesh的顶点和面、point是真值、it是当前迭代数
    pred_points = kaolin.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]#.to(device) #.unsqueeze(0)用于在第0个地方添加一个维度，若是.squeeze(0)则是在这里减一个维度。
    '''points.shape = torch.Size([100000, 3])
    points = tensor([[-0.3080,  0.1091,  0.0110],
        [-0.3566,  0.1161, -0.0140],
        [-0.3518, -0.0123,  0.0018],
        ...,
        [-0.2108, -0.0960,  0.0109],
        [ 0.1543, -0.0717,  0.0650],
        [-0.0127, -0.0007,  0.1905]], device='cuda:0')

    pred_points.shape = torch.Size([50000, 3])
    pred_points = tensor([[-0.0047, -0.2714, -0.1144],
        [ 0.0048,  0.2994, -0.0427],
        [ 0.0071,  0.2902,  0.0704],
        ...,
        [-0.0390,  0.2713,  0.1192],
        [ 0.1616, -0.0199,  0.2534],
        [ 0.2456,  0.0799,  0.1473]], device='cuda:0',
       grad_fn=<SelectBackward0>)
    '''

    #pred_points使用生成出来的mesh的顶点和面 通过kaolin.ops.mesh.sample_points采样采出来的点。这样可以和真值的点云做chamfer distance loss。
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
    if it > iterations//2: #如果it(当前迭代数)大于总迭代数的一半(//整除符号)，那么加入laplace正则项loss。
        lap = laplace_regularizer_const(mesh_verts, mesh_faces) #这是刚定义的laplace loss，Laplace正则项通常用于平滑处理或者是防止过拟合。这种策略可能是为了在训练的后半段加强模型的泛化能力。
        return chamfer + lap * laplacian_weight #合成的总loss
    return chamfer


def save_mesh(pointnp_px3, facenp_fx3, fname, partinfo=None):
    if partinfo is None:
        fid = open(fname, 'w')
        ss = ''
        for pidx, p in enumerate(pointnp_px3):
            pp = p
            ss += 'v %f %f %f\n' % (pp[0], pp[1], pp[2])
        for f in facenp_fx3:
            f1 = f + 1
            ss += 'f %d %d %d\n' % (f1[0], f1[1], f1[2])
        fid.write(ss)
        fid.close()
    else:
        fid = open(fname, 'w')
        for pidx, p in enumerate(pointnp_px3):
            if partinfo[pidx, -1] == 0:
                pp = p
                color = [1, 0, 0]
            else:
                pp = p
                color = [0, 0, 1]
            fid.write('v %f %f %f %f %f %f\n' % (pp[0], pp[1], pp[2], color[0], color[1], color[2]))
        for f in facenp_fx3:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
        fid.close()
    return

def collate_fn(batch_list):
        data = dict()
        data['verts'] = [da[0][0][0] for da in batch_list]
        data['faces'] = [da[0][0][1] for da in batch_list]
        data['sample_points'] = torch.cat([da[0][1].unsqueeze(dim=0) for da in batch_list], dim=0)
        data['name'] = [da[1][0]['name'] for da in batch_list]
        data['synset'] = [da[1][0]['synset'] for da in batch_list]
        data['sdf_point'] = torch.cat([da[0][2][0].unsqueeze(dim=0) for da in batch_list], dim=0)
        data['sdf_value'] = torch.cat([da[0][2][1].unsqueeze(dim=0) for da in batch_list], dim=0)
        return data
