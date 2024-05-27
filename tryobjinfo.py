import trimesh
import numpy as np

mesh = trimesh.load_mesh('./planes/all/b82731071bd39b66e4c15ad8a2edd2e.obj')
surface_area = mesh.area
volume = mesh.volume
scale = mesh.scale
# 获取包围盒的最小和最大顶点
min_bound, max_bound = mesh.bounds
# 计算最远的两点之间的距离
max_distance = np.linalg.norm(max_bound - min_bound)

print(f"物体中相距最远的两点的距离为：{max_distance:.4f}")
print(f"表面积：{surface_area:.4f}")
print(f"体积：{volume:.4f}")