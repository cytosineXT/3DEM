import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np
import glob
import os

# Convert spherical coordinates to cartesian
def spherical_to_cartesian(theta, phi, device="cpu"):
    return torch.tensor([torch.sin(phi) * torch.cos(theta),
                         torch.sin(phi) * torch.sin(theta),
                         torch.cos(phi)], device=device, dtype=torch.float32)

# Calculate the projected area of a triangle for a given view vector
def calculate_projected_area(normal, area, view_vector):
    dot_product = torch.dot(normal, view_vector)
    return area * torch.abs(dot_product)

# Calculate angular similarity between two views
def angular_similarity(mesh, view1, view2, weight_area=0.5, weight_normals=0.5, device="cpu"):
    view_vector1 = spherical_to_cartesian(view1[0], view1[1], device=device)
    view_vector2 = spherical_to_cartesian(view2[0], view2[1], device=device)

    normals = torch.tensor(mesh.face_normals, dtype=torch.float32, device=device)
    areas = torch.tensor(mesh.area_faces, dtype=torch.float32, device=device)

    area_view1 = 0.
    area_view2 = 0.
    for face_idx in range(len(mesh.faces)):
        normal = normals[face_idx]
        area = areas[face_idx]
        projected_area1 = calculate_projected_area(normal, area, view_vector1)
        projected_area2 = calculate_projected_area(normal, area, view_vector2)
        if torch.dot(normal, view_vector1) < 0:
            area_view1 += projected_area1
        if torch.dot(normal, view_vector2) < 0:
            area_view2 += projected_area2

    projected_areas_difference = torch.abs(area_view1 - area_view2)
    similarity = 1 / (1 + 10 * projected_areas_difference.item())
    similarity = max(0, min(similarity, 1))
    return similarity, 1 - similarity

# Calculate frequency similarity (absolute difference)
def frequency_similarity(freq1, freq2):
    diff = torch.abs(freq1 - freq2)
    similarity = 1 - diff  # 直接使用差值的反转作为相似性
    return similarity, diff

# Custom contrastive loss for angles and frequencies

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1, obj_folder='planes/'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.obj_folder = obj_folder

    def load_mesh(self, plane_name):
        # Get the first 4 characters of the plane name
        plane_prefix = plane_name[:4]

        # Find the .obj file that matches the plane_prefix in the folder
        matching_files = glob.glob(os.path.join(self.obj_folder, f'{plane_prefix}*.obj'))
        if len(matching_files) == 0:
            raise FileNotFoundError(f"No matching .obj file found for {plane_name}")
        
        # Load the first matching file (assuming there's only one match)
        obj_file = matching_files[0]
        print(f"Loading mesh for {plane_name} from {obj_file}")
        
        # Load the mesh using trimesh
        mesh = trimesh.load(obj_file)
        return mesh

    def forward(self, rcs, in_em,device, beta=0.1):
        batch_size = rcs.shape[0]  # 6 in your case

        # Split in_em into plane names, incident angles (2 values), and frequency (1 value)
        plane_names = in_em[0]  # shape: (batch_size,)
        angles = in_em[1:3]     # shape: (6, 2) - second and third columns are the incident angles
        freqs = in_em[3]        # shape: (6,) - fourth column is the frequency

        # Initialize the total loss and pair counter
        total_loss = 0.0
        pair_count = 0

        # Loop over all pairs of RCS matrices in the batch
        for i in range(batch_size):
            # Load the mesh for the current plane name
            # mesh_i = self.load_mesh(plane_names[i])

            for j in range(i + 1, batch_size):
                # Only compare samples from the same plane
                if plane_names[i] == plane_names[j]:
                    # Extract the RCS matrices for the ith and jth samples
                    rcs_i = rcs[i]  # shape: (360, 720)
                    rcs_j = rcs[j]  # shape: (360, 720)

                    # Compute the MSE loss between the two RCS matrices
                    rcs_loss = F.mse_loss(rcs_i, rcs_j)

                    # Load the mesh for the jth plane name
                    mesh_j = self.load_mesh(plane_names[j])

                    # Compute the angle similarity between the ith and jth samples
                    view1 = (np.radians(angles[0][i]), np.radians(angles[1][i]))
                    view2 = (np.radians(angles[0][j]), np.radians(angles[1][j]))
                    
                    # Use the mesh from the ith plane (since both planes are the same here)
                    angle_sim, angle_dif = angular_similarity(mesh_j, view1, view2, weight_area=0.5, weight_normals=0., device=device)

                    # Compute the frequency similarity between the ith and jth samples
                    freq_sim, freq_dif = frequency_similarity(freqs[i], freqs[j])

                    # Calculate angle and frequency losses
                    angle_loss = rcs_loss * angle_sim - max(0, rcs_loss - self.margin) * angle_dif
                    freq_loss = rcs_loss * freq_sim - max(0, rcs_loss - self.margin) * freq_dif

                    # Combine the two losses
                    pairwise_loss = angle_loss + beta * freq_loss

                    # Accumulate the pairwise loss
                    total_loss += pairwise_loss

                    # Increment the pair counter since we found a matching pair
                    pair_count += 1

        # Return the total contrastive loss, averaged by the number of valid pairs
        if pair_count > 0:
            return total_loss / pair_count
        else:
            # If no pairs were found, return zero loss
            return torch.tensor(0.0, device=rcs.device)
        
# Example usage in your training loop
in_em = [('b7fd', 'bb7c', 'b7fd', 'bb7c'), 
         torch.tensor([30, 150, 150, 150]), 
         torch.tensor([270, 60, 150, 330]), 
         torch.tensor([0.9375, 0.8192, 0.8591, 0.9227], device='cuda:0')]

decoded = torch.rand([4, 1, 360, 720], device='cuda:0')
device = 'cuda:0'

# Instantiate the loss function
contrastive_loss_fn = ContrastiveLoss()

# Calculate contrastive loss
loss = contrastive_loss_fn(decoded, in_em, device=device)
print("Contrastive Loss:", loss.item())