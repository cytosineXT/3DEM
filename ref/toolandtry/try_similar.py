import numpy as np 
import trimesh 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def spherical_to_cartesian (theta, phi):
    """Convert spherical coordinates (azimuth theta, elevation phi) to Cartesian coordinates.""" 
    return np.array([ np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi) ])

def calculate_projected_area(normal, area, view_vector ):
    """Calculate the projected area of a triangle for a given view vector."""
    dot_product = np.dot(normal, view_vector) 
    return area * abs(dot_product)

def calculate_angular_change (normal1, normal2 ):
    """Calculate the angular change between two normal vectors.""" 
    cos_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2)) 
    cos_angle = np.clip(cos_angle, - 1.0, 1.0) # Ensure the value is within [-1, 1] to avoid numerical issues 
    return np.arccos(cos_angle)

def calculate_similarity (mesh, view1, view2, weight_area=0.5, weight_normals=0.5):
    """Calculate the similarity between two views based on projected areas and normal vector changes.""" 
    view_vector1 = spherical_to_cartesian(*view1) 
    view_vector2 = spherical_to_cartesian(*view2) # Initialize arrays for projected areas and angular changes 
    projected_areas_similarity = [] 
    angular_changes = [] # Iterate through all faces (triangles) 
    for face_idx in range(len(mesh.faces)): 
        normal = mesh.face_normals[face_idx] 
        area = mesh.area_faces[face_idx] # Calculate projected areas for both views 
        projected_area1 = calculate_projected_area(normal, area, view_vector1) 
        projected_area2 = calculate_projected_area(normal, area, view_vector2) # Calculate similarity based on projected areas (cosine similarity) 
        projected_areas_similarity.append(projected_area1 * projected_area2) # Calculate angular change between the normal vectors from both views 
        angular_change = calculate_angular_change(view_vector1, view_vector2) 
        angular_changes.append(angular_change) # Normalize projected area similarities 
        areas_similarity = np. sum(projected_areas_similarity) / (np.linalg.norm(projected_areas_similarity) + 1e-8) # Normalize angular changes (smaller changes should lead to higher similarity) 
        angular_similarity = np.exp(-np.mean(angular_changes)) # Use exponential decay for angular difference # Combine both metrics using weighted sum 
        combined_similarity = weight_area * areas_similarity + weight_normals * angular_similarity 
        return combined_similarity 
    
def extract_triangle_features (mesh): 
    """Extract triangle features: vertex coordinates, interior angles, and normal vectors.""" 
    for face_idx in range(len(mesh.faces)):
        vertices = mesh.vertices[mesh.faces[face_idx]] 
        normal = mesh.face_normals[face_idx] 
        edges = np.array([np.linalg.norm(vertices[i] - vertices[(i+ 1)%3]) for i in range(3)]) # Calculate interior angles using the cosine rule 
        angles = np.array([ np.arccos(np.dot(vertices[ 1]-vertices[0], vertices[2]-vertices[0]) / (np.linalg.norm(vertices[ 1]-vertices[0]) * np.linalg.norm(vertices[ 2]-vertices[0]))), np.arccos(np.dot(vertices[ 2]-vertices[1], vertices[0]-vertices[1]) / (np.linalg.norm(vertices[ 2]-vertices[1]) * np.linalg.norm(vertices[ 0]-vertices[1]))), np.arccos(np.dot(vertices[ 0]-vertices[2], vertices[1]-vertices[2]) / (np.linalg.norm(vertices[ 0]-vertices[2]) * np.linalg.norm(vertices[ 1]-vertices[2]))) ]) 
        # print(f"Triangle {face_idx + 1}:") 
        # print(f"Vertices: \n{vertices}") 
        # print(f"Normal: {normal}") 
        # print(f"Interior angles (radians): {angles}") 
        # print("-" * 50) # Load the STL file 
        
# def load_stl_file(filename): 
#     """Load STL file and return a trimesh object.""" 
#     mesh = trimesh.load_mesh(filename) 
#     if isinstance (mesh, trimesh.Scene):
#             mesh = mesh.dump(concatenate= True) 
#     return mesh 
def load_file(filename): 
    """Load OBJ file and return a trimesh object."""
    mesh = trimesh.load(filename)  # trimesh.load can handle obj files directly
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh

# def visualize_mesh(mesh):
#     """Visualize the 3D mesh using trimesh's show method."""
#     mesh.show()

def visualize_mesh_with_matplotlib(mesh):
    """Visualize the 3D mesh using matplotlib."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract vertex coordinates and faces (triangles)
    vertices = mesh.vertices
    faces = mesh.faces

    # Create a Poly3DCollection for the mesh triangles
    poly3d = [[vertices[vertex] for vertex in face] for face in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5))

    # Auto scale to the mesh size
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    # plt.show()
    plt.savefig('b943.png')

# Example usage 
if __name__ == "__main__": 
    # Load your STL file 
    stl_filename = 'planes/b943b632fd36f75ac1ccec171a275967.obj' # Replace with your STL file path 
    mesh = load_file(stl_filename) # Extract triangle features 
    extract_triangle_features(mesh) # Define view angles (azimuth and elevation in radians) 
    # view1 = (np.radians(30), np.radians( 45)) # View 1: (theta, phi) 
    # view2 = (np.radians(60), np.radians( 30)) # View 2: (theta, phi) 
    viewls = []
    for i in range(20):
        viewls.append((np.random.randint(0,180),np.random.randint(0,360),np.random.randint(0,180),np.random.randint(0,360)))
    for view in viewls:
    # for view in [(30,45,30,135),(30,45,30,225),(30,45,30,45)]:
        view1 = (np.radians(view[0]), np.radians(view[1])) # View 1: (theta, phi) 
        view2 = (np.radians(view[2]), np.radians(view[3])) # View 2: (theta, phi) 
        # Calculate the similarity between the two views using both projected area and normal
        similarity = calculate_similarity(mesh, view1, view2, weight_area= 0.5, weight_normals=0.5) 
        print(f"Similarity between {view} (theta phi theta phi): {similarity:.4f}")
    # mesh.show()
    visualize_mesh_with_matplotlib(mesh)