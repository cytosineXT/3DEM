import torch

from net.meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
)

# autoencoder

autoencoder = MeshAutoencoder(
    num_discrete_coors = 128
)

# mock inputs

vertices = torch.randn((2, 121, 3))            # (batch, num vertices, coor (3))
faces = torch.randint(0, 121, (2, 64, 3))      # (batch, num faces, vertices (3))

# make sure faces are padded with `-1` for variable lengthed meshes

# forward in the faces

loss = autoencoder(
    vertices = vertices,
    faces = faces
)

loss.backward()

# after much training...
# you can pass in the raw face data above to train a transformer to model this sequence of face vertices

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

# after much training of transformer, you can now sample novel 3d assets

faces_coordinates, face_mask = transformer.generate(texts = ['a long table'])

print('face_coordinates:',faces_coordinates)
print('face_mask:',face_mask)
# (batch, num faces, vertices (3), coordinates (3)), (batch, num faces)
# now post process for the generated 3d asset