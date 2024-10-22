import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from networks import ResnetGenerator
import torchvision.transforms as transforms
import cv2
from PIL import Image

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)


# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")




scene_root_path = "./data/scenes"
scene_name = "model.obj"
#"000ecb5b-b877-4f9a-ab6f-90f385931658.obj"
    
img_root_path = './data/sketch'
img_name = 'sketch1.jpg'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
scene_json_pt = os.path.join(scene_root_path,scene_name)
sketch_pt = os.path.join(img_root_path,img_name)
scene_json = []
sketch = []

photo2sketch_model_path = './latest_net_G.pth'
photo2sketch_model = ResnetGenerator(
    input_nc=3,
    output_nc=1,
    use_dropout=False,
    n_blocks=9
)
photo2sketch_model.load_state_dict(torch.load(photo2sketch_model_path))

swint_model_path = './best.pth'
swint_model = torch.load(swint_model_path)

# transfer scenejson to mesh
# operate(scene_json_pt)









# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj(scene_json_pt)
faces = faces_idx.verts_idx












# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
teapot_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)

# Initialize a perspective camera.
cameras = FoVPerspectiveCameras(device=device)

# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
# edges. Refer to blending.py for more details. 
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=224, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=100, 
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader. 
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)


# We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=224, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
# We can add a point light in front of the object. 
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
)



# Select the viewpoint using spherical angles  
distance = 20   # distance from camera to the object
elevation = 50.0   # angle of elevation in degrees
azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

eye = torch.Tensor([[0,2,4]])
at = torch.Tensor([[-1.5,1,0]])


# Get the position of the camera based on the spherical angles
# R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
R, T = look_at_view_transform(eye=eye,at = at, device=device)

# Render the teapot providing the values of R and T. 
silhouette = silhouette_renderer(meshes_world=teapot_mesh, R=R, T=T)
image_ref = phong_renderer(meshes_world=teapot_mesh, R=R, T=T)
sketch_ref = cv2.imread(sketch_pt)

silhouette = silhouette.cpu().numpy()
image_ref = image_ref.cpu().numpy()


print('here')
print(image_ref)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(silhouette.squeeze()[...,3])  # only plot the alpha channel of the RGBA image
plt.grid(False)
plt.subplot(1, 2, 2)
plt.imshow(image_ref.squeeze())
plt.grid(False)
plt.show()




class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref,encoder, p2s):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        self.encoder = encoder
        self.p2s = p2s
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        print('image_ref shape: ', image_ref.shape)
        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([[0,  10, 1]], dtype=np.float32)).to(meshes.device))
        
        self.at = nn.Parameter(
            torch.from_numpy(np.array([[0,0,0]], dtype= np.float32)).to(meshes.device)
        )

    def forward(self):
        
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        R, T = look_at_view_transform(eye=self.camera_position, at= self.at, device=self.device)

        # R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)
        


        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        
        # Calculate the silhouette loss
        loss = torch.sum((image[..., 3] - self.image_ref[..., 3]) ** 2)
        return loss, image
    
    def rec_loss(self, image):
        l2_loss = torch.sum((image[..., 3] - self.image_ref[..., 3]) ** 2)
        # with torch.no_grad():
        #     im_r = self.image_ref.permute(0,3,1,2)
        #     im = self.image_ref.permute(0,3,1,2)
        #     input_feature = self.encoder.forward_features(im_r)
        #     now_feature = self.encoder.forward_features(im)
        # f_loss = torch.sum((input_feature-now_feature)**2)
        return l2_loss
    

    def photo2sketch(self, render_image):
        render_image = render_image[...,0:3]
        # plt.figure(figsize=(10,10))
        # plt.imshow(render_image.detach().squeeze().cpu().numpy())
        # plt.show()
        render_image = torch.permute(render_image,(0,3,1,2))
        # cv2.imwrite('example1.png', render_image.detach().squeeze().cpu().numpy())
        print("render image shape:",render_image.shape)
        render_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(render_image)
        
        # cv2.imshow('test', render_image.detach().squeeze().cpu().numpy())
        # cv2.waitKey(0)
       
            
        render_sketch = self.p2s(render_image)
        print(render_sketch.shape)
        # plt.figure(figsize=(10,10))
        im = render_sketch.detach().squeeze().cpu().numpy()
        
        return render_sketch
        



    # We will save images periodically and compose them into a GIF.
filename_output = "./teapot_optimization_demo.gif"
writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

# Initialize a model using the renderer, mesh and reference image
model = Model(meshes=teapot_mesh, renderer=silhouette_renderer, image_ref=silhouette,encoder=swint_model,p2s=photo2sketch_model).to(device)

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)



plt.figure(figsize=(10, 10))

_, image_init = model()
plt.subplot(1, 2, 1)
plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3])
plt.grid(False)
plt.title("start")

plt.subplot(1, 2, 2)
plt.imshow(model.image_ref.cpu().numpy().squeeze()[..., 3])
plt.grid(False)
plt.title("Reference silhouette")

plt.show()


loop = tqdm(range(200))
for i in loop:
    optimizer.zero_grad()
    _, render_image = model()
    # sketch = model.photo2sketch(render_image)
    loss = model.rec_loss(render_image)
    print(loss)
    loss.backward()
    optimizer.step()
    
    loop.set_description('Optimizing (loss %.4f)' % loss.data)
    
    if loss.item() < 200:
        break
    
    # Save outputs to create a GIF. 
    if i % 10 == 0:
        # R = look_at_rotation(model.camera_position[None, :], device=model.device)
        # T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]   # (1, 3)
        
        R, T = look_at_view_transform(eye=model.camera_position, at= model.at, device=model.device)

        image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
        image = image[0, ..., :3].detach().squeeze().cpu().numpy()
        image = img_as_ubyte(image)
        writer.append_data(image)
        
        plt.figure()
        plt.imshow(image[..., :3])
        plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
        plt.axis("off")
    
writer.close()