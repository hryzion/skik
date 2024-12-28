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
from clip_loss import LossFunc
import config

room_id = "00a4ff0c-ec69-4202-9420-cc8536ffffe0"

# io utils
from pytorch3d.io import load_obj, load_objs_as_meshes

# datastructures
from pytorch3d.structures import Meshes,join_meshes_as_scene

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate,euler_angles_to_matrix

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    SoftPhongShader
    
)


# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
DATA_DIR =   "/mnt/e/dataset/scenes"

OBJ_DIR = '/mnt/e/dataset/3DFront_p/object/'
scene_filename = os.path.join(DATA_DIR, f"{room_id}.json")


# Load obj file


import json
with open(scene_filename, 'r') as f:
    scene_json = json.load(f)
    f.close()



all_meshes = []
for room in scene_json['rooms']:
    for obj in room['objList']:
        if obj['inDatabase']:
            obj_filename = os.path.join(OBJ_DIR, obj['modelId'],f'{obj['modelId']}.obj')
            mesh = load_objs_as_meshes([obj_filename],device=device)
            translate = torch.Tensor(obj['translate'])
            scale = torch.Tensor(obj['scale'])
            rotate = torch.Tensor(obj['rotate'])

            S = torch.diag(torch.tensor([scale[0],scale[1],scale[2],1.0])).to(device)

            x_axis = torch.tensor([1,0,0]).to(device)
            y_axis = torch.tensor([0,1,0]).to(device)
            z_axis = torch.tensor([0,0,1]).to(device)

            euler_angle = torch.tensor(rotate)
            R = euler_angles_to_matrix(euler_angle,convention=obj['rotateOrder'])
            R = torch.cat((R, torch.zeros(3, 1)), dim=1)
            R = torch.cat((R, torch.tensor([[0, 0, 0, 1.0]])), dim=0)
            R = R.to(device)

            T = torch.eye(4)
            T[:3, 3] = torch.tensor(translate)
            T = T.to(device=device)
            transform = T @ R @ S

            temp = mesh.transform_verts(transform)

            all_meshes.append(temp)

scene = join_meshes_as_scene(all_meshes)



    
img_root_path = './data/sketch'
img_name = 'sketch1.jpg'


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


# Initialize a perspective camera.
cameras = FoVPerspectiveCameras(device=device)

blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
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
    image_size=512, 
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
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# Select the viewpoint using spherical angles  
distance = 20   # distance from camera to the object
elevation = 50.0   # angle of elevation in degrees
azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

eye = torch.Tensor([[0, 2, 4]])
at = torch.Tensor([[0,0,0]])


# Get the position of the camera based on the spherical angles
# R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
R, T = look_at_view_transform(eye=eye,at = at, device=device)

# Render the teapot providing the values of R and T. 
silhouette = silhouette_renderer(meshes_world=scene, R=R, T=T)
image_ref = phong_renderer(meshes_world=scene, R=R, T=T)
transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像大小调整为 224x224
              # 将 PIL.Image 转换为 torch.Tensor
            # 可以添加其他转换操作，例如 Normalize
        ])





class DiffModel(nn.Module):
    def __init__(self, meshes, renderer, image_ref, p2s, camera = None,at = None, sil_render = None):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        self.sil_renderer = sil_render

        self.p2s = p2s
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.camera_position =  torch.from_numpy(np.array([[0, 2.1, 4.1]], dtype=np.float32)).to(meshes.device)
        
        
        self.at = torch.from_numpy(np.array([[0,0,0]], dtype= np.float32)).to(meshes.device)
   
        
    def get_rotate_matrix(self):
        
        Rx = torch.Tensor([
        [1, 0, 0],
        [0, torch.cos(self.theta_x), -torch.sin(self.theta_x)],
        [0, torch.sin(self.theta_x), torch.cos(self.theta_x)]
    ])
    # 绕y轴旋转的旋转矩阵
        Ry = torch.Tensor([
        [torch.cos(self.theta_y), 0, torch.sin(self.theta_y)],
        [0, 1, 0],
        [-torch.sin(self.theta_y), 0, torch.cos(self.theta_y)]
    ])
    # 绕z轴旋转的旋转矩阵
        Rz = torch.Tensor([
        [torch.cos(self.theta_z), -torch.sin(self.theta_z), 0],
        [torch.sin(self.theta_z), torch.cos(self.theta_z), 0],
        [0, 0, 1]
    ])
    # 按照 ZYX 的顺序计算总的旋转矩阵
        R = Rz @ Ry @ Rx
        
        R = R.to(self.device).unsqueeze(0)
        # print(R.grad)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[:, :, None])[:, :, 0]

        return R, T

    def forward(self):
        self.R, self.T = look_at_view_transform(eye=self.camera_position,at = self.at,device=self.device)
        image = self.renderer(meshes_world=self.meshes.clone(), R=self.R, T=self.T)
        # image = self.sil_renderer(meshes_world = self.meshes.clone(), R = self.R, T = self.T
        return image
    
    def phong_render(self):
        self.R, self.T = look_at_view_transform(self.distance,self.elevation,self.azimuth,device=self.device)
        image = self.renderer(meshes_world = self.meshes.clone(), R = self.R, T = self.T)
        return image
    

    def photo2sketch(self, render_image):
        render_image = render_image[...,0:3]
        render_image = torch.permute(render_image,(0,3,1,2))
        render_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(render_image)
        
        render_sketch = self.p2s(render_image) # B, C, H, W     C = 1
        channel1 = render_sketch
        channel2 = render_sketch.clone()
        channel3 = render_sketch.clone()
        rgb_output = torch.cat((channel1, channel2, channel3), dim = 1)
        return rgb_output
    
    def set_camera_at(self,camera = None, at= None):
        if camera.all() != None:
            self.camera_position = torch.from_numpy(camera).unsqueeze(0).to(device)
        if at.all() != None:
            self.at =torch.from_numpy(at).unsqueeze(0).to(device)
    

if __name__ == '__main__':
    import json
    with open(f'/mnt/e/dataset/scenes/{room_id}.json','r') as f:
        scene_json = json.load(f)
    if not os.path.exists(f'./sketch_results/{room_id}/'):
        os.makedirs(f'./sketch_results/{room_id}/')
    if not os.path.exists(f'./sample_results/{room_id}/'):
        os.makedirs(f'./sample_results/{room_id}/')
    
    model = DiffModel(meshes=scene, renderer=phong_renderer, image_ref=image_ref,p2s=photo2sketch_model,sil_render=silhouette_renderer).to(device)
    for i,room in enumerate(scene_json['rooms']):
        count = 0
        for obj in room['objList']:
            if obj['coarseSemantic'] == "Door" or obj['coarseSemantic']=="Window":
                continue
            count+=1
        
        if count == 0:
            continue
        else:
            # sample
            bbox = room['bbox']
            box_ma = np.array(bbox['max'],dtype = np.float32)
            box_mi = np.array(bbox['min'],dtype = np.float32)

            center = (box_ma + box_mi) / 2
            y = center[1]
            box_ma[1] = y
            box_mi[1] = y
            eightpt = [
                box_ma,
                box_mi,
                np.array([box_ma[0], y, box_mi[2]],dtype = np.float32),
                np.array([box_mi[0], y, box_ma[2]],dtype = np.float32),
                np.array([box_ma[0], y, center[2]],dtype = np.float32),
                np.array([center[0], y, box_ma[2]],dtype = np.float32),
                np.array([box_mi[0], y, center[2]],dtype = np.float32),
                np.array([center[0], y, box_mi[2]],dtype = np.float32),
            ]
            print(eightpt)
            print("center: ", center)
            for j, camera in enumerate(eightpt):
                print(camera)
                model.set_camera_at(camera,center)
                rast = model()

                sketch = model.photo2sketch(rast)
                sketch = sketch.detach().cpu().squeeze().numpy().transpose((1,2,0))
                sketch *= 256

                rast = rast[...,:3].detach().cpu().squeeze().numpy()
                rast *= 256
                cv2.imwrite(f'./sample_results/{room_id}/room{i}_camera{j}.jpg',rast)
                cv2.imwrite(f'./sketch_results/{room_id}/room{i}_camera{j}.jpg',sketch)

                



