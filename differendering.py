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
from loss import LossFunc
import config

# io utils
from pytorch3d.io import load_obj,load_objs_as_meshes

# datastructures
from pytorch3d.structures import Meshes,join_meshes_as_scene

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate, euler_angles_to_matrix

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    TexturesAtlas  
)

# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")



room_id = "000ecb5b-b877-4f9a-ab6f-90f385931658"
  
img_root_path = './data/sketch'
img_name = 'sketch1.jpg'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


DATA_DIR = "/mnt/e/dataset/scenes"
OBJ_DIR = '/mnt/e/dataset/3DFront_p/object/'
scene_filename = os.path.join(DATA_DIR, f"{room_id}.json")

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
    image_size=256, 
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

eye = torch.Tensor([[5.4498997, 1.4 ,3  ]])
at = torch.Tensor([[2.0168998, 1.229513,  2.793385 ]])


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


sketch_ref = Image.open(sketch_pt)
sketch_ref = transform(sketch_ref)

class DiffModel(nn.Module):
    def __init__(self, meshes, renderer, image_ref, p2s, sil_render = None):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        self.sil_renderer = sil_render

        self.p2s = p2s
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([[5.4498997 ,1.3 ,1.0902   ]], dtype=np.float32)).to(meshes.device))
        
        self.at = nn.Parameter(
            torch.from_numpy(np.array([[2.0168998, 1.229513,  2.793385 ]], dtype= np.float32)).to(meshes.device)
        )

        self.distance = nn.Parameter(
            torch.from_numpy(np.array([[7]], dtype=np.float32)).to(meshes.device))

        self.elevation = nn.Parameter(
            torch.from_numpy(np.array([[10]], dtype=np.float32)).to(meshes.device))
        self.azimuth = nn.Parameter(
            torch.from_numpy(np.array([[-10]], dtype=np.float32)).to(meshes.device))

        self.theta_x = nn.Parameter(
            torch.Tensor([0]).to(meshes.device)
        )
        self.theta_y = nn.Parameter(
            torch.Tensor([torch.pi]).to(meshes.device)
        )
        self.theta_z = nn.Parameter(
            torch.Tensor([0]).to(meshes.device)
        )

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
        return render_sketch
        


args = config.parse_arguments()

if not os.path.exists(f'./runs/exp{args.exp}'):
    os.mkdir(f'./runs/exp{args.exp}')


filename_output = f"./runs/exp{args.exp}/teapot_optimization_demo.gif"
filename_sketch_output = f'./runs/exp{args.exp}/test_sketch.gif'
writer = imageio.get_writer(filename_output, mode='I', duration=10)
sketch_writer = imageio.get_writer(filename_sketch_output, mode='I',duration=10)

# Initialize a model using the renderer, mesh and reference image
model = DiffModel(meshes=scene, renderer=phong_renderer, image_ref=image_ref,p2s=photo2sketch_model,sil_render=silhouette_renderer).to(device)

with torch.no_grad():
    sketch_ref = model.photo2sketch(image_ref)
    


# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
# optimizer_orient = torch.optim.Adam([model.theta_x,model.theta_y,model.theta_z], lr=0.05)
optimizer_pose = torch.optim.Adam([model.camera_position],lr = 0.02)
# optimizer_at = torch.optim.Adam([model.at],l)



plt.figure(figsize=(10, 10))

image_init = model()
print('image_init:',image_init.shape)
plt.subplot(1, 2, 1)
plt.imshow(image_init[...,0:3].detach().squeeze().cpu().numpy())
plt.grid(False)
plt.title("start")



sketch_test = sketch_ref.clone().permute(0,2,3,1)
plt.subplot(1, 2, 2)
plt.imshow(sketch_test.cpu().numpy().squeeze())
plt.grid(False)
plt.title("reference sketch")
print(sketch_test.shape)
cv2.imwrite(f'./runs/exp{args.exp}/ref_sketch.jpg',sketch_test.cpu().numpy().squeeze()*256)
plt.show()







clip_loss_func = LossFunc(args)
best_loss, best_image = torch.Tensor([1000000]).to(device), 0
best_view = []



loss_record = []
# image_ref = image_ref[...,0:3]
# image_ref = torch.permute(image_ref,(0,3,1,2))

for i in tqdm(range(200)):
    optimizer_pose.zero_grad()
    # optimizer_at.zero_grad()
    
    try:
        image = model()
    except:
        print(model.at, model.camera_position)
    image_c = image[...,0:3]
    image_c = torch.permute(image_c,(0,3,1,2))
    sketch = model.photo2sketch(image)
    # losses_dict = clip_loss_func(image_c, sketch_ref)
    # loss = sum(list(losses_dict.values()))
    loss = torch.mean((sketch-sketch_ref)**2) * sketch.shape[3] * sketch.shape[2] 

    print(loss,'\n')

    loss_record.append(loss.detach().cpu())
    if loss < best_loss:
        best_loss = loss
        best_image = image_c.detach().cpu().squeeze().numpy()
        best_view = (model.camera_position)
        
    loss.backward()
    optimizer_pose.step()
    # optimizer_at.step()
    if loss.item() < 1500:
        break
    
    # Save outputs to create a GIF. 
    if i % 10 == 0:
        image = model.renderer(meshes_world =model.meshes.clone(), R = model.R, T = model.T)
        write_sketch = model.photo2sketch(image).detach().squeeze().cpu().numpy()
        write_sketch = img_as_ubyte(write_sketch)
        sketch_writer.append_data(write_sketch)

        image = image[0, ..., 0:3].detach().squeeze().cpu().numpy()
        
        image = img_as_ubyte(image)
        writer.append_data(image)
        
        # plt.figure()
        # plt.imshow(image[..., :3])
        # plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
        # plt.axis("off")


ep = [i for i in range(len(loss_record))]
plt.plot(ep, loss_record)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.savefig(f'./runs/exp{args.exp}/loss.png')


best_image = best_image.transpose((1,2,0))
best_image = img_as_ubyte(best_image)
print(best_view)

cv2.imwrite(f'./runs/exp{args.exp}/test_best_sketch.jpg',cv2.cvtColor(best_image,cv2.COLOR_RGB2BGR))


writer.close()
sketch_writer.close()