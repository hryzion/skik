import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights
)




class DiffRenderer(nn.Module):
    def __init__(self, meshes, p2s, args= None):
        super().__init__()

        # mesh and device
        self.meshes = meshes
        self.device = args.device

        # renderer
        cameras = FoVPerspectiveCameras(device=self.device)
        raster_settings = RasterizationSettings(
            image_size=256, 
            blur_radius=0.0, 
            faces_per_pixel=1,
        )
        lights = PointLights(device=self.device, location=((2.0, 2.0, -2.0),))
        self.renderer = MeshRenderer(
            rasterizer = MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader = HardPhongShader(device=self.device, cameras=cameras, lights=lights)
        )
        
        self.p2s = p2s
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([[7, 1.5 ,5]], dtype=np.float32)).to(meshes.device))
        
        self.at = nn.Parameter(
            torch.from_numpy(np.array([[2.0168998, 1.229513,  2.793385 ]], dtype= np.float32)).to(meshes.device)
        )

    def forward(self):
        self.R, self.T = look_at_view_transform(eye=self.camera_position,at = self.at,device=self.device)
        image = self.renderer(meshes_world=self.meshes.clone(), R=self.R, T=self.T)
        return image

    def phong_renderer(self):
        self.R, self.T = look_at_view_transform(self.distance,self.elevation,self.azimuth,device=self.device)
        image = self.renderer(meshes_world = self.meshes.clone(), R = self.R, T = self.T)
        return image
    
    def photo2sketch(self, render_image):
        render_image = render_image[...,0:3]
        render_image = torch.permute(render_image,(0,3,1,2))
        render_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(render_image)
        render_sketch = self.p2s(render_image) # B, C, H, W     C = 1
        return render_sketch

    def visualize_img(self, img):
        if img.shape[1] == 1:
            img = img.repeat(1,3,1,1)
            img = img.clone().permute(0,2,3,1)  # 1,h,w,c
        
        return img.detach().cpu().squeeze().numpy()
    
    def get_image_ref(self, camera, look_at):
        R, T = look_at_view_transform(eye= camera,at=look_at,device=self.device)
        with torch.no_grad():
            return self.renderer(meshes_world=self.meshes.clone(),R = R,T = T)
        
