import torch
import torch.nn as nn
import nvdiffrast.torch as dr
import torch.nn.functional as F
from pytorch3d.renderer.mesh.textures import TexturesUV, TexturesVertex
import numpy as np
import cv2,os
import torchvision.transforms as transforms
from torchviz import make_dot
def transform_pos(mtx, pos): 
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx #4x4
    if len(t_mtx.shape)==2:
        t_mtx = t_mtx[None]
    # (x,y,z) -> (x,y,z,1)
    if pos.shape[-1]==3:
        posw = torch.nn.functional.pad(pos, pad=(0,1), mode='constant', value=1.0)
    else:
        posw=pos
    if len(t_mtx.shape)==2:
        return torch.matmul(posw, t_mtx.t())[None,...]
    else:
        return torch.matmul(posw, t_mtx.permute(0,2,1))

class NVDiffRastFullRenderer(nn.Module):
    def __init__(self, device, settings, resolution, num_views=None):
        super().__init__()
        self.device = device
        self.glctx = dr.RasterizeCudaContext()
        bg = settings.get("background","white")
        self.multi_bg=False
        if bg=="white":
            self.background=torch.ones((1,resolution[1],resolution[0],3)).cuda()
        elif bg=="black":
            self.background=torch.zeros((1,resolution[1],resolution[0],3)).cuda()
        elif bg=="random":
            self.background=torch.rand((1,resolution[1],resolution[0],3)).cuda()
        else:
            if num_views==None:
                bg = cv2.imread(bg)
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                bg = cv2.flip(bg,0)
                bg = cv2.resize(bg,(resolution[0],resolution[1]))*1.0/255.0
                self.background = torch.tensor(bg).cuda().reshape((1,resolution[1],resolution[0],3)).float()
            else:
                self.backgrounds=[]
                bg_folder=bg
                for i in range(num_views):
                    bg = cv2.imread(os.path.join(bg_folder,"%d_bg.png"%(i)))
                    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                    bg = cv2.flip(bg,0)
                    bg = cv2.resize(bg,(resolution[0],resolution[1]))*1.0/255.0
                    self.backgrounds.append(torch.tensor(bg)[None].float())
                self.background = torch.cat(self.backgrounds).cuda()
                self.multi_bg=True
        
        self.light_power = settings.get("light_power",1.0)
        self.shading = settings.get("shading",True)
        self.resolution = resolution
        envlight_path = settings.get("envlight_path",None)
        if envlight_path!=None:
            envlight = np.load(envlight_path)
            self.envmap = torch.from_numpy(envlight).cuda().float()
        else:
            self.envmap=None
    
    def render_point_image(self, cam_mtx, persp_mtx, rast_out, pos, pos_idx, resolution: int):
        pos_view = transform_pos(cam_mtx, pos)
        msk = (rast_out[...,3]!=0)
        # print(pos_view.shape)
        pos_3d,_ = dr.interpolate(pos_view, rast_out, pos_idx)
        depth = -pos_3d[...,2]
        depth[rast_out[...,3]==0]=1000000
        s = pos_3d.shape
        pos_3d = transform_pos(persp_mtx, pos_3d.reshape(-1,4)).reshape(s)
        pos_2d = pos_3d[...,:2]/pos_3d[...,3:]
        return depth, pos_2d, msk    
    
    def render_semantic_image(self,  color_list, pos_idx, rast_out):
        tex = color_list.to(self.device)
        semantic_color, _ = dr.interpolate(tex[None, ...], rast_out,pos_idx)
        return semantic_color


    def render_mesh(self, mesh, mtx, view_pos, light_pos, light_power, resolution, material=None, rast_out=None, envmap=None, color_list = None, semantic_list=None):
        pos = mesh.verts_list()[0].to(self.device).contiguous()
        pos_idx = mesh.faces_list()[0].to(self.device).to(torch.int32).contiguous()

        color_list = color_list.to(self.device)
        semantic_list = semantic_list.to(self.device)
        if type(mesh.textures)==TexturesUV:
            uv_idx = mesh.textures.faces_uvs_list()[0].to(self.device).to(torch.int32)
            uv = mesh.textures.verts_uvs_list()[0].to(self.device)
            tex = mesh.textures.maps_list()[0].to(self.device)
            tex = torch.flipud(tex)
        elif type(mesh.textures)==TexturesVertex:
            tex = mesh.textures.verts_features_list()[0].to(self.device)
            uv = None
            uv_idx = None
        else:
            uv = None
            uv_idx = None
            tex = None

        # pos = pos.detach()

        pos_clip = transform_pos(mtx, pos)
        # if pos_clip.requires_grad:
        #     pos_clip.retain_grad()
        # self.pos_clip = pos_clip
        # if mtx.requires_grad:
        #     mtx.retain_grad()
        # self.mtx = mtx
        
        if rast_out==None:
            rast_out, rast_out_db = dr.rasterize(self.glctx, pos_clip, pos_idx, resolution=[resolution[1], resolution[0]]) 
        else:
            rast_out_db=None
        
        # if rast_out.requires_grad:
        #     rast_out.retain_grad()
        # self.rast_out = rast_out


        # if DcDt==False:
        rast_out = rast_out.detach()
        
        if tex != None and uv!=None:
            texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
            color = dr.texture(tex[None, ...], texc, filter_mode='linear')
            # if color.requires_grad:
            #     g = make_dot(color)
            #     g.render("./color")
            # if texc.requires_grad:
            #     texc.retain_grad()
            #     self.texc = texc
            
                
        elif tex!=None:
            color,_ = dr.interpolate(tex[None, ...], rast_out, pos_idx)
        else:
            color = torch.zeros_like(rast_out)[...,:3]
        
        semantics, _ = dr.interpolate(color_list[None, ...],rast_out, pos_idx)
        s_channels, _ = dr.interpolate(semantic_list[None,...],rast_out,pos_idx)
        # if semantics.requires_grad:
        #     semantics.retain_grad()
        # self.semantics = semantics
        # if color.requires_grad:
        #     g = make_dot(semantics)
        #     g.render("./semantics")
        #     exit()
        for i in range(rast_out.shape[0]):
            color[i][rast_out[i,..., -1]==0]=self.render_result[i][rast_out[i,..., -1]==0]
        
        # if DcDt:
        #     color = dr.antialias(color.contiguous(), rast_out, pos_clip, pos_idx)
        
        return color, [rast_out], [rast_out_db], semantics, s_channels
    
    def render(self, view, scene_mesh, color_list, semantic_list, rasts_list=None):
        meshes = scene_mesh
        perspective_matrix = view["perspective_mtx"]
        resolution = self.resolution
        vp = view['view_mtx']
        camera_matrix = view["camera_mtx"]
        lookFrom = view['position']
        num_view = 1
        self.render_result = self.background[...,:3].clone()
        depth_buffer = torch.ones((num_view,self.resolution[1],self.resolution[0])).to(self.device)*1000000000.0
        pos_buffer = torch.ones((num_view,self.resolution[1],self.resolution[0],2)).to(self.device)*-1
        msk_buffer = torch.zeros((num_view,self.resolution[1],self.resolution[0]),dtype=torch.bool).to(self.device)
        new_rasts_list = []

        render_image, rasts, dbs, semantic_img, semantics_channels= self.render_mesh(mesh = meshes, mtx=vp, view_pos=lookFrom, light_pos=lookFrom, light_power=self.light_power, resolution=resolution, color_list=color_list, semantic_list=semantic_list)
        new_rasts_list.append([rasts,dbs])
        # if render_image.requires_grad:
        #     render_image.retain_grad()
        # print(render_image.requires_grad)
        pos = meshes.verts_list()[0].to(self.device)
        pos_idx = meshes.faces_list()[0].to(self.device).to(torch.int32).contiguous()
        
        # semantic_img = self.render_semantic_image(color_list, pos_idx, rasts[-1])
        depth_map, pos_map, msk = self.render_point_image(camera_matrix, perspective_matrix, rasts[-1].detach(), pos, pos_idx, resolution)      
        # print(pos_map.requires_grad)
        # print(semantic_img.requires_grad)
        
        msk_buffer = msk_buffer | msk
        pos_buffer = pos_map
        self.render_result = render_image
        depth_buffer = depth_map

        res = {"images":self.render_result,
                "depth":depth_buffer,
                "msk":msk_buffer,
                "pos": pos_buffer,
                "rasts": new_rasts_list,
                "semantics":semantic_img,
                "semantics_c":semantics_channels
            }
        
        return res



    def _render(self,scene,DcDt,rasts_list=None,view=None):
        envlight = scene.get("envlight",self.envmap)
        meshes = scene["meshes"]
        
        sensor = scene["sensors"][0]
        material = scene["material"]
        resolution = self.resolution[0]#sensor["resolution"][0]
        perspective_matrix = sensor["perspective_matrix"]
        if view==None:
            vp = sensor["matrix"]
            camera_matrix = sensor["camera_matrix"]
            lookFrom = sensor["position"]
        else:
            vp = sensor["matrix"][view:view+1]
            camera_matrix = sensor["camera_matrix"][view:view+1]
            lookFrom = sensor["position"][view:view+1]
        
        views = vp.shape[0]

        # print("camera matrix: ",camera_matrix[0])
        # print('projection matrix: ',perspective_matrix[0])
        # print('matrix total', vp[0])
        # print('projection cross camera', torch.matmul(perspective_matrix[0], camera_matrix[0]))
        if self.multi_bg:
            if view!=None:
                self.render_result = self.background[view:view+1,...,:3].clone().repeat((views,1,1,1))
            else:
                self.render_result = self.background[...,:3].clone()
        else:
            self.render_result = self.background[...,:3].clone().repeat((views,1,1,1))
        
        depth_buffer = torch.ones((views,self.resolution[1],self.resolution[0])).to(self.device)*1000000000.0
        pos_buffer = torch.ones((views,self.resolution[1],self.resolution[0],2)).to(self.device)*-1
        msk_buffer = torch.zeros((views,self.resolution[1],self.resolution[0]),dtype=torch.bool).to(self.device)
        new_rasts_list = []

        for id,mesh_info in enumerate(meshes):
            mesh = mesh_info.get("model")
            if rasts_list is None:
                
                render_image, rasts, dbs = self.render_mesh(mesh = mesh, mtx=vp, view_pos=lookFrom, light_pos=lookFrom, light_power=self.light_power, resolution=resolution, DcDt=DcDt, envmap=envlight, material=material)
                # Logger.save_img("img_gt.png", render_image[0].cpu().numpy())
                
                new_rasts_list.append([rasts,dbs])
            else:
                render_image, rasts, dbs = self.render_mesh(mesh = mesh, mtx=vp, view_pos=lookFrom, light_pos=lookFrom, light_power=self.light_power, resolution=resolution, rast_out = rasts_list[id][0][0], DcDt=DcDt, envmap=envlight, material=material)
                rasts = rasts_list[id][0]   
            
            pos = mesh.verts_list()[0].to(self.device)
            pos_idx = mesh.faces_list()[0].to(self.device).to(torch.int32).contiguous()
            depth_map, pos_map, msk = self.render_point_image(camera_matrix, perspective_matrix, rasts[-1].detach(), pos, pos_idx, resolution)      
            
            msk_buffer = msk_buffer | msk
            pos_buffer = pos_map
            self.render_result = render_image
            depth_buffer = depth_map
        
        # Logger.save_img("img_gt.png", self.render_result[0].cpu().numpy())
        # self.render_sketch = self.photo2sketch(self.render_result)
        res = {"images":self.render_result,
                "depth":depth_buffer,
                "msk":msk_buffer,
                "pos": pos_buffer,
                "rasts": new_rasts_list,
                # "sketch" : self.render_sketch
            }
        
        return res

    