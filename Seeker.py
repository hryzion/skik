import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import config
from Inititalizer import Initializer
from render.Pytorch3dRender import DiffRenderer
from render.NvDiffrastRenderer import NVDiffRastFullRenderer
from nets.networks import ResnetGenerator
import json
from loss import LossFunc
from pytorch3d.renderer import (
    look_at_rotation_nvdiff
)
import tqdm
from utils import *
args = config.parse_arguments()

class SceneSeeker:
    def __init__(self,args):
        # 初始化ViewInitializer
        self.args = args
        self.device = args.device
        with open(args.settings, 'r') as f:
            self.settings = json.load(f)
            f.close()
        
        # 初始化场景mesh和
        self.initializer = Initializer(args)
        self.rooms_meshes_list = self.initializer.room_mesh_list

        # 初始化Renderer
        self.renderer = NVDiffRastFullRenderer(self.device, self.settings,args.res,1)

        # 根据gt渲染出target_img
        self.gt_view = self.settings['gt_view']
        mtce = get_camera_matrix(self.gt_view, self.device)
        with torch.no_grad():
            self.gt_img = self.renderer.render(view=mtce,scene_mesh=self.rooms_meshes_list,DcDt=False)    
        
        
    def seek(self):
        # 使用ViewInitializer初始化视点, dict
        view = self.initializer.initialize_view()
        epoch = self.args.epoch
        self.position = view['position']
        self.center = view['center']
        # self.fov = init_views['fov']
        self.optimizer = torch.optim.Adam([self.position, self.center],lr = 0.02)

        # 进入主循环
        loss_func = LossFunc(args)
        for i in tqdm.tqdm(range(epoch)):
            # 梯度清零
            self.optimizer.zero_grad()
            # 根据view计算camera_mtx & perspect_mtx
            mtce = get_camera_matrix(view, self.device)
            # Renderer渲染 (img & pos)
            render_res = self.renderer.render(mtce, scene_mesh=self.rooms_meshes_list,DcDt=False)
            # 计算SinkHornLoss
            losses_dict = loss_func(render_res, self.gt_img)
            loss = sum(list(losses_dict.values()))
            # Backwards
            loss.backward()
            self.optimizer.step()

            if i % self.args.save_interval == 0:
                pass
        
        # 根据position和center渲染最终图像
        with torch.no_grad():
            mtce = get_camera_matrix(view, self.device)
            final_res = self.renderer.render()
            final_img = final_res['image']
        
        # 计算总体差距 position & direction(norm(p-c))
        view_mae = calculate_view_mae(view, self.gt_view)
        
        

        


