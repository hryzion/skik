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
import os
import imageio
import tqdm
from utils import *
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

args = config.parse_arguments()
class SceneSeeker:
    def __init__(self,args):
        # 初始化ViewInitializer
        self.args = args
        self.device = args.device
        if not os.path.exists(f'./runs/exp{args.exp}'):
            os.mkdir(f'./runs/exp{args.exp}')
        with open(os.path.join("./experiments",args.settings+".json"), 'r') as f:
            self.settings = json.load(f)
            f.close()
        
        # 初始化场景mesh和
        self.initializer = Initializer(args)
        self.rooms_meshes_list = self.initializer.room_mesh_list
        self.color_list = self.initializer.color_list
        self.resolution = (args.res, args.res)
        # 初始化Renderer
        self.renderer = NVDiffRastFullRenderer(self.device, self.settings,self.resolution,1)

        # 根据gt渲染出target_img
        self.gt_view = self.settings['gt_view']
        mtce = get_camera_matrix(self.gt_view, self.device)
        with torch.no_grad():
            self.gt_img = self.renderer.render(view=mtce,scene_mesh=self.rooms_meshes_list[self.gt_view['room_id']], color_list=self.color_list[self.gt_view["room_id"]],DcDt=False)    
        save_gt = self.gt_img["images"].cpu().squeeze(0).numpy()
        save_gt = get_visualized_img(save_gt,use_cv2=True)
        save_semantic = self.gt_img['semantics'].cpu().squeeze(0).numpy()
        save_semantic = get_visualized_img(save_semantic, use_cv2=True)
        cv2.imwrite(f"./runs/exp{args.exp}/gt_img.png",save_gt)
        cv2.imwrite(f'./runs/exp{args.exp}/gt_semantic.png', save_semantic)
        
        self.initializer.set_gt_img(self.gt_img)
        
    def seek(self):
        # 使用ViewInitializer初始化视点, dict
        view = self.initializer.initialize_view(self.renderer)
        if self.settings['mode'] == "manual":
            view = self.settings['init_view']
        
        epoch = self.args.epoch
        self.position = torch.tensor(view['position'], device=self.device, requires_grad=True)
        self.center = torch.tensor(view['center'], device=self.device, requires_grad=True)
        self.room_id = view['room_id']
        # self.fov = init_views['fov']
        self.optimizer = torch.optim.Adam([self.position, self.center],lr = 0.05)
        
        
        filename_output = f"./runs/exp{args.exp}/matching_video.gif"

        writer = imageio.get_writer(filename_output, mode='I', duration=10)

        loss_record = []
        # 进入主循环
        loss_func = LossFunc(args)
        for i in tqdm.tqdm(range(epoch)):
            # 梯度清零
            self.optimizer.zero_grad()
            grad_view = {
                'position':self.position,
                'center': self.center
            }

            # 根据view计算camera_mtx & perspect_mtx
            R = look_at_rotation_nvdiff(self.position, self.center, device=self.device)
            T = -torch.bmm(R.transpose(1, 2), self.position[:, :, None])[:, :, 0]
            camera_mtx = get_world_to_view_transform(R=R, T=T).get_matrix().transpose(1,2).to(self.device)
            fov = 60.0
            znear = 0.01
            zfar = 100
            perspective_mtx = torch.tensor(np.array(glm.perspective(
                glm.radians(fov), 1.0, znear, zfar)), device=self.device)
            view_mtx = torch.matmul(perspective_mtx, camera_mtx).to(self.device)
            mtce = {
                "view_mtx" : view_mtx,
                "camera_mtx" : camera_mtx,
                "perspective_mtx" : perspective_mtx,
                "position":self.position,
                "center":self.center
            }
            # print(mtce)
            # Renderer渲染 (img & pos)
            render_res = self.renderer.render(mtce, scene_mesh=self.rooms_meshes_list[self.room_id],DcDt=False)
            # 计算SinkHornLoss
            losses_dict = loss_func(render_res, self.gt_img)
            # print(losses_dict)
            loss = sum(list(losses_dict.values()))
            # Backwards
            loss.backward()
            self.optimizer.step()

            loss_record.append(loss.detach().cpu())
            

            if i % self.args.save_interval == 0 or i == epoch -1:
                save_img = render_res['images'].clone().detach().cpu().squeeze(0).numpy()
                save_img = get_visualized_img(save_img)
                writer.append_data(save_img)
        writer.close()

        ep = [i for i in range(len(loss_record))]
        plt.plot(ep, loss_record)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.savefig(f'./runs/exp{args.exp}/loss.png')
        
        # 根据position和center渲染最终图像
        with torch.no_grad():
            grad_view = {
                'position':self.position,
                'center': self.center
            }
            mtce = get_camera_matrix(grad_view, self.device)
            final_res = self.renderer.render(mtce, self.rooms_meshes_list[self.room_id],DcDt=False)
            final_img = final_res['images'].detach().cpu().squeeze(0).numpy()
            final_img = get_visualized_img(final_img,use_cv2=True)
            cv2.imwrite(f"./runs/exp{args.exp}/final.png", final_img)

        
        # 计算总体差距 position & direction(norm(p-c))
            view_mae = calculate_view_mae(grad_view, self.gt_view)
            print(f"The final view mae: {view_mae}")
        



if __name__ == "__main__":
    args = config.parse_arguments()
    seeker = SceneSeeker(args)
    seeker.seek()