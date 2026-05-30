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
from  torchviz import make_dot
from config import DATA_DIR, OBJ_DIR, category_to_color, fine_to_coarse, coarse_categories, coarse_to_color

from pytorch3d.transforms import Rotate, Translate, euler_angles_to_matrix
from pytorch3d.structures import Meshes,join_meshes_as_scene
from pytorch3d.io import load_obj,load_objs_as_meshes

args = config.parse_arguments()
class SceneSeeker:
    def __init__(self,args):
        # 初始化ViewInitializer
        self.args = args
        self.device = args.device

        if not args.val:
            self.base_dir = './experiments'
            self.save_dir_root = f'./runs/exp{args.exp}'
            self.save_dir = self.save_dir_root
            if not os.path.exists(f'./runs/exp{args.exp}'):
                os.mkdir(f'./runs/exp{args.exp}')
            os.makedirs(f'./runs/exp{args.exp}/interval_image',exist_ok=True)
            with open(os.path.join(self.base_dir,args.settings+".json"), 'r') as f:
                self.settings = json.load(f)
            f.close()
            self.initializer = Initializer(args,scene_name=self.settings['scene'])

        else:
            self.base_dir = os.path.join(config.VIEWS_BASE_DIR, args.scene, "views", 'test')
            self.save_dir_root = os.path.join(f"./runs/{args.scene}/")
            os.makedirs(self.save_dir_root,exist_ok=True)
            self.initializer = Initializer(args,scene_name=args.scene)
        
        # 初始化场景mesh和
        self.rooms_meshes_list = self.initializer.room_mesh_list
        self.color_list = self.initializer.color_list
        self.semantic_list = self.initializer.semantic_list
        self.resolution = (args.width, args.height)

        # 初始化Renderer
        self.renderer = NVDiffRastFullRenderer(self.device ,self.resolution,1)

        
    def set_settings(self, settings):
        self.settings = settings
            
    def get_gt(self):
        self.gt_scene_name = self.settings['gt_scene']
        with open(os.path.join(self.initializer.dataset_dir, self.gt_scene_name+'.json'),'r') as f:
            self.gt_scene = json.load(f)
            f.close()

        # 根据gt渲染出target_img
        self.gt_view = self.settings['gt_view']
        self.gt_room = self.gt_scene["rooms"][self.gt_view['room_id']]
        self.gt_mesh, self.gt_color_list,self.gt_semantics = self.initializer.load_room_as_scene(self.gt_room)
        
        mtce = get_camera_matrix(self.gt_view, self.device,aspect=args.width/args.height)
        with torch.no_grad():
            self.gt_img = self.renderer.render(view=mtce,scene_mesh=self.gt_mesh, color_list=self.gt_color_list,semantic_list=self.gt_semantics)    
                
        
        save_gt = self.gt_img["images"].cpu().squeeze(0).numpy()
        save_gt = get_visualized_img(save_gt,use_cv2=True)
        save_semantic = self.gt_img['semantics'].cpu().squeeze(0).numpy()
        save_semantic = get_visualized_img(save_semantic, use_cv2=True)

        cv2.imwrite(f"{self.save_dir}/gt_img.png",save_gt)
        cv2.imwrite(f'{self.save_dir}/gt_semantic.png', save_semantic)
        
            
        self.initializer.set_gt_img(self.gt_img)
        self.gt_set = self.initializer.gt_set
    
    def semantic_segment(self):
        pass


    def evaulate_room(self):
        s_count = 81
        correct = 77
        for settings_name in tqdm.tqdm(os.listdir(self.base_dir)[0:27]):
            s_count+=1
            self.save_dir = './tmp'
            # print(f"processing {settings_name}...")
            with open(os.path.join(self.base_dir,settings_name),'r') as f:
                settings = json.load(f)
                self.set_settings(settings)
                f.close()
            if  self.test_room_id():
                correct += 1
            
        
        print(f"Total {s_count} rooms, {correct} rooms are correct!")
        print(f"Accuracy: {correct/s_count:.4f}")


    def evaluate(self):
        sum_count=0
        acc_count = {}

        for i in range(1,6):
            acc_count[str(i*200)]=0

        records = {
            'rgb' : {
                't' : {
                    '0.5' : acc_count.copy(),
                    '0.1' : acc_count.copy(),
                    '0.01' : acc_count.copy()
                },
                'r':{
                    '10' : acc_count.copy(),
                    '1' : acc_count.copy(),
                    '0.5' : acc_count.copy()
                }
            },
            'combine' : {
                't' : {
                    '0.5' : acc_count.copy(),
                    '0.1' : acc_count.copy(),
                    '0.01' : acc_count.copy()
                },
                'r':{
                    '10' : acc_count.copy(),
                    '1' : acc_count.copy(),
                    '0.5' : acc_count.copy()
                }
            },
            'l2' : {
                't' : {
                    '0.5' : acc_count.copy(),
                    '0.1' : acc_count.copy(),
                    '0.01' : acc_count.copy()
                },
                'r':{
                    '10' : acc_count.copy(),
                    '1' : acc_count.copy(),
                    '0.5' : acc_count.copy()
                }
            }
        }


        
        for settings_name in tqdm.tqdm(os.listdir(self.base_dir)[0:27]):
            sum_count+=1
            print(f"processing {settings_name}...")
            with open(os.path.join(self.base_dir,settings_name),'r') as f:
                settings = json.load(f)
                self.set_settings(settings)
                f.close()
            os.makedirs(os.path.join(self.save_dir_root, settings_name.split('.')[0]), exist_ok=True)
            

            
            for mode in ['l2', 'rgb', 'combine']:
                self.save_dir = os.path.join(self.save_dir_root, settings_name.split('.')[0],mode)
                os.makedirs(self.save_dir,exist_ok=True)
                if mode == 'l2':
                    args.l2_loss = True
                    loss_func = LossFunc(args)
                    self.seek(loss_func,records[mode])
                    args.l2_loss = False
                elif mode == 'rgb':
                    args.sample_loss = True
                    args.mode = 'rgb'
                    loss_func = LossFunc(args)
                    self.seek(loss_func,records[mode])
                    args.sample_loss = False
                
                elif mode == "combine":
                    args.sample_loss = True
                    args.mode = 'combine'
                    loss_func = LossFunc(args)
                    self.seek(loss_func,records[mode])
                    args.sample_loss = False
        
        records['sum'] = sum_count
        with open(os.path.join(self.save_dir_root, 'records.json'),'w') as f:
            json.dump(records,f)
            f.close()
        # print(records)
        # for md, rcd  in records.items():
        #     print()
        #     for it, num in rcd.items():
        #         print(f"{md}  Iter {it} with Accuracy: {num/sum_count :.4f}")

    def refine_meshes(self, room):
        all_meshes = []

        count = 0
        scene_vert_color = None
        scene_vert_semantic = None
        print("self.initializer.gt_set",self.initializer.gt_set)
        for obj in room['objList']:
            if obj['coarseSemantic'] == 'Bed Frame':
                continue
            if obj['inDatabase'] and fine_to_coarse[obj['coarseSemantic']] in self.initializer.gt_set:
                obj_filename = os.path.join(OBJ_DIR, obj['modelId'],f'{obj["modelId"]}.obj')
                mesh = load_objs_as_meshes([obj_filename],device=self.device)
                translate = torch.Tensor(obj['translate'])
                scale = torch.Tensor(obj['scale'])
                rotate = torch.Tensor(obj['rotate'])

                S = torch.diag(torch.tensor([scale[0],scale[1],scale[2],1.0])).to(self.device)

                euler_angle = torch.tensor(rotate)
                R = euler_angles_to_matrix(euler_angle,convention=obj['rotateOrder'])
                R = torch.cat((R, torch.zeros(3, 1)), dim=1)
                R = torch.cat((R, torch.tensor([[0, 0, 0, 1.0]])), dim=0)
                R = R.to(self.device)

                T = torch.eye(4)
                T[:3, 3] = torch.tensor(translate)
                T = T.to(device=self.device)
                transform = T @ R @ S

                temp = mesh.transform_verts(transform)
                all_meshes.append(temp)
                obj_semantic = torch.zeros((len(mesh.verts_list()[0]), len(coarse_categories)))
                obj_semantic[..., coarse_categories.index(fine_to_coarse[obj['coarseSemantic']])] = 1
                obj_color = torch.tensor([coarse_to_color[fine_to_coarse[obj['coarseSemantic']]].copy()] * len(mesh.verts_list()[0]))
                if count == 0:
                    scene_vert_color = obj_color
                    scene_vert_semantic = obj_semantic
                else:
                    scene_vert_color = torch.cat([scene_vert_color, obj_color], dim=0)
                    scene_vert_semantic = torch.cat([scene_vert_semantic, obj_semantic],dim=0)
                count +=1 
        
        scene = join_meshes_as_scene(all_meshes)
        return scene, scene_vert_color, scene_vert_semantic

    
    def test_room_id(self):
        self.get_gt()
        view = self.initializer.initialize_view(self.renderer, 'top')

        room_id = view['room_id']
        if room_id == self.gt_view['room_id']:
            return True
        return False


    def seek(self, loss_func, record = None, record_all = None):
        self.get_gt()
        # 使用ViewInitializer初始化视点, dict
        view = self.initializer.initialize_view(self.renderer, '4d')
        if self.settings['mode'] == "manual":
            view = self.settings['init_view']

        refined_scene, refined_vert_color, refined_vert_semantic = self.refine_meshes(self.initializer.scene['rooms'][view['room_id']])
        print(refined_vert_color.shape)
        epoch = self.args.epoch
        self.position = torch.tensor(view['position'], device=self.device, requires_grad=True)
        self.center = torch.tensor(view['center'], device=self.device, requires_grad=True)
        self.room_id = view['room_id']
        # self.fov = init_views['fov']
        self.optimizer = torch.optim.Adam([self.position, self.center],lr = 0.05)
        
        
        filename_output = f"{self.save_dir}/matching_video.gif"

        writer = imageio.get_writer(filename_output, mode='I', duration=10)

        loss_record = []
        # 进入主循环

        pose_record = []
        flag_found = False
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
            fov = 60
            znear = 0.01
            zfar = 100
            perspective_mtx = torch.tensor(np.array(glm.perspective(
                glm.radians(fov), self.args.width/self.args.height, znear, zfar)), device=self.device)
            view_mtx = torch.matmul(perspective_mtx, camera_mtx).to(self.device)
            view_mtx.retain_grad()
            mtce = {
                "view_mtx" : view_mtx,
                "camera_mtx" : camera_mtx,
                "perspective_mtx" : perspective_mtx,
                "position":self.position,
                "center":self.center
            }
            # print(mtce)
            # Renderer渲染 (img & pos)
            render_res = self.renderer.render(mtce, scene_mesh=refined_scene, color_list=refined_vert_color,semantic_list=refined_vert_semantic,l2 = args.l2_loss)
            # 计算SinkHornLoss
            # render_res['semantics'].retain_grad()
            losses_dict = loss_func(render_res, self.gt_img)

            loss = sum(list(losses_dict.values()))
            loss.backward()

            self.optimizer.step()

            loss_record.append(loss.detach().cpu())
            

            if i % self.args.save_interval == 0 or i == epoch -1:
                img = render_res['images'].clone().detach().cpu().squeeze(0).numpy()
                save_img = get_visualized_img(img)
                interval_img = get_visualized_img(img,use_cv2=True)
                cv2.imwrite(f"{self.save_dir}/interval_image/interval_{i}.png",interval_img)
                writer.append_data(save_img)
                grad_view = {
                    'position':self.position,
                    'center': self.center
                }
                pose_record.append(to_direction(grad_view, save = True))

            
            if record is not None:
                if (i+1) % 200 == 0:
                    view_mae = calculate_view_mae(grad_view, self.gt_view)
                    if view_mae[0] < 0.5:
                        record['t']['0.5'][str(i+1)] += 1
                        if view_mae[0] < 0.1:
                            record['t']['0.1'][str(i+1)] += 1
                            if view_mae[0] < 0.01:
                                record['t']['0.01'][str(i+1)] += 1
                    if view_mae[1] < 10:
                        record['r']['10'][str(i+1)] += 1
                        if view_mae[1] < 1:
                            record['r']['1'][str(i+1)] += 1
                            if view_mae[1] < 0.5:
                                record['r']['0.5'][str(i+1)] += 1
                    
                    
                    
                    # record[str(i+1)] = view_mae

            if record_all is not None:
                if (i+1) % 200 == 0:
                    view_mae = calculate_view_mae(grad_view, self.gt_view)
                    if view_mae[0] < 0.5 and view_mae[1] < 10:
                        record_all[str(i+1)] += 1




        writer.close()

        ep = [i for i in range(len(loss_record))]
        plt.plot(ep, loss_record)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.savefig(f'{self.save_dir}/loss.png')
        plt.cla()
        
        # 根据position和center渲染最终图像
        with torch.no_grad():
            grad_view = {
                'position':self.position,
                'center': self.center,
                'room_id' : self.room_id
            }
            mtce = get_camera_matrix(grad_view, self.device, aspect=self.args.width/self.args.height)
            final_res = self.renderer.render(mtce, self.rooms_meshes_list[self.room_id],self.color_list[self.room_id],self.semantic_list[self.room_id])
            final_img = final_res['images'].detach().cpu().squeeze(0).numpy()
            final_img = get_visualized_img(final_img,use_cv2=True)
            cv2.imwrite(f"{self.save_dir}/final.png", final_img)

            final_semantic = final_res['semantics'].detach().cpu().squeeze(0).numpy()
            final_semantic = get_visualized_img(final_semantic,use_cv2=True)
            cv2.imwrite(f"{self.save_dir}/final_semantic.png",final_semantic)

        
        # 计算总体差距 position & direction(norm(p-c))
            view_mae = calculate_view_mae(grad_view, self.gt_view)
            print(f"The final view mae: {view_mae[0]}m, {view_mae[1]}d")

            if view_mae[0] < 0.5 and view_mae[1] < 10:
                print("The final view is correct!")
                flag_found = True
        
        with open(f"{self.save_dir}/pose_record.json",'w') as f:
            json.dump(
                {
                    "pose":pose_record,
                    "scene": self.settings['scene'],
                    'view': {
                        'position':self.position.cpu().detach().squeeze(0).numpy().tolist(),
                        'center': self.center.cpu().detach().squeeze(0).numpy().tolist(),
                        'room_id' : self.room_id
                    }
                },f
            )
        f.close()
        
        print(grad_view)
        return grad_view, flag_found




batch = [
    'scene_1',
    'scene_2',
    'scene_3',
    'scene_4',
    'scene_5',
    'scene_6',
    'scene_7',
    'scene_8',
    'scene_9',
    'scene_10',
    'scene_110',
    'scene_111',
    'scene_112',
    'scene_113',
    'scene_114',
    'scene_115',
    'scene_116',
    'scene_118',
    'scene_119',
    'scene_120',
    'scene_121',
    'scene_122',
    'scene_123',
    'scene_124',
    'scene_125',
    'scene_126',
    'scene_127',
    'scene_128',
    'scene_129',
    'scene_130',
]





if __name__ == "__main__":
    args = config.parse_arguments()

    if args.val_manual:
        record_all = {
            'rgb' : {
                '200' : 0,
                '400' : 0,
                '600' : 0,
                '800' : 0,
                '1000' : 0
            },
            'combine' : {
                '200' : 0,
                '400' : 0,
                '600' : 0,
                '800' : 0,
                '1000' : 0
            },
            'l2' : {
                '200' : 0,
                '400' : 0,
                '600' : 0,
                '800' : 0,
                '1000' : 0
            }
        }
        for setting in tqdm.tqdm(batch):
            args.settings = setting
            args.exp = 100 + batch.index(setting)
            seeker = SceneSeeker(args)
            for mode in ['l2', 'rgb', 'combine']:
                seeker.save_dir = os.path.join(seeker.save_dir_root ,mode)
                os.makedirs(seeker.save_dir,exist_ok=True)
                if mode == 'l2':
                    args.l2_loss = True
                    loss_func = LossFunc(args)
                    _, flag = seeker.seek(loss_func, record_all=record_all[mode])
                    args.l2_loss = False
                elif mode == 'rgb':
                    args.sample_loss = True
                    args.mode = 'rgb'
                    loss_func = LossFunc(args)
                    _, flag = seeker.seek(loss_func,record_all=record_all[mode])
                    args.sample_loss = False
                
                elif mode == "combine":
                    args.sample_loss = True
                    args.mode = 'combine'
                    loss_func = LossFunc(args)
                    _, flag = seeker.seek(loss_func,record_all=record_all[mode])
                    args.sample_loss = False
        record_all['sum'] = len(batch)
        with open(os.path.join('records.json'),'w') as f:
            json.dump(record_all,f)
            f.close()
        exit()


    seeker = SceneSeeker(args)
    if args.val:
        seeker.evaulate_room()
        # seeker.evaluate()
    else:
        loss_func = LossFunc(args)
        seeker.seek(loss_func)