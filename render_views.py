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
from pytorch3d.io import load_obj,load_objs_as_meshes
from pytorch3d.transforms import Rotate, Translate, euler_angles_to_matrix
from pytorch3d.structures import Meshes,join_meshes_as_scene
from config import DATA_DIR, OBJ_DIR, category_to_color,coarse_categories, fine_to_coarse, coarse_to_color

args = config.parse_arguments()

def load_room_as_scene(room):
    all_meshes = []

    count = 0
    scene_vert_color = None
    scene_vert_semantic = None
    for obj in room['objList']:
        if obj['inDatabase']:
            obj_filename = os.path.join(OBJ_DIR, obj['modelId'],f'{obj['modelId']}.obj')
            mesh = load_objs_as_meshes([obj_filename],device=args.device)
            translate = torch.Tensor(obj['translate'])
            scale = torch.Tensor(obj['scale'])
            rotate = torch.Tensor(obj['rotate'])

            S = torch.diag(torch.tensor([scale[0],scale[1],scale[2],1.0])).to(args.device)

            euler_angle = torch.tensor(rotate)
            R = euler_angles_to_matrix(euler_angle,convention=obj['rotateOrder'])
            R = torch.cat((R, torch.zeros(3, 1)), dim=1)
            R = torch.cat((R, torch.tensor([[0, 0, 0, 1.0]])), dim=0)
            R = R.to(args.device)

            T = torch.eye(4)
            T[:3, 3] = torch.tensor(translate)
            T = T.to(device=args.device)
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

dataset  = args.dataset
cnt = 0
for scene_name in os.listdir("/mnt/e/dataset/tmp/views/data"):
    room_mesh_list = []
    color_list = []
    semantic_list = []
    with open(f'{dataset}/{scene_name}.json','r') as f:
        scene = json.load(f)
        f.close()   
    for room_id,room in enumerate(scene['rooms']):
        mesh, vert_color, vert_semantic = load_room_as_scene(room)
        room_mesh_list.append(mesh)
        color_list.append(vert_color)
        semantic_list.append(vert_semantic)

    
    

    print(f"Processing {scene_name}...")
    for settings in tqdm.tqdm(os.listdir(f"/mnt/e/dataset/tmp/views/data/{scene_name}")):
        if settings.split('.')[-1] != 'json':
            continue
        with open(f"/mnt/e/dataset/tmp/views/data/{scene_name}/{settings}", 'r') as f:
            setting = json.load(f)
            f.close()
        
        renderer = NVDiffRastFullRenderer(args.device, setting, (args.width, args.height), 1)
        view = setting['gt_view']
        room = scene["rooms"][view['room_id']]
        
        mesh, color, semantics = room_mesh_list[view['room_id']], color_list[view['room_id']], semantic_list[view['room_id']]
        mtce = get_camera_matrix(view, args.device, aspect=args.width/args.height)
        with torch.no_grad():
            img = renderer.render(view=mtce, scene_mesh=mesh, color_list=color, semantic_list=semantics)
        save_img = img["images"].cpu().squeeze(0).numpy()
        save_img = get_visualized_img(save_img, use_cv2=True)
        cv2.imwrite(f"/mnt/e/dataset/tmp/views/train/img/{cnt}.png", save_img)
        save_semantic = img['semantics'].cpu().squeeze(0).numpy()
        save_semantic = get_visualized_img(save_semantic, use_cv2=True)

        save_semantic_channels = img['semantics_c'].cpu().squeeze(0).numpy()
        # flip semantic channels 
        save_semantic_channels = np.flip(save_semantic_channels, axis=0)
        np.save(f'/mnt/e/dataset/tmp/views/train/gt/{cnt}_semantic.npy', save_semantic_channels)
        cv2.imwrite(f'/mnt/e/dataset/tmp/views/train/img/{cnt}_semantic.png', save_semantic)
        cnt+=1

    