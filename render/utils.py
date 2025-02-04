import json
import os
import torch
from pytorch3d.transforms import  euler_angles_to_matrix
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.io import load_objs_as_meshes


DATA_DIR = "/mnt/e/dataset/scenes"
OBJ_DIR = '/mnt/e/dataset/3DFront_p/object/'



def get_scene(scene_id,device = "cpu"):
    scene_filename = os.path.join(DATA_DIR, f"{scene_id}.json")
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
    return scene

def get_camera_matrix():
    pass