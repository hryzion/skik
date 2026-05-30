import torch

from render.NvDiffrastRenderer import NVDiffRastFullRenderer
import json
import os
from utils import *
import tqdm
from pytorch3d.io import load_obj,load_objs_as_meshes
from pytorch3d.transforms import Rotate, Translate, euler_angles_to_matrix
from pytorch3d.structures import Meshes,join_meshes_as_scene
from config import DATA_DIR, OBJ_DIR, category_to_color,coarse_categories, fine_to_coarse, coarse_to_color
import argparse

ORIGIN_OBJ_DIR = "/mnt/disk-1/zhx24/dataset/3D-FUTURE-model"
# renew a args 
parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default="3dfront")
parser.add_argument("--savedir", type=str, default="atiss")
parser.add_argument("--format", type=str, default="origin")
parser.add_argument("--width", type=int, default=256)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--camera_height", type=float, default=5.0)
parser.add_argument("--room_type", type=str, default='bedroom')
args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THREED_FRONT_FURNITURE = {'Barstool': 'stool', 'Bookcase / jewelry Armoire': 'bookshelf', 'Bunk Bed': 'bunk_bed', 'Ceiling Lamp': 'ceiling_lamp', 'Chaise Longue Sofa': 'chaise_longue_sofa', 'Children Cabinet': 'cabinet', 'Classic Chinese Chair': 'chinese_chair', 'Coffee Table': 'coffee_table', 'Corner/Side Table': 'corner_side_table', 'Desk': 'desk', 'Dining Chair': 'dining_chair', 'Dining Table': 'dining_table', 'Drawer Chest / Corner cabinet': 'cabinet', 'Dressing Chair': 'dressing_chair', 'Dressing Table': 'dressing_table', 'Footstool / Sofastool / Bed End Stool / Stool': 'stool', 'Kids Bed': 'kids_bed', 'King-size Bed': 'double_bed', 'L-shaped Sofa': 'l_shaped_sofa', 'Lazy Sofa': 'lazy_sofa', 'Lounge Chair / Cafe Chair / Office Chair': 'lounge_chair', 'Loveseat Sofa': 'loveseat_sofa', 'Nightstand': 'nightstand', 'Pendant Lamp': 'pendant_lamp', 'Round End Table': 'round_end_table', 'Shelf': 'shelf', 'Sideboard / Side Cabinet / Console table': 'console_table', 'Single bed': 'single_bed', 'TV Stand': 'multi_seat_sofa', 'Three-seat / Multi-seat Sofa': 'tv_stand', 'Wardrobe': 'wardrobe', 'Wine Cabinet': 'wine_cabinet', 'Armchair': 'armchair', 'armchair': 'armchair', 'Bed Frame' : 'single_bed',
                          'chair':'armchair', 'children_cabinet': "cabinet", 'table':"round_end_table", 'sofa':'lazy_sofa'}


THREED_FRONT_BEDROOM_FURNITURE = {
    "desk":                                    "desk",
    "nightstand":                              "nightstand",
    "king-size bed":                           "double_bed",
    "single bed":                              "single_bed",
    "kids bed":                                "kids_bed",
    "ceiling lamp":                            "ceiling_lamp",
    "pendant lamp":                            "pendant_lamp",
    "bookcase / jewelry armoire":                "bookshelf",
    "tv stand":                                "tv_stand",
    "wardrobe":                                "wardrobe",
    "lounge chair / cafe chair / office chair":    "chair",
    "dining chair":                            "chair",
    "classic chinese chair":                   "chair",
    "armchair":                                "armchair",
    "dressing table":                          "dressing_table",
    "dressing chair":                          "dressing_chair",
    "corner/side table":                       "table",
    "dining table":                            "table",
    "round end table":                         "table",
    "drawer chest / corner cabinet":             "cabinet",
    "sideboard / side cabinet / console table":    "cabinet",
    "children cabinet":                        "children_cabinet",
    "shelf":                                   "shelf",
    "footstool / sofastool / bed end stool / stool": "stool",
    "coffee table":                            "coffee_table",
    "loveseat sofa":                           "sofa",
    "three-seat / multi-seat sofa":              "sofa",
    "l-shaped sofa":                           "sofa",
    "lazy sofa":                               "sofa",
    "chaise longue sofa":                      "sofa",
    'bunk bed' :                                'single_bed',
    'barstool' :                                'stool', 
}



THREED_FRONT_LIBRARY_FURNITURE = {
    "bookcase / jewelry armoire":                "bookshelf",
    "desk":                                    "desk",
    "pendant lamp":                            "pendant_lamp",
    "ceiling lamp":                            "ceiling_lamp",
    "lounge chair /cafe chair / office chair":    "lounge_chair",
    "dining chair":                            "dining_chair",
    "dining table":                            "dining_table",
    "corner/side table":                       "corner_side_table",
    "classic chinese chair":                   "chinese_chair",
    "armchair":                                "armchair",
    "shelf":                                   "shelf",
    "sideboard / side cabinet / console table":    "console_table",
    "footstool / sofastool / bed end stool / stool": "stool",
    "barstool":                                "stool",
    "round end table":                         "round_end_table",
    "loveseat sofa":                           "loveseat_sofa",
    "drawer chest / corner cabinet":             "cabinet",
    "wardrobe":                                "wardrobe",
    "three-seat / multi-seat sofa":              "multi_seat_sofa",
    "wine cabinet":                            "wine_cabinet",
    "coffee table":                            "coffee_table",
    "lazy sofa":                               "lazy_sofa",
    "children cabinet":                        "cabinet",
    "chaise longue sofa":                      "chaise_longue_sofa",
    "l-shaped sofa":                           "l_shaped_sofa",
    "dressing table":                          "dressing_table",
    "dressing chair":                          "dressing_chair",
}

THREED_FRONT_LIVINGROOM_FURNITURE = {
    "bookcase / jewelry armoire":                "bookshelf",
    "desk":                                    "desk",
    "pendant lamp":                            "pendant_lamp",
    "ceiling lamp":                            "ceiling_lamp",
    "lounge chair / cafe chair / office chair":    "lounge_chair",
    "dining chair":                            "dining_chair",
    "dining table":                            "dining_table",
    "corner/side table":                       "corner_side_table",
    "classic chinese chair":                   "chinese_chair",
    "armchair":                                "armchair",
    "shelf":                                   "shelf",
    "sideboard / side cabinet / console table":    "console_table",
    "footstool / sofastool / bed end stool / stool": "stool",
    "barstool":                                "stool",
    "round end table":                         "round_end_table",
    "loveseat sofa":                           "loveseat_sofa",
    "drawer chest / corner cabinet":             "cabinet",
    "wardrobe":                                "wardrobe",
    "three-seat / multi-seat sofa":              "multi_seat_sofa",
    "wine cabinet":                            "wine_cabinet",
    "coffee table":                            "coffee_table",
    "lazy sofa":                               "lazy_sofa",
    "children cabinet":                        "cabinet",
    "chaise longue sofa":                      "chaise_longue_sofa",
    "l-shaped sofa":                           "l_shaped_sofa",
    "tv stand":                                "tv_stand"
}


THREED_FRONT_CATEGORY = ['dressing_table', 'console_table', 'round_end_table', 'chaise_longue_sofa', 'kids_bed', 'dressing_chair', 'tv_stand', 'bookshelf', 'lazy_sofa', 'dining_table', 'wardrobe', 'corner_side_table', 'armchair', 'chinese_chair', 'cabinet', 'nightstand', 'multi_seat_sofa', 'loveseat_sofa', 'stool', 'l_shaped_sofa', 'double_bed', 'bunk_bed', 'pendant_lamp', 'lounge_chair', 'dining_chair', 'single_bed', 'ceiling_lamp', 'wine_cabinet', 'coffee_table', 'shelf', 'desk']
THREED_FRONT_COLOR = [
    (0.90, 0.10, 0.10),  # red
    (0.10, 0.60, 0.10),  # green
    (0.10, 0.30, 0.90),  # blue
    (0.90, 0.60, 0.10),  # orange
    (0.60, 0.10, 0.80),  # purple
    (0.10, 0.70, 0.70),  # cyan
    (0.80, 0.20, 0.50),  # pink
    (0.60, 0.40, 0.10),  # brown
    (0.40, 0.40, 0.40),  # dark gray
    (0.75, 0.75, 0.10),  # olive

    (0.85, 0.25, 0.15),
    (0.15, 0.75, 0.25),
    (0.20, 0.45, 0.85),
    (0.85, 0.55, 0.15),
    (0.55, 0.15, 0.85),
    (0.15, 0.75, 0.75),
    (0.85, 0.35, 0.55),
    (0.55, 0.35, 0.15),
    (0.25, 0.25, 0.25),
    (0.70, 0.70, 0.20),

    (0.75, 0.15, 0.25),
    (0.25, 0.65, 0.25),
    (0.25, 0.35, 0.75),
    (0.75, 0.45, 0.15),
    (0.55, 0.25, 0.75),
    (0.15, 0.55, 0.65),
    (0.75, 0.25, 0.45),
    (0.55, 0.30, 0.20),
    (0.30, 0.30, 0.30),
    (0.65, 0.65, 0.15),

    (0.85, 0.30, 0.30),
    (0.30, 0.70, 0.30),
]
import random
def get_color(category ,room_type = 'bedroom'):
    if category in THREED_FRONT_CATEGORY:
        idx = THREED_FRONT_CATEGORY.index(category)
        # rgba
        return THREED_FRONT_COLOR[idx]
    else:
        print(category)
        random.seed(hash(category) % (2**32))
        return (random.random(), random.random(), random.random())

def get_category_idx(category):
    if category in THREED_FRONT_CATEGORY:
        return THREED_FRONT_CATEGORY.index(category)
    else:
        return 0  # 返回一个默认的索引，表示未知类别

def load_room_as_scene_real(room, room_type='bedroom'):
    all_meshes = []

    count = 0
    scene_vert_color = None
    scene_vert_semantic = None
    for obj in room['objList']:
        if obj['inDatabase']:
            obj_filename = os.path.join(OBJ_DIR, obj['modelId'],f'{obj["modelId"]}.obj')
            mesh = load_objs_as_meshes([obj_filename],device=args.device,load_textures=False)
            translate = torch.Tensor(obj['translate'])
            scale = torch.Tensor(obj['scale'])
            rotate = torch.Tensor([0., obj['orient'], 0.])
            class_label = obj['coarseSemantic'].lower()

            if class_label not in THREED_FRONT_CATEGORY:
                class_label = THREED_FRONT_FURNITURE[class_label]

            S = torch.diag(torch.tensor([scale[0],scale[1],scale[2],1.0])).to(args.device)

            euler_angle = torch.tensor(rotate)
            R = euler_angles_to_matrix(euler_angle,convention='XYZ')
            R = torch.cat((R, torch.zeros(3, 1)), dim=1)
            R = torch.cat((R, torch.tensor([[0, 0, 0, 1.0]])), dim=0)
            R = R.to(args.device)

            T = torch.eye(4)
            T[:3, 3] = torch.tensor(translate)
            T = T.to(device=args.device)
            transform = T @ R @ S

            temp = mesh.transform_verts(transform)
            all_meshes.append(temp)
            obj_semantic = torch.zeros((len(mesh.verts_list()[0]), len(THREED_FRONT_CATEGORY)))
            obj_semantic[..., get_category_idx(class_label)] = 1
            obj_color = torch.tensor([get_color(class_label)] * len(mesh.verts_list()[0]))
            if count == 0:
                scene_vert_color = obj_color
                scene_vert_semantic = obj_semantic
            else:
                scene_vert_color = torch.cat([scene_vert_color, obj_color], dim=0)
                scene_vert_semantic = torch.cat([scene_vert_semantic, obj_semantic],dim=0)
            count +=1 
    
    scene = join_meshes_as_scene(all_meshes,False)
    return scene, scene_vert_color, scene_vert_semantic

def load_room_as_scene_selfgen(room, room_type = 'bedroom'):
    all_meshes = []

    count = 0
    scene_vert_color = None
    scene_vert_semantic = None
    for obj in room['objList']:
        if obj['inDatabase']:
            obj_filename = os.path.join(OBJ_DIR, obj['modelId'],f'{obj["modelId"]}.obj')
            mesh = load_objs_as_meshes([obj_filename],device=args.device,load_textures=False)
            translate = torch.Tensor(obj['translate'])
            scale = torch.Tensor(obj['scale'])
            rotate = torch.Tensor([0., obj['orient'], 0.])
            class_label = obj['coarseSemantic']
            if class_label in THREED_FRONT_FURNITURE.keys():
                class_label =  THREED_FRONT_FURNITURE[class_label]

            S = torch.diag(torch.tensor([scale[0],scale[1],scale[2],1.0])).to(args.device)

            euler_angle = torch.tensor(rotate)
            R = euler_angles_to_matrix(euler_angle,convention='XYZ')
            R = torch.cat((R, torch.zeros(3, 1)), dim=1)
            R = torch.cat((R, torch.tensor([[0, 0, 0, 1.0]])), dim=0)
            R = R.to(args.device)

            T = torch.eye(4)
            T[:3, 3] = torch.tensor(translate)
            T = T.to(device=args.device)
            transform = T @ R @ S

            temp = mesh.transform_verts(transform)
            all_meshes.append(temp)
            obj_semantic = torch.zeros((len(mesh.verts_list()[0]), len(THREED_FRONT_CATEGORY)))
            obj_semantic[..., get_category_idx(class_label)] = 1
            obj_color = torch.tensor([get_color(class_label)] * len(mesh.verts_list()[0]))
            if count == 0:
                scene_vert_color = obj_color
                scene_vert_semantic = obj_semantic
            else:
                scene_vert_color = torch.cat([scene_vert_color, obj_color], dim=0)
                scene_vert_semantic = torch.cat([scene_vert_semantic, obj_semantic],dim=0)
            count +=1 
    
    scene = join_meshes_as_scene(all_meshes,False)
    return scene, scene_vert_color, scene_vert_semantic

def load_room_as_scene_origin(room, room_type = 'bedroom'):
    all_meshes = []
    count = 0
    scene_vert_color = None
    scene_vert_semantic = None
    for obj in room['objects']:
        modelId = obj['jid']
        coarseSemantic = obj['queryLabel']
        if coarseSemantic not in THREED_FRONT_CATEGORY:
            coarseSemantic =  THREED_FRONT_FURNITURE[coarseSemantic]

        obj_filename = os.path.join(ORIGIN_OBJ_DIR, modelId,'raw_model.obj')
        mesh = load_objs_as_meshes([obj_filename],device=args.device)
        translate = torch.Tensor(obj['translation'])
        
        rotate = torch.Tensor(obj['rotation'])

        S = torch.diag(torch.tensor([1,1,1,1.0])).to(args.device)

        euler_angle = torch.tensor(rotate)
        R = euler_angles_to_matrix(euler_angle,convention='XYZ')
        R = torch.cat((R, torch.zeros(3, 1)), dim=1)
        R = torch.cat((R, torch.tensor([[0, 0, 0, 1.0]])), dim=0)
        R = R.to(args.device)

        T = torch.eye(4)
        T[:3, 3] = torch.tensor(translate)
        T = T.to(device=args.device)
        transform = T @ R @ S

        temp = mesh.transform_verts(transform)
        all_meshes.append(temp)
        obj_semantic = torch.zeros((len(mesh.verts_list()[0]), len(THREED_FRONT_CATEGORY)))
        obj_semantic[..., get_category_idx(coarseSemantic)] = 1
        obj_color = torch.tensor([get_color(coarseSemantic)] * len(mesh.verts_list()[0]))
        if count == 0:
            scene_vert_color = obj_color
            scene_vert_semantic = obj_semantic
        else:
            scene_vert_color = torch.cat([scene_vert_color, obj_color], dim=0)
            scene_vert_semantic = torch.cat([scene_vert_semantic, obj_semantic],dim=0)
        count +=1 
    
    scene = join_meshes_as_scene(all_meshes)
    return scene, scene_vert_color, scene_vert_semantic


def render_scene_topdown(scene_data, savedir, save_name, renderer, H = 256, W = 256 , format = 'origin', camera_height = 5.0, room_type = 'bedroom'):
    os.makedirs(savedir, exist_ok=True)
    if format == 'origin':
        mesh, color, semantics = load_room_as_scene_origin(scene_data)
    elif format == 'real':
        mesh, color, semantics = load_room_as_scene_real(scene_data['rooms'][0], room_type)
    else:
        mesh, color, semantics = load_room_as_scene_selfgen(scene_data['rooms'][0])
    
    view = {
        'position' : [[0.,camera_height,0.]],
        'center' : [[0.01,0.,0.]]
    }
    mtce = get_camera_matrix(view, args.device, aspect=W/H,ortho=True)


    with torch.no_grad():
        img = renderer.render(view=mtce, scene_mesh=mesh, color_list=color, semantic_list=semantics)

    save_semantic = img['semantics'].cpu().squeeze(0).numpy()
    save_semantic = get_visualized_img(save_semantic, use_cv2=True)
    cv2.imwrite(os.path.join(savedir, f"{save_name}.png"), save_semantic)



renderer = NVDiffRastFullRenderer(args.device, (args.width, args.height), 1)
dir_path = args.src
savedir = args.savedir
PREFIX_DIR = "./generated"
tag = os.path.basename(os.path.normpath(dir_path))
savedir =os.path.join(PREFIX_DIR, savedir, tag)
os.makedirs(savedir, exist_ok=True)
for filename in tqdm.tqdm(os.listdir(dir_path)[:]):
    # print(filename)
    if filename.endswith('.json'):
        file_fp = os.path.join(dir_path, filename)
        room_id = 0
        subdir = os.path.basename(filename).split('.')[0]
        with open(file_fp, 'r') as f:
            scene_data = json.load(f)
            f.close()
        render_scene_topdown(scene_data, savedir, subdir, renderer, args.height, args.width, args.format, args.camera_height, args.room_type)



