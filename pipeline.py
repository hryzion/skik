import os
from simu import simulated_annealing
import json
import numpy as np
import cv2
import torch

# 参数设置
initial_temp = 10000    # 初始温度
cooling_rate = 0.99     # 冷却速率
min_temp = 1            # 最小温度
initial_solution = 0    # 初始解



'''
根据scene_json 输出一个初始的视点 可以是房间包围盒的中心点 或者所有物体的重心
ouput: {'pos':[x, y, z], 'direction':[dx, dy, dz]}
pos: 房间包围盒的中心点
direction: scene json的front方向
'''

def initialize_view(scene_json:dict):
    bbox_dict: dict = scene_json.get('bbox', {})
    min_list: list = bbox_dict.get('min', [0, 0, 0])
    max_list: list = bbox_dict.get('max', [0, 0, 0])
    pos_list: list = [(i+j)/2 for i, j in zip(min_list, max_list)]
    dir_list: list = scene_json.get('front', [0, 0, 1])
    return {'pos': pos_list, 'direction': dir_list}

def sketch2view(sketch, scene_json,photo2sketch_model,swint_model):
    init_view = initialize_view(scene_json)
    simulated_annealing(sketch, scene_json, init_view,photo2sketch_model,swint_model)
    
    pass


# 读入scenejson
def main():
    scene_root_path = r"D:\zhx_workspace\3DScenePlatformDev\dataset\alilevel_door2021"
    scene_name = ""
    
    img_root_path = ''
    img_name = ''
    
    scene_json_pt = os.path.join(scene_root_path,scene_name)
    sketch_pt =  os.path.join(img_root_path,img_name)
    scene_json = []
    sketch = []

    photo2sketch_model_path = './lastest_net_G.pth'
    photo2sketch_model = torch.load(photo2sketch_model_path)

    swint_model_path = ''
    swint_model = torch.load(swint_model_path)

    with open(scene_json_pt, 'r') as f:
        scene_json = json.loads(f)
    sketch = cv2.imread(sketch_pt)
    potential_views = sketch2view(sketch,scene_json,photo2sketch_model,swint_model)



if __name__ == "__main__":
    main()