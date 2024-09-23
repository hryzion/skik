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
ouput: {'origin':[x, y, z], 'direction':[dx, dy, dz]}
origin: 房间包围盒的中心点 y设置为1.3
direction: scene json的front方向
'''

def initialize_view(scene_json:dict):
    bbox_dict: dict = scene_json.get('bbox', {})
    min_list: list = bbox_dict.get('min', [0, 0, 0])
    max_list: list = bbox_dict.get('max', [0, 0, 0])
    # origin_list: list = [(i+j)/2 for i, j in zip(min_list, max_list)]
    origin_list: list = [
        (min_list[0] + max_list[0]) / 2,
        1.3,
        (min_list[2] + max_list[2]) / 2
    ]
    origin_list = np.array(origin_list)
    dir_list: list = scene_json.get('front', [0, 0, 1])
    dir_list = np.array(dir_list)
    up = np.array([0,1,0])
    return {'origin': origin_list, 'direction': dir_list, 'target':origin_list+dir_list, "up":up}

def sketch2view(sketch, scene_json,photo2sketch_model,swint_model):
    init_view = initialize_view(scene_json)
    simulated_annealing(sketch, scene_json, init_view,photo2sketch_model,swint_model)
    
    pass


# 读入scenejson
def main():
    scene_root_path = r"D:\zhx_workspace\3DScenePlatformDev\dataset\Levels2021"
    scene_name = ""
    
    img_root_path = './test'
    img_name = 'sketch1.jpg'
    
    scene_json_pt = os.path.join(scene_root_path,scene_name)
    sketch_pt =  os.path.join(img_root_path,img_name)
    scene_json = []
    sketch = []

    photo2sketch_model_path = './lastest_net_G.pth'
    photo2sketch_model = torch.load(photo2sketch_model_path)

    swint_model_path = './best.pth'
    swint_model = torch.load(swint_model_path)

    with open(scene_json_pt, 'r') as f:
        scene_json = json.loads(f)
    sketch = cv2.imread(sketch_pt)
    potential_view = sketch2view(sketch,scene_json,photo2sketch_model,swint_model)
    scene_json["PerspectiveCamera"] = potential_view
    with open('./run/test.json', 'w') as f:
        json.dump(scene_json, f)
        f.close()



if __name__ == "__main__":
    main()