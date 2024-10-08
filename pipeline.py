import os
from simu import simulated_annealing,scene2photo,inference
import json
import numpy as np
import cv2
import torch
from SKIS_model import SwinTransformerEncoder
from networks import ResnetGenerator
import copy
import torchvision.transforms as transforms



room_id = 6



'''
根据scene_json 输出一个初始的视点 可以是房间包围盒的中心点 或者所有物体的重心
ouput: {'origin':[x, y, z], 'direction':[dx, dy, dz]}
origin: 房间包围盒的中心点 y设置为1.3
direction: scene json的front方向
'''

def initialize_view(scene_json:dict):
    room_json = scene_json['rooms'][0]
    bbox_dict: dict = room_json.get('bbox', {})
    min_list: list = bbox_dict.get('min', [0, 0, 0])
    max_list: list = bbox_dict.get('max', [0, 0, 0])
    # origin_list: list = [(i+j)/2 for i, j in zip(min_list, max_list)]
    origin_list: list = [
        (min_list[0] + max_list[0]) / 2,
        1.3,
        (min_list[2] + max_list[2]) / 2
    ]
    origin_list = np.array(origin_list,float)
    dir_list: list = scene_json.get('front', [0, 0, 1])
    dir_list = np.array(dir_list,float)
    
    up = np.array([0,1,0],float)
    return {'origin': list(origin_list), 'direction': list(dir_list), 'target':list(origin_list+dir_list), "up":list(up)}

def sketch2view(sketch, scene_json, photo2sketch_model,swint_model):
    init_view = initialize_view(scene_json)
    view = simulated_annealing(sketch, scene_json, init_view, photo2sketch_model, swint_model)
    return view
    


# 读入scenejson
def main():
    scene_root_path = r"D:\zhx_workspace\3DScenePlatformDev\dataset\Levels2021"
    scene_name = "000ecb5b-b877-4f9a-ab6f-90f385931658.json"
    
    img_root_path = './test'
    img_name = 'sketch1.jpg'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scene_json_pt = os.path.join(scene_root_path,scene_name)
    sketch_pt = os.path.join(img_root_path,img_name)
    scene_json = []
    sketch = []

    photo2sketch_model_path = './latest_net_G.pth'
    photo2sketch_model = ResnetGenerator(
        input_nc=3,
        output_nc=1,
        use_dropout=False,
        n_blocks=9
    )
    photo2sketch_model.load_state_dict(torch.load(photo2sketch_model_path))
    

    swint_model_path = './best.pth'
    swint_model = torch.load(swint_model_path)
    # swint_model = SwinTransformerEncoder()
    # swint_model.load_state_dict(torch.load(swint_model_path))
    

    with open(scene_json_pt, 'r') as f:
        scene_json = json.load(f)

    room_json = scene_json['rooms'][room_id]
    scene_json['rooms'] = []
    scene_json['rooms'].append(room_json)
    sketch = np.array(cv2.imread(sketch_pt))

    # 给json做个备份 防止里面的墙体被修改
    in_scene_json = copy.deepcopy(scene_json)
    sketch = cv2.resize(sketch,(224,224))
    sketch = sketch.transpose((2,0,1))
    sketch = np.expand_dims(sketch,0).astype(np.float32)
    print(sketch.shape)
    sketch = torch.from_numpy(sketch).to(device)
    
    test(scene_json)

    # 使用备份json进行计算，将计算机结果插入到原json中
    potential_view = sketch2view(sketch,in_scene_json,photo2sketch_model,swint_model)
    for key in potential_view.keys():
        potential_view[key] = potential_view[key].tolist()
    scene_json["PerspectiveCamera"] = potential_view
    scene_json["PerspectiveCamera"]['fov'] = 75
    scene_json['canvas'] = {
         "width": 600,
        "height": 337
    }
    with open('./run/test.json', 'w') as f:
        json.dump(scene_json, f)
        f.close()

def test(scene_json):
    scene_json["PerspectiveCamera"] = initialize_view(scene_json)
    scene_json["PerspectiveCamera"]['fov'] = 75
    scene_json['canvas'] = {
         "width": 600,
        "height": 337
    }
    with open('./trash/test.json', 'w') as f:
        json.dump(scene_json, f)
        f.close()


def validation():
    scene_root_path = './run/test_view2.json'
    with open(scene_root_path, 'r') as f:
        scene_json = json.load(f)
    view = scene_json['PerspectiveCamera']
    view['direction'] = np.array(view['target']) - np.array(view['origin'])
    view['direction'] /= np.linalg.norm(view['direction'])
    up = np.array(view['up'])
    direction = np.array(view['direction'])
    print(np.dot(up,direction))
    test_photo = scene2photo(copy.deepcopy(scene_json),view,debug=True)
    sketch_pt = './test/sketch1.jpg'
    sketch = np.array(cv2.imread(sketch_pt))
    sketch_pt = './test/sketch6.jpg'
    sketch2 = np.array(cv2.imread(sketch_pt))

    swint_model_path = './best.pth'
    swint_model = torch.load(swint_model_path)

    sketch = cv2.resize(sketch,(224,224))
    sketch = sketch.transpose((2,0,1))
    sketch = np.expand_dims(sketch,0).astype(np.float32)
    sketch = torch.from_numpy(sketch).to('cuda')
    sketch_feature = swint_model(sketch).flatten().detach().cpu().numpy()

    sketch2 = cv2.resize(sketch2,(224,224))
    sketch2 = sketch2.transpose((2,0,1))
    sketch2 = np.expand_dims(sketch2,0).astype(np.float32)
    sketch2 = torch.from_numpy(sketch2).to('cuda')
    sketch2_feature = swint_model(sketch2).flatten().detach().cpu().numpy()

    test_photo = cv2.resize(test_photo,(224,224))
    test_photo = cv2.cvtColor(test_photo,cv2.COLOR_GRAY2BGR)
    test_photo = transforms.ToTensor()(test_photo)
    test_photo = test_photo.unsqueeze(0)
    test_photo = test_photo.to('cuda')
    test_feature = swint_model(test_photo).flatten().detach().cpu().numpy()

    print(np.linalg.norm(sketch_feature-test_feature))
    print(np.linalg.norm(sketch_feature-sketch2_feature))

if __name__ == "__main__":
    main()
    # test()
    # validation()