import math
import random
import numpy as np
from config import *

# 目标函数，这里以寻找某个函数的最小值为例
def objective_function(x):
    return math.sin(x) * math.cos(x) + x * x
    

def photo2sketch(photo):
    return photo


'''
根据给定scene_json和view,输出一个单通道图片,
将scenejson中的包围盒边绘制在该图片上,输出
'''
def scene2photo(scene_json, view):
    pass

def swintransformer(sketch):
    return 0

def inference(sketch, is_from_scene):
    if is_from_scene:
        # transfer 2 sketch-sketch
        sketch = photo2sketch(sketch)
    return swintransformer(sketch)
    
    

# 模拟退火算法
def simulated_annealing_pos(sketch_feature,scene_json, init_view,photo2sketch_model,swint_model):
    view = np.array(init_view)
    scene_sketch = scene2photo(scene_json,view)
    scene_feature = inference(scene_sketch,True)
    sketch_feature = np.array(sketch_feature)
    scene_feature = np.array(scene_feature)
    current_value = np.linalg.norm(sketch_feature,scene_feature)

    temp = initial_temp
    
    while temp > min_temp:
        new_view = view
        new_view['pos'] = new_view['pos'] + (random.random() - 0.5) * temp * new_view['direction']
        
        scene_sketch = scene2photo(scene_json,view)
        scene_feature = inference(scene_sketch,True)
        scene_feature = np.array(scene_feature)
        new_value = np.linalg.norm(sketch_feature,scene_feature)
        
        # 计算接受概率
        if new_value < current_value or random.random() < math.exp((current_value - new_value) / temp):
            view = new_view
        
        # 降温
        temp *= cooling_rate
    
    return view


def simulated_annealing_rot(sketch_feature,scene_json, init_view,photo2sketch_model,swint_model):
    view = np.array(init_view)
    scene_sketch = scene2photo(scene_json,view)
    scene_feature = inference(scene_sketch,True)
    sketch_feature = np.array(sketch_feature)
    scene_feature = np.array(scene_feature)
    current_value = np.linalg.norm(sketch_feature,scene_feature)

    temp = initial_temp
    
    while temp > min_temp:
        new_view = view
        new_view['pos'] = new_view['pos'] + (random.random() - 0.5) * temp * new_view['direction']
        
        scene_sketch = scene2photo(scene_json,view)
        scene_feature = inference(scene_sketch,True)
        scene_feature = np.array(scene_feature)
        new_value = np.linalg.norm(sketch_feature,scene_feature)
        
        # 计算接受概率
        if new_value < current_value or random.random() < math.exp((current_value - new_value) / temp):
            view = new_view
        
        # 降温
        temp *= cooling_rate

    
    
    
    pass

def simulated_annealing(sketch,scene_json, init_view,photo2sketch_model,swint_model):
    sketch_feature = inference(sketch,False)
    view = simulated_annealing_rot(sketch_feature,scene_json, init_view,photo2sketch_model,swint_model)
    view = simulated_annealing_pos(sketch_feature,scene_json, init_view,photo2sketch_model,swint_model)
    return view
    




