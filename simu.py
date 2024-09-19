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
view: {'pos':[x, y, z], 'direction':[dx, dy, dz]}
'''
def scene2photo(scene_json: dict, view: dict):
    pass

def swintransformer(sketch):
    return 0

def inference(sketch, is_from_scene):
    if is_from_scene:
        # transfer 2 sketch-sketch
        sketch = photo2sketch(sketch)
    return swintransformer(sketch)



def rotate_vector_around_x_axis(v, theta):
    """Rotate a 3D vector around the X-axis by theta degrees."""
    theta_rad = np.radians(theta)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta_rad), -np.sin(theta_rad)],
        [0, np.sin(theta_rad), np.cos(theta_rad)]
    ])
    return np.dot(rotation_matrix, v)

def rotate_vector_around_y_axis(v, theta):
    """Rotate a 3D vector around the Y-axis by theta degrees."""
    theta_rad = np.radians(theta)
    rotation_matrix = np.array([
        [np.cos(theta_rad), 0, np.sin(theta_rad)],
        [0, 1, 0],
        [-np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])
    return np.dot(rotation_matrix, v)

def rotate_vector_around_z_axis(v, theta):
    """Rotate a 3D vector around the Z-axis by theta degrees."""
    theta_rad = np.radians(theta)
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad), np.cos(theta_rad), 0],
        [0, 0, 1]
    ])
    return np.dot(rotation_matrix, v)
    
    

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
        rot_x=random.randint(0, 359)
        rot_y=random.randint(0, 359)
        rot_z=random.randint(0, 359)
        new_view['direction']= rotate_vector_around_x_axis(new_view['direction'],rot_x)
        new_view['direction']= rotate_vector_around_y_axis(new_view['direction'],rot_y)
        new_view['direction']= rotate_vector_around_z_axis(new_view['direction'],rot_z)
        

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
    




