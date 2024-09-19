import math
import random
import numpy as np
import cv2
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
    # 可修改参数：横向fov，图片宽度、高度
    fov = 75                                               # 相机与画布的角度
    width = 300                                            # 纵横比 4:3
    height = 400

    # 读取参数：相机位置、相机朝向
    cam_pos: list = view.get('pos', [0, 0, 0])             # 相机中心点
    cam_dir: list = view.get('direction', [0, 0, 1])       # 相机朝向
    
    # 将所有物体的包围盒存入列表
    rooms: list = scene_json.get('rooms', [])
    bboxes: list = []
    for room in rooms:
        objList: list = room.get('objList', [])
        for obj in objList:
            bboxes.append(obj.get('bbox'))

    # 将 cam_dir 归一化
    cam_dir = np.array(cam_dir)
    cam_dir = cam_dir / np.linalg.norm(cam_dir)

    # 计算相机的右向量（cross product with up vector [0, 1, 0]）
    up_vector = np.array([0, 1, 0])
    right_vector = np.cross(cam_dir, up_vector)
    right_vector = right_vector / np.linalg.norm(right_vector)

    # 重新计算相机的上向量，确保正交
    up_vector = np.cross(right_vector, cam_dir)
    up_vector = up_vector / np.linalg.norm(up_vector)

    # 计算画布的实际高度和宽度（假设画布距离相机1个单位距离）
    aspect_ratio = width / height
    canvas_width_world = 2 * np.tan(np.radians(fov) / 2)  # 使用 fov 计算画布的横向宽度
    canvas_height_world = canvas_width_world / aspect_ratio  # 通过宽高比计算画布的纵向高度

    # 计算画布中心点位置
    canvas_center = cam_pos + cam_dir

    def compute_intersection_with_canvas(ray_origin, ray_direction):
        # 计算相机到画布中心的向量
        canvas_normal = cam_dir  # 画布法向量与相机方向一致
        to_canvas = canvas_center - ray_origin
        # 计算相机光线方向与画布法向量的点积
        denom = np.dot(canvas_normal, ray_direction)
        if abs(denom) > 1e-6:  # 确保不与画布平行
            # 计算相机到画布的距离
            t = np.dot(to_canvas, canvas_normal) / denom
            if t >= 0:
                # 计算交点
                intersection = ray_origin + t * ray_direction
                return intersection
        return None

    # 创建一个空白图像，单通道（灰度图）
    image = np.zeros((height, width), dtype=np.uint8)

    for bbox in bboxes:
        # bbox : {'min':[x, y, z], 'max':[x, y, z]}
        min_bbox = bbox.get('min')
        max_bbox = bbox.get('max')
        bbox_vertices = [
            [min_bbox[0], min_bbox[1], min_bbox[2]],
            [min_bbox[0], min_bbox[1], max_bbox[2]],
            [min_bbox[0], max_bbox[1], min_bbox[2]],
            [min_bbox[0], max_bbox[1], max_bbox[2]],
            [max_bbox[0], min_bbox[1], min_bbox[2]],
            [max_bbox[0], min_bbox[1], max_bbox[2]],
            [max_bbox[0], max_bbox[1], min_bbox[2]],
            [max_bbox[0], max_bbox[1], max_bbox[2]]
        ]

        # 将包围盒顶点投影到画布上
        canvas_intersections = []
        for vertex in bbox_vertices:
            # 计算从相机源点到每个包围盒顶点的方向向量
            ray_direction = np.array(vertex) - np.array(cam_pos)
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            
            # 计算相交点
            intersection = compute_intersection_with_canvas(cam_pos, ray_direction)
            if intersection is not None:
                canvas_intersections.append(intersection)
            else:
                canvas_intersections.append(None)  # 表示没有与画布相交

            # 绘制包围盒的12条边（在画布上的12条线段）
        bbox_edges_indices = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5),
            (4, 6), (5, 7), (6, 7)
        ]

        for start_idx, end_idx in bbox_edges_indices:
            start_point = canvas_intersections[start_idx]
            end_point = canvas_intersections[end_idx]
            
            # 如果两点都与画布相交，绘制边
            if start_point is not None and end_point is not None:
                # 将三维点投影到二维画布坐标系
                def project_to_canvas(point):
                    # 将三维坐标投影到二维画布像素坐标
                    # 这里假设画布中心在画布的中间
                    relative_position = point - canvas_center
                    u = np.dot(relative_position, right_vector) / canvas_width_world * width + width / 2
                    v = -np.dot(relative_position, up_vector) / canvas_height_world * height + height / 2
                    return int(u), int(v)

                # 获取画布上的像素坐标
                p1 = project_to_canvas(start_point)
                p2 = project_to_canvas(end_point)

                # 绘制线段到图像上（单通道图像，颜色值为1）
                cv2.line(image, p1, p2, color=1, thickness=1)

    return image

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
    




