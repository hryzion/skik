import math
import random
import numpy as np
import cv2
from config import *
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


# 目标函数，这里以寻找某个函数的最小值为例
def objective_function(x):
    return math.sin(x) * math.cos(x) + x * x
    

def photo2sketch(photo,photo2sketch_model):
    # print(photo.shape)
    
    photo = Image.fromarray(photo).convert('RGB')
    photo = photo.resize((224, 224), Image.BICUBIC)
    # photo.show()

    
    photo = transforms.ToTensor()(photo)
    photo = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(photo)
    photo = photo.unsqueeze(0)
    
    
    
    return photo2sketch_model(photo)


'''
根据给定scene_json和view,输出一个单通道图片,
将scenejson中的包围盒边绘制在该图片上,输出
view: {'pos':[x, y, z], 'direction':[dx, dy, dz]}
注意: roomid: scene_json中的id, 为一个字符串,如果字符串为空,则默认为第一个. 如果不符合需要请调整
裁剪后, 图片shape为 (1, 1, 265, 355), 奇怪的尺寸是裁剪的原因

如需保存为图片：
    import matplotlib.pyplot as plt
    plt.imshow(image[0,0], cmap='gray')
    plt.savefig('1.png')
'''
def scene2photo(scene_json: dict, view: dict, roomid: str = ''):
    # 可修改参数：横向fov，图片宽度、高度
    fov_large = 120                                   # 放一个更大的角度做初始绘制，再裁剪为目标角度，能够一定程度上优化结果
    fov_target = 120                                   # 相机与画布的目标角度
    canvas_width = 800                                # 横纵比 4:3
    canvas_height = 600
    # print(view)

    

    # 读取参数：相机位置、相机朝向
    cam_pos: list = view.get('origin', [0, 0, 0])             # 相机中心点
    cam_dir: list = view.get('direction', [0, 0, 1])       # 相机朝向


    
    # 将目标房间的包围盒存入列表
    rooms: list = scene_json.get('rooms', [])
    bboxes: list = []
    roomShape: list = []
    roomNorm: list = []
    if roomid == '':
        room = rooms[0]
        objList: list = room.get('objList', [])
        roomShape: list = room.get('roomShape')
        roomNorm: list = room.get('roomNorm')
        for obj in objList:
            bboxes.append(obj.get('bbox'))
    else:
        for room in rooms:
            if room.get('id', '*') == roomid:
                objList: list = room.get('objList', [])
                for obj in objList:
                    bboxes.append(obj.get('bbox'))
                roomShape: list = room.get('roomShape')
                roomNorm: list = room.get('roomNorm')
                break


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

    # 计算较大fov对应的画布的实际宽度和高度
    aspect_ratio = canvas_width / canvas_height
    canvas_width_world_large = 2 * np.tan(np.radians(fov_large) / 2)  # 使用较大的 fov 计算画布的横向宽度
    canvas_height_world_large = canvas_width_world_large / aspect_ratio  # 通过宽高比计算纵向高度

    # 计算画布的实际高度和宽度（假设画布距离相机1个单位距离）
    canvas_width_world_target = 2 * np.tan(np.radians(fov_target) / 2)  # 使用 fov 计算画布的横向宽度
    canvas_height_world_target = canvas_width_world_target / aspect_ratio  # 通过宽高比计算画布的纵向高度

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
    image_large = np.ones((canvas_height, canvas_width), dtype=np.uint8)
    image_large *= 255
    

    # 绘制包围盒
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
                def project_to_canvas(point, canvas_width_world, canvas_height_world):
                    # 将三维坐标投影到二维画布像素坐标
                    relative_position = point - canvas_center
                    u = np.dot(relative_position, right_vector) / canvas_width_world * canvas_width + canvas_width / 2
                    v = -np.dot(relative_position, up_vector) / canvas_height_world * canvas_height + canvas_height / 2
                    return int(u), int(v)

                # 获取画布上的像素坐标
                p1 = project_to_canvas(start_point, canvas_width_world_large, canvas_height_world_large)
                p2 = project_to_canvas(end_point, canvas_width_world_large, canvas_height_world_large)

                # 绘制线段到图像上（单通道图像，颜色值为1）
                
                cv2.line(image_large, p1, p2, color=0, thickness=5)


    def divide_line_into_segments(roomShape, num_segments=30):
        points = []
        
        for i in range(len(roomShape) - 1):  # 不连接最后一个和第一个
            start = roomShape[i]
            end = roomShape[i + 1]
            
            x_diff = (end[0] - start[0]) / num_segments
            y_diff = (end[1] - start[1]) / num_segments
            
            for j in range(num_segments+1):
                new_point = [start[0] + j * x_diff, start[1] + j * y_diff]
                points.append(new_point)
        
        return points


    # 处理墙体，房间形状的输入
    wall_height = 2.6  # 墙的高度固定为2.6

    # 遍历分割前墙体的x-z坐标对,绘制其垂直边
    step_size = (2.6) / (30)
    roomShape_vertical_divide = [i * step_size for i in range(31)]
    for wall_point in roomShape:
        for y_value in roomShape_vertical_divide:

            start = [wall_point[0], 0, wall_point[1]]
            end = [wall_point[0], y_value, wall_point[1]]
            wall_vertices = [
                [wall_point[0], 0, wall_point[1]],
                [wall_point[0], y_value, wall_point[1]]
            ]
            wall_intersections = []
            for vertex in wall_vertices:
                ray_direction = np.array(vertex) - np.array(cam_pos)
                ray_direction = ray_direction / np.linalg.norm(ray_direction)
                intersection = compute_intersection_with_canvas(cam_pos, ray_direction)
                if intersection is not None:
                    wall_intersections.append(intersection)
                else:
                    wall_intersections.append(None)

            start_point = wall_intersections[0]
            end_point = wall_intersections[1]
            if start_point is not None and end_point is not None:
                p1 = project_to_canvas(start_point, canvas_width_world_large, canvas_height_world_large)
                p2 = project_to_canvas(end_point, canvas_width_world_large, canvas_height_world_large)
                # print(p1,p2)
                cv2.line(image_large, p1, p2, color=0, thickness=5)

    

    roomShape.append(roomShape[0])  # 将墙体首尾相连
    roomShape_horizonal_divide = divide_line_into_segments(roomShape=roomShape)
    
    # 遍历分割后墙体的x-z坐标对，绘制其水平边
    for i in range(len(roomShape_horizonal_divide) - 1):
        start = roomShape_horizonal_divide[i]
        end = roomShape_horizonal_divide[i + 1]
        
        # 墙的四个顶点
        wall_vertices = [
            [start[0], 0, start[1]],  # 底部左
            [start[0], wall_height, start[1]],  # 顶部左
            [end[0], 0, end[1]],  # 底部右
            [end[0], wall_height, end[1]]  # 顶部右
        ]

        # 投影每个墙顶点到画布
        wall_intersections = []
        for vertex in wall_vertices:
            ray_direction = np.array(vertex) - np.array(cam_pos)
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            intersection = compute_intersection_with_canvas(cam_pos, ray_direction)
            if intersection is not None:
                wall_intersections.append(intersection)
            else:
                wall_intersections.append(None)
        
        # 绘制墙体的两条水平边
        for j in range(2):
            start_point = wall_intersections[j]
            end_point = wall_intersections[j + 2]
            if start_point is not None and end_point is not None:
                p1 = project_to_canvas(start_point, canvas_width_world_large, canvas_height_world_large)
                p2 = project_to_canvas(end_point, canvas_width_world_large, canvas_height_world_large)
                
                cv2.line(image_large, p1, p2, color=0, thickness=5)

    # 计算裁剪范围，以fov=75度为目标
    crop_x_min = int((canvas_width / 2) - (canvas_width * (canvas_width_world_target / canvas_width_world_large) / 2))
    crop_x_max = int((canvas_width / 2) + (canvas_width * (canvas_width_world_target / canvas_width_world_large) / 2))
    crop_y_min = int((canvas_height / 2) - (canvas_height * (canvas_height_world_target / canvas_height_world_large) / 2))
    crop_y_max = int((canvas_height / 2) + (canvas_height * (canvas_height_world_target / canvas_height_world_large) / 2))

    # 裁剪图像，得到目标fov下的画布
    image_cropped = image_large[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    # image_cropped = np.expand_dims(image_cropped, axis=-1)
    # plt.imshow(image_large,cmap='gray')
    # cv2.imshow('test', image_large)
    # cv2.waitKey(0)
    return image_cropped

def swintransformer(sketch,swint_model):
    return swint_model.forward_features(sketch)

def inference(sketch, is_from_scene, photo2sketch_model, swint_model):
    
    if is_from_scene:
        # transfer 2 sketch-sketch
        # sketch = photo2sketch(sketch,photo2sketch_model)
        
        # print(sketch)
        # sketch = sketch.detach()[0][0].numpy()
        # print(sketch)
        # exit()
        sketch = cv2.resize(sketch,(224,224))
        sketch = cv2.cvtColor(sketch,cv2.COLOR_GRAY2BGR)
        sketch = transforms.ToTensor()(sketch)
        sketch= sketch.unsqueeze(0)
    sketch = sketch.to('cuda')
    feat = swintransformer(sketch,swint_model).flatten()
    feat = feat.detach().cpu().numpy()
    
    # print(feat.shape)
    return feat



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

def rotate_vector_3d(v, theta_x, theta_y, theta_z):
    # 转换角度为弧度
    theta_x = np.radians(theta_x)
    theta_y = np.radians(theta_y)
    theta_z = np.radians(theta_z)
    
    # 定义旋转矩阵
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转
    v_rotated = np.dot(R_z, np.dot(R_y, np.dot(R_x, v)))
    return v_rotated

def cartesian_to_spherical(v):
    """
    将笛卡尔坐标转换为方位角和俯仰角
    :param x: x坐标
    :param y: y坐标
    :param z: z坐标
    :return: 方位角φ（弧度），俯仰角θ（弧度）
    """
    # 计算方位角，结果在 [-π, π]
    x,y,z = v[0], v[1], v[2]
    phi = np.arctan2(y, x)
    
    # 计算俯仰角，结果在 [0, π]
    theta = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))
    
    return np.degrees(phi), np.degrees(theta)

def direction_vector_from_angles(phi, theta):
    """
    根据给定的方位角和俯仰角计算新的方向向量
    :param phi: 方位角，单位为弧度
    :param theta: 俯仰角，单位为弧度
    :return: 新的方向向量
    """
    # 计算新的向量的x, y, z分量
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # 返回新的方向向量
    return np.array([x, y, z])
    
    

# 模拟退火算法
def simulated_annealing_rot(sketch_feature,scene_json, init_view,photo2sketch_model,swint_model):
    view = init_view
    scene_sketch = scene2photo(scene_json,view)
    scene_feature = inference(scene_sketch,True,photo2sketch_model,swint_model)
    
    current_value = np.linalg.norm(sketch_feature-scene_feature)

    temp = initial_temp
    
    while temp > min_temp:
        new_view = view
        dtheta_x = random.randint(0,359) * temp*(random.random()-0.5)
        dtheta_y = random.randint(0,359) * temp*(random.random()-0.5)
        dtheta_z = random.randint(0,359) * temp*(random.random()-0.5)

        new_view['direction'] = rotate_vector_3d(new_view['direction'],dtheta_x,dtheta_y,dtheta_z)
        new_view['up'] = rotate_vector_3d(new_view['up'],dtheta_x,dtheta_y,dtheta_z)
        
        # print(new_view['up'], new_view['direction'])
        # print(np.dot(new_view['direction'],new_view['up']))
        assert(np.dot(new_view['direction'],new_view['up']) < 1e-6)

        new_view['target'] = new_view['direction'] + new_view['origin']

        scene_sketch = scene2photo(scene_json,view)
        scene_feature = inference(scene_sketch,True,photo2sketch_model,swint_model)
        
        new_value = np.linalg.norm(sketch_feature-scene_feature)
        
        # 计算接受概率
        if new_value < current_value or random.random() < math.exp((current_value - new_value) / temp):
            view = new_view
            current_value = new_value
        
        # 降温
        temp *= cooling_rate
    
    return view


def simulated_annealing_pos(sketch_feature,scene_json, init_view,photo2sketch_model,swint_model):
    view = init_view
    scene_sketch = scene2photo(scene_json,view)
    # print('scene_sketch', scene_sketch.shape)
    scene_feature = inference(scene_sketch,True,photo2sketch_model,swint_model)
    
    current_value = np.linalg.norm(sketch_feature-scene_feature)

    temp = initial_temp
    
    while temp > min_temp:
        new_view = view
        # print(new_view)
        new_view['origin'] = np.array(new_view['origin']) + (random.random() - 0.5) * temp * np.array(new_view['direction'])
        
        scene_sketch = scene2photo(scene_json,view)
        scene_feature = inference(scene_sketch,True,photo2sketch_model,swint_model)
        
        new_value = np.linalg.norm(sketch_feature-scene_feature)
        
        # 计算接受概率
        if new_value < current_value or random.random() < math.exp((current_value - new_value) / temp):
            view = new_view
        
        # 降温
        temp *= cooling_rate

    
    
    
    pass

def simulated_annealing(sketch,scene_json, init_view,photo2sketch_model,swint_model):
    sketch_feature = inference(sketch,False,photo2sketch_model,swint_model)
    view = simulated_annealing_rot(sketch_feature,scene_json, init_view,photo2sketch_model,swint_model)
    print('finish annealing_rot')
    view = simulated_annealing_pos(sketch_feature,scene_json, init_view,photo2sketch_model,swint_model)
    print('finish annealing_pos')
    return view
    




