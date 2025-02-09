import cv2
import torch.nn.functional as F
import glm
import torch
import numpy as np
from pytorch3d.renderer import (
    look_at_rotation_nvdiff,
    get_world_to_view_transform
)


def compute_histogram(image):
    cv2.imshow("test", image)
    cv2.waitKey()

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    print(hist)
    hist = cv2.normalize(hist, hist).flatten()  # 归一化并展平
    return hist

def cosine_distance(hist1, hist2):
    cosine_sim = F.cosine_similarity(hist1, hist2, dim=1)
    # 余弦距离 = 1 - 余弦相似度
    return 1 - cosine_sim.item()

def get_camera_matrix(view, device):
    position = torch.tensor(view["position"],device=device)
    center = torch.tensor(view['center'], device=device)
    
    R = look_at_rotation_nvdiff(position, center, device=device)
    T = -torch.bmm(R.transpose(1, 2), position[:, :, None])[:, :, 0]
    camera_mtx = get_world_to_view_transform(R=R, T=T).get_matrix().transpose(1,2).to(device)

    fov = view.get("fov", 60.0)
    znear = view.get("znear", 0.01)
    zfar = view.get("zfar", 100)
    perspective_mtx = torch.tensor(np.array(glm.perspective(
            glm.radians(fov), 1.0, znear, zfar)), device=device)
    view_mtx = torch.matmul(perspective_mtx, camera_mtx).to(device)
    return {
        "view_mtx" : view_mtx,
        "camera_mtx" : camera_mtx,
        "perspective_mtx" : perspective_mtx,
        "position":position,
        "center":center
    }


def to_direction(view):
    position = view['position']
    center = view["center"]
    direction = (center - position) / np.linalg.norm(position - center)
    return position, direction


def calculate_view_mae(view, gt):
    view = torch.tensor(to_direction(view))
    gt = torch.tensor(to_direction(gt))
    return torch.sum((view-gt)**2)

def find_room_obb(room):
    for obj in room['objList']:
        if obj['coarseSemantic'] == "Door" or obj['coarseSemantic'] == "Window":
            continue
        g_max = torch.tensor(obj["bbox"]['max'])
        g_min = torch.tensor(obj['bbox']['min'])
        break
        
    for obj in room['objList']:
        if obj['coarseSemantic'] == "Door" or obj['coarseSemantic'] == "Window":
            continue
        g_min[0] = min(obj['bbox']['min'][0],obj['bbox']['max'][0],g_min[0])
        g_min[1] = min(obj['bbox']['min'][1],obj['bbox']['max'][1],g_min[1])
        g_min[2] = min(obj['bbox']['min'][2],obj['bbox']['max'][2],g_min[2])
        g_max[0] = max(obj['bbox']['min'][0],obj['bbox']['max'][0],g_max[0])
        g_max[1] = max(obj['bbox']['min'][1],obj['bbox']['max'][1],g_max[1])
        g_max[2] = max(obj['bbox']['min'][2],obj['bbox']['max'][2],g_max[2])

        if obj['bbox']['min'][0] < g_min[0]:
            g_min[0] = obj['bbox']['min'][0]
        if obj['bbox']['min'][2] < g_min[2]:
            g_min[2] = obj['bbox']['min'][2]
        if obj['bbox']['min'][1] < g_min[1]:
            g_min[1] = obj['bbox']['min'][1]
        if obj['bbox']['max'][0] > g_max[0]:
            g_max[0] = obj['bbox']['max'][0]
        if obj['bbox']['max'][2] > g_max[2]:
            g_max[2] = obj['bbox']['max'][2]
        if obj['bbox']['max'][1] >g_max[1]:
            g_max[1] = obj['bbox']['max'][1]
    return g_max,g_min