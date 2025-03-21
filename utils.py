import cv2
import torch.nn.functional as F
import glm
import torch
import numpy as np
from pytorch3d.renderer import (
    look_at_rotation_nvdiff,
    get_world_to_view_transform,
    look_at_rotation
)

def get_uint_mask(msk):
        tmp = msk.squeeze(0).cpu()
        mask = np.zeros(msk.squeeze(0).cpu().shape, np.uint8)
        mask[tmp == True] = 1
        return mask

def get_visualized_img(img, flip = True, use_cv2 = False):
        img = img.astype(np.float32)
        img = img[...,:3]
        if img.shape[-1]==3 and use_cv2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if flip:
            img = cv2.flip(img, 0)
        
        img = (np.clip(img,0,1)*255).astype(np.uint8)
        return img

def bhattacharyya_distance(H1, H2):
    return -torch.log(torch.sum(np.sqrt(H1 * H2),dim=1))

def compute_histogram(image ,msk = None):
    hist = cv2.calcHist([image], [0, 1, 2],msk, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # 归一化并展平
    return hist

def cosine_distance(hist1, hist2):
    cosine_sim = F.cosine_similarity(hist1, hist2, dim=1)
    # 余弦距离 = 1 - 余弦相似度
    return 1 - cosine_sim.item()

def get_camera_matrix(view, device, aspect = 1):
    position = torch.tensor(view["position"], device=device, requires_grad=True)
    center = torch.tensor(view['center'], device=device, requires_grad=True)
    
    R = look_at_rotation_nvdiff(position, center, device=device)
    T = -torch.bmm(R.transpose(1, 2), position[:, :, None])[:, :, 0]

    camera_mtx = get_world_to_view_transform(R=R, T=T).get_matrix().transpose(1,2).to(device)

    fov = view.get("fov", 60.0)
    znear = view.get("znear", 0.01)
    zfar = view.get("zfar", 100)
    perspective_mtx = torch.tensor(np.array(glm.perspective(
            glm.radians(fov), aspect, znear, zfar)), device=device)
    view_mtx = torch.matmul(perspective_mtx, camera_mtx).to(device)
    return {
        "view_mtx" : view_mtx,
        "camera_mtx" : camera_mtx,
        "perspective_mtx" : perspective_mtx,
        "position":position,
        "center":center
    }


def to_direction(view):
    position = torch.tensor(view['position'])
    center = torch.tensor(view["center"])
    direction = (center - position) / torch.norm(position - center)
    print(position,direction)
    return {
        "position":position.squeeze(0).cpu(),
        "direction":direction.squeeze(0).cpu()
    }


def calculate_view_mae(view, gt):
    view = to_direction(view)
    gt = to_direction(gt)
    d_p = torch.sqrt(torch.sum((view['position'] - gt['position'])**2))
    # calculate the angle between two directions
    
    d_d = torch.acos(torch.dot(view["direction"], gt["direction"]) / (torch.norm(view["direction"]) * torch.norm(gt["direction"]))) * 180 / np.pi
    return d_p , d_d

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



def show_room(room,scale):
    img = np.zeros((scale,scale,3),np.uint8)
    img[:] = (255,255,255)
    centre = (scale//2, scale//2)
    
    
 
    K = 100

    roomShape = room['roomShape']
    bbox = np.array(find_room_obb(room))
    old_centre = (bbox[0]+bbox[1])/2
    old_centre = np.array([old_centre[0],old_centre[2]])

    # draw walls of the room
    for wall_index in range(len(roomShape)):
        wall_next = (wall_index+1) % len(roomShape)
        p1 = (np.array(roomShape[wall_index])-old_centre)*K+centre
        p1[0] = int(p1[0])
        p1[1] = int(p1[1])
        p2 = (np.array(roomShape[wall_next])-old_centre)*K+centre
        p2[0] = int(p2[0])
        p2[1] = int(p2[1])
        p1 = np.array(p1,np.int32)
        p2 = np.array(p2,np.int32)
        cv2.line(img,p1,p2,(255,255,255),8)
        
    # draw objects 
    colors = [(0,0,255),(255,0,0), (0,255,0),(255,255,0),(0,255,255), (255,0,255),(0,200,200),(128,128,128),(156,12,245)]
    idx = 0
    for obj in room['objList']:
        if 'coarseSemantic' not in obj or obj['coarseSemantic'] == 'Door' or obj['coarseSemantic'] == 'Window':
            continue
        p_max = (np.array((obj['bbox']['max'][0],obj['bbox']['max'][2]))-old_centre)*K+centre
        p_min = (np.array((obj['bbox']['min'][0],obj['bbox']['min'][2]))-old_centre)*K+centre
        p_max = np.array(p_max,np.int32)
        p_min = np.array(p_min,np.int32)
        


        # cv2.rectangle(img,p_min,p_max,colors[groups[idx]],8)
        cv2.rectangle(img,p_min,p_max,colors[0],8)
        idx+=1
    
    
        
    cv2.imshow('tst',img)
    #cv2.imwrite(f'C:\\Users\\evan\\Desktop\\zhx_workspace\\SceneViewer\\cluster_result\\{scene_json["origin"]}.jpg',img)

    cv2.waitKey(0)

