
from sklearn.cluster import AgglomerativeClustering 
import numpy as np

category_distance =np.loadtxt(rf"./cluster/category_distance.txt")
category_list = ['Barstool', 'Bookcase / jewelry Armoire', 'Bunk Bed', 'Ceiling Lamp', 'Chaise Longue Sofa', 'Children Cabinet', 'Classic Chinese Chair', 'Coffee Table', 'Corner/Side Table', 'Desk', 'Dining Chair', 'Dining Table', 'Drawer Chest / Corner cabinet', 'Dressing Chair', 'Dressing Table', 'Footstool / Sofastool / Bed End Stool / Stool', 'Kids Bed', 'King-size Bed', 'L-shaped Sofa', 'Lazy Sofa', 'Lounge Chair / Cafe Chair / Office Chair', 'Loveseat Sofa', 'Nightstand', 'Pendant Lamp', 'Round End Table', 'Shelf', 'Sideboard / Side Cabinet / Console table', 'Single bed', 'TV Stand', 'Three-seat / Multi-seat Sofa', 'Wardrobe', 
'Wine Cabinet', 'armchair', 'door', 'window']


def centreOfObj(obj):
    """
        return the centre of obj's bounding box    
    """
    v1_max = np.array(obj['bbox']['max'])
    v1_min = np.array(obj['bbox']['min'])
    return (v1_max+v1_min)/2

def volumeOfObj(obj):
    """
        return the volume of obj's bounding box
    """
    v1_max = np.array(obj['bbox']['max'])
    v1_min = np.array(obj['bbox']['min'])
    return abs(np.prod(v1_max-v1_min))

def findBBox(roomShape):
    right_up = [-1000,-1000]
    left_down = [100000,100000]
    for wall in roomShape:
        if wall[0]>right_up[0]:
            right_up[0] = wall[0]
        if wall[0] < left_down[0]:
            left_down[0] = wall[0]
        if wall[1] > right_up[1]:
            right_up[1] = wall[1]
        if wall[1] < left_down[1]:
            left_down[1] = wall[1]
    return np.array([right_up,left_down])

def findBBoxGroup(group):
    assert len(group) > 0
    max_b = group[0]['bbox']['max']
    min_b = group[0]['bbox']['min']
    for obj in group:
        max_b = np.maximum(max_b,obj['bbox']['max'])
        min_b = np.minimum(min_b,obj['bbox']['min'])

    return np.array([max_b,min_b])

def findCentreGroup(group):
    assert len(group) > 0
    centre = np.array([0.,0.,0.])
    sum_weight = 0
    for obj in group:
        # print(centreOfObj(obj) , volumeOfObj(obj))
        centre += centreOfObj(obj)*volumeOfObj(obj)
        sum_weight += volumeOfObj(obj)
    return centre/sum_weight



def furnitureCluster(room):
    objects = []
    objects_without_dw =[]
    length_of_objs =0
    for obj1 in room['objList']:
        if 'coarseSemantic' not in obj1 or obj1["coarseSemantic"] == 'Window' or obj1['coarseSemantic'] == 'Door':
            continue
        length_of_objs+=1
        objects_without_dw.append(obj1)
    dis_matrix = [[0 for _ in range(length_of_objs)] for __ in range(length_of_objs)]
    for i in range(length_of_objs):
        obj1 = objects_without_dw[i]
        for j in range(i+1,length_of_objs):
            obj2 = objects_without_dw[j]
            centre1 = centreOfObj(obj1)
            centre2 = centreOfObj(obj2)
            type1 = category_list.index(obj1['coarseSemantic'])
            type2 = category_list.index(obj2['coarseSemantic'])
            dis_matrix[i][j] = dis_matrix[j][i] = np.linalg.norm(centre1-centre2)+ category_distance[type1][type2]
    if length_of_objs == 0:
        return None
    roomShape = room['roomShape']
    bbox = findBBox(roomShape)
    span = max(bbox[0]-bbox[1])
    threshold = span/3+4.5
    try:
        agglomerative_label = AgglomerativeClustering(n_clusters=None,metric='precomputed',distance_threshold=threshold,linkage='average').fit_predict(dis_matrix)     
    except:
        print("WARNING: AgglomerativeClustering failed return as a single cluster")
        return np.zeros(length_of_objs)
    return agglomerative_label