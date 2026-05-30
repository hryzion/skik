import os
import json
from cluster.cluster import *
import tqdm

DATA_DIR = "/mnt/disk-1/zhx24/dataset/3dfront"
VIEW_DIR = "/mnt/disk-1/zhx24/dataset/views/"

if not os.path.exists(VIEW_DIR):
    os.mkdir(VIEW_DIR)

for scene_name in tqdm.tqdm(os.listdir(os.path.join(DATA_DIR,'Levels2021'))[867:1000]):
    # print(scene_name)
    if not scene_name.endswith(".json"):
        continue
    cnt = 0
    with open(os.path.join(os.path.join(DATA_DIR,'Levels2021'),scene_name) ,"r") as f:
        scene_json=json.load(f)
        f.close()
    
    for room in scene_json['rooms']:
        length_of_objs = 0
        for obj in room['objList']:
            if 'coarseSemantic' not in obj or obj["coarseSemantic"] == 'Window' or obj['coarseSemantic'] == 'Door':
                continue
            length_of_objs +=1
        if length_of_objs == 0:
            continue

        labels = furnitureCluster(room)
        if labels.any() == None:
            continue
        numobj = len(labels)
        num_groups = len(set(labels))
        if numobj == num_groups:
            num_groups = 1
            labels = [0 for _ in range(numobj)]
        # print(num_groups)
        groups = [[] for _ in range(num_groups)]

        count = 0
        for obj in room['objList']:
            if 'coarseSemantic' not in obj or obj["coarseSemantic"] == 'Window' or obj['coarseSemantic'] == 'Door':
                continue
            groups[labels[count]].append(obj)
            count+=1
        
        for idx, group in enumerate(groups):
            # print("________________________")
            if len(group) <=2:
                continue
            # for obj in group:
            #     print(obj['coarseSemantic'])
            
            g_centre = findCentreGroup(group)
            g_centre[1] = 0.7

            # sample in a hemisphere
            radius_list = [2,3,4,5]
            for radius in radius_list:
                for i in range(10):
                    for j in range(60):
                        theta = j/60 * np.pi
                        phi =(i+0.5) /10*np.pi/2
                        x = radius*np.cos(theta)*np.sin(phi)
                        z = radius*np.sin(theta)*np.sin(phi)
                        y = radius*np.cos(phi)
                        origin = g_centre + np.array([x,y,z])

                        view = {
                            'gt_view':{
                                "position": [origin.tolist()],
                                "center": [g_centre.tolist()],
                                "room_id": room['roomId']
                            },
                            "mode":'manual',
                            "init_view": {
                                "position": [origin.tolist()],
                                "center": [g_centre.tolist()],
                                "room_id": room['roomId']
                            },
                            "scene": scene_json['origin'],
                            "gt_scene": scene_json['origin'],
                            "group_id" : idx
                        }

                        if not os.path.exists(f'{VIEW_DIR}/{scene_json["origin"]}/views/'):
                            os.makedirs(f'{VIEW_DIR}/{scene_json["origin"]}/')
                            os.makedirs(f'{VIEW_DIR}/{scene_json["origin"]}/views/')

                        with open(os.path.join(f'{VIEW_DIR}/{scene_json["origin"]}/views/',"scene_"+str(cnt)+".json"),"w") as f:
                            json.dump(view,f)
                            f.close()
                        cnt+=1