from sklearn.cluster import AgglomerativeClustering 
import json
import os
import numpy as np
from PIL import Image
import config
from config import DATA_DIR, OBJ_DIR

import CLIP_.clip as clip
import torch
import tqdm
from pytorch3d.transforms import Rotate, Translate, euler_angles_to_matrix
from pytorch3d.structures import Meshes,join_meshes_as_scene
from pytorch3d.io import load_obj,load_objs_as_meshes


class Initializer:
    
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_dir = args.dataset
        self.scene_name = args.scene
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device,jit=False)
        with open(os.path.join(self.dataset_dir,self.scene_name+'.json'),'r') as f:
            self.scene = json.load(f)
            f.close()
        self.args = args
        self.sketch = self.preprocess(Image.open(args.sketch)).unsqueeze(0).to(args.device)
        self.eps = args.eps
        self.category_distance =np.loadtxt(rf"D:\zhx_workspace\SceneViewer\category_distance.txt")
        self.room_mesh_list = []



    def furnitureCluster(self,room):
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
                centre1 = config.centreOfObj(obj1)
                centre2 = config.centreOfObj(obj2)
                type1 = config.category_list.index(obj1['coarseSemantic'])
                type2 = config.category_list.index(obj2['coarseSemantic'])
                dis_matrix[i][j] = dis_matrix[j][i] = np.linalg.norm(centre1-centre2)+ self.category_distance[type1][type2]
        if length_of_objs == 0:
            return None
        roomShape = room['roomShape']
        bbox = config.findBBox(roomShape)
        span = max(bbox[0]-bbox[1])
        threshold = span/3+4.5
        try:
            agglomerative_label = AgglomerativeClustering(n_clusters=None,affinity='precomputed',distance_threshold=threshold,linkage='average').fit_predict(dis_matrix)     
        except:
            agglomerative_label =  [0 for _ in range(length_of_objs)]
        return agglomerative_label



    def load_room_as_scene(self,room):
        all_meshes = []
        for obj in room['objList']:
            if obj['inDatabase']:
                obj_filename = os.path.join(OBJ_DIR, obj['modelId'],f'{obj['modelId']}.obj')
                mesh = load_objs_as_meshes([obj_filename],device=self.device)
                translate = torch.Tensor(obj['translate'])
                scale = torch.Tensor(obj['scale'])
                rotate = torch.Tensor(obj['rotate'])

                S = torch.diag(torch.tensor([scale[0],scale[1],scale[2],1.0])).to(self.device)

                x_axis = torch.tensor([1,0,0]).to(self.device)
                y_axis = torch.tensor([0,1,0]).to(self.device)
                z_axis = torch.tensor([0,0,1]).to(self.device)

                euler_angle = torch.tensor(rotate)
                R = euler_angles_to_matrix(euler_angle,convention=obj['rotateOrder'])
                R = torch.cat((R, torch.zeros(3, 1)), dim=1)
                R = torch.cat((R, torch.tensor([[0, 0, 0, 1.0]])), dim=0)
                R = R.to(self.device)

                T = torch.eye(4)
                T[:3, 3] = torch.tensor(translate)
                T = T.to(device=self.device)
                transform = T @ R @ S

                temp = mesh.transform_verts(transform)
                all_meshes.append(temp)
        scene = join_meshes_as_scene(all_meshes)
        return scene        

    def initialize(self):
        semantic_tokens = []
        for room in self.scene['rooms']:
            objCount = 0
            obj_dict = {}
            roomType = room['roomTypes'][0]
            for obj in room['objList']:
                if obj['coarseSemantic'] in ['Window','Door']:
                    continue
                objCount+=1
                if obj['coarseSemantic'] in obj_dict.keys():
                    obj_dict[obj['coarseSemantic']] += 1
                else:
                    obj_dict[obj['coarseSemantic']] = 1

            if objCount == 0:
                semantic_tokens.append('empty nothing')
                continue

            room_token = f'a sketch of {roomType}'
            for key,value in obj_dict.items():
                
                room_token += f',{value} {key}'
            
            semantic_tokens.append(room_token[:77])

        if args.debug:
            for t in semantic_tokens:
                print(t)
        text = clip.tokenize(semantic_tokens).to(self.device)

        self.model.eval()
        # sketch_feature = self.model.encode_image(self.sketch)
        # text_feature = self.model.encode_text(text)

        logits_per_image, logits_per_text = self.model(self.sketch, text)

        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0]
        # print(probs)
        
        probs = list(probs)

        new_p = probs[:]
        new_p.sort(reverse=True)

        self.probe_idx = []
        for p in new_p:
            if p < self.eps:
                break
            self.probe_idx.append(probs.index(p))
        # print(self.probe_idx)
        

        # init views


        for room_id in self.probe_idx[:3]:
            room = self.scene['rooms'][room_id]
            self.room_mesh_list.append(self.load_room_as_scene(room))
            labels = self.furnitureCluster(room)

            furniture_groups = [[] for _ in range(len(labels))]
            idx =0
            for obj in room['objList']:
                if 'coarseSemantic' not in obj or obj["coarseSemantic"] == 'Window' or obj['coarseSemantic'] == 'Door':
                    continue
                furniture_groups[labels[idx]].append(obj)
                idx += 1
            
            group_tokens = []

            for g in furniture_groups:
                token = 'a sketch containing'

                for obj in g:
                    token += f' {obj['coarseSemantic']}'
                
                group_tokens.append(token)
            

            group_t = clip.tokenize(group_tokens)

            logits_per_image, logits_per_text = self.model(self.sketch, group_t)

            probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0]
            

            print(probs)                                                                                


                


                


            
        












        
        
    def test(self):
        sketch_dir = "../PhotoSketch/Exp/PhotoSketch/SketchResults"
        # sketch_dir = "../SKIS/data/sketches"
        b1sum = 0
        b1correct=0
        b3correct=0
        for i,token in tqdm.tqdm(enumerate(os.listdir(sketch_dir))):
            with open(os.path.join(self.dataset_dir,token+'.json'),'r') as f:
                self.scene = json.load(f)
                f.close()

            for img in os.listdir(os.path.join(sketch_dir,token)):
                # print(img)
                if len(img.split('-')) <2 or len(img.split('-'))>4:
                    continue
                b1sum+=1
                roomid = int(img.split('-')[0].split('room')[-1])
            
                self.sketch = self.preprocess(Image.open(os.path.join(sketch_dir,token,img))).unsqueeze(0).to(self.args.device)
                pbs = self.initialize()

                if roomid == pbs[0]:
                    b1correct+=1
                if roomid in pbs[:2]:
                    b3correct+=1
            
            if (i+1)%10==0:
                print(f'Round {i+1}-Best 1 Accuracy:{b1correct/b1sum}')
                print(f'Round {i+1}-Best 3 Accuracy:{b3correct/b1sum}')
        print(f'Best 1 Accuracy:{b1correct/b1sum}')
        print(f'Best 3 Accuracy:{b3correct/b1sum}')
        


        


                
                



if __name__ == '__main__':
    args = config.parse_arguments()
    a = Initializer(args)
    a.initialize()


