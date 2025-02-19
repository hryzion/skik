from sklearn.cluster import AgglomerativeClustering 
import json
import os
import numpy as np
from PIL import Image
import config
from config import DATA_DIR, OBJ_DIR, category_to_color

import CLIP_.clip as clip
import torch
import tqdm
from pytorch3d.transforms import Rotate, Translate, euler_angles_to_matrix
from pytorch3d.structures import Meshes,join_meshes_as_scene
from pytorch3d.io import load_obj,load_objs_as_meshes
import torchvision.transforms as transforms

from utils import *
import matplotlib.pyplot as plt


class Initializer:
    def __init__(self, args, settings):
        self.device = args.device
        self.dataset_dir = args.dataset
        self.scene_name = settings['scene']
        
        self.model, clip_preprocess = clip.load("ViT-B/32", device=self.device,jit=False)
        self.preprocess =transforms.Compose([
            transforms.Resize((224,224)),
            clip_preprocess.transforms[-1] # normalize
        ])
        with open(os.path.join(self.dataset_dir,self.scene_name+'.json'),'r') as f:
            self.scene = json.load(f)
            f.close()
        
        self.args = args
        self.eps = args.eps
        self.room_mesh_list = []
        self.color_list = []
        self.semantic_list = []
        self.initialize_room()


    def set_gt_img(self,img):
        self.gt_img = img['images'].to(self.device)
        img_uint = self.gt_img.clone().cpu().squeeze(0).numpy()
        img_uint = get_visualized_img(img_uint,use_cv2=True)
        tmp = img['msk'].squeeze(0).cpu()
        mask = np.zeros(img['msk'].squeeze(0).cpu().shape, np.uint8)
        mask[tmp == True] = 1
        self.hist = compute_histogram(img_uint,mask)
        if self.args.debug:
            plt.plot(self.hist)
            plt.show()
    

    
    def get_texts_and_colorhisto(self,renderer, room, room_id, mode="top"):
        # top, 4d, 8d
        # show_room(room,512)
        
        # top-down views
        if mode == "top":
            bbox = find_room_obb(room)
            center = ((torch.tensor(bbox[0]) + torch.tensor(bbox[1]))/2).unsqueeze(0)
            tmp = torch.abs(torch.tensor(bbox[0]-bbox[1]))
            scale = torch.max(tmp[0],tmp[2])
            position = center.clone()
            position[...,1] += scale
            position[...,2] += 0.1
            view ={
                'position': position,
                'center': center,
                "room_id" : room_id
            }
            mtce = get_camera_matrix(view=view, device=self.device)
            scene = self.room_mesh_list[room_id]
            color = self.color_list[room_id]
            with torch.no_grad():
                render_res = renderer.render(mtce,scene.clone(), color.clone(),self.semantic_list[room_id])
                img = render_res['images']
                
                msk = get_uint_mask(render_res["msk"])
            
            if img.shape[-1] == 4:
                img = img[..., 0:3]
            img = img.cpu().squeeze(0).numpy()
            img = get_visualized_img(img,use_cv2=True)

            # get histo
            hist = compute_histogram(img,msk)
            if self.args.debug:
                pass
                # plt.plot(hist)
                # plt.show()

            # get texts
            roomType = room['roomTypes'][0]
            objCount = 0
            obj_dict = {}
            for obj in room['objList']:
                if obj['coarseSemantic'] in ['Window','Door']:
                    continue
                objCount+=1
                if obj['coarseSemantic'] in obj_dict.keys():
                    obj_dict[obj['coarseSemantic']] += 1
                else:
                    obj_dict[obj['coarseSemantic']] = 1
            text = f'a photo of {roomType}'
            for key,value in obj_dict.items(): 
                text += f',{value} {key}'
            return [hist], [text], [view]
        
        else:
            pass



    def load_room_as_scene(self,room):
        all_meshes = []

        count = 0
        scene_vert_color = None
        scene_vert_semantic = None
        for obj in room['objList']:
            if obj['inDatabase']:
                obj_filename = os.path.join(OBJ_DIR, obj['modelId'],f'{obj['modelId']}.obj')
                mesh = load_objs_as_meshes([obj_filename],device=self.device)
                translate = torch.Tensor(obj['translate'])
                scale = torch.Tensor(obj['scale'])
                rotate = torch.Tensor(obj['rotate'])

                S = torch.diag(torch.tensor([scale[0],scale[1],scale[2],1.0])).to(self.device)

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
                obj_semantic = torch.zeros((len(mesh.verts_list()[0]), len(category_to_color)))
                obj_semantic[..., list(category_to_color.keys()).index(obj['coarseSemantic'])] = 1
                obj_color = torch.tensor([category_to_color[obj['coarseSemantic']].copy()] * len(mesh.verts_list()[0]))
                if count == 0:
                    scene_vert_color = obj_color
                    scene_vert_semantic = obj_semantic
                else:
                    scene_vert_color = torch.cat([scene_vert_color, obj_color], dim=0)
                    scene_vert_semantic = torch.cat([scene_vert_semantic, obj_semantic],dim=0)
                count +=1 
        
        scene = join_meshes_as_scene(all_meshes)
        return scene, scene_vert_color, scene_vert_semantic

    def initialize_room(self):
        for room_id,room in enumerate(self.scene['rooms']):
            mesh, vert_color, vert_semantic = self.load_room_as_scene(room)
            self.room_mesh_list.append(mesh)
            self.color_list.append(vert_color)
            self.semantic_list.append(vert_semantic)

    def initialize_view(self, renderer ,mode = "top"):
        if mode == "manual":
            return 
        histos = []
        texts = []
        candidate_views= []
        for room_id, room in enumerate(self.scene['rooms']):
            objCount = 0
            obj_dict = {}
            for obj in room['objList']:
                if obj['coarseSemantic'] in ['Window','Door']:
                    continue
                objCount+=1
                if obj['coarseSemantic'] in obj_dict.keys():
                    obj_dict[obj['coarseSemantic']] += 1
                else:
                    obj_dict[obj['coarseSemantic']] = 1
            if objCount == 0:
                continue
            h, t, v= self.get_texts_and_colorhisto(renderer,room, room_id, mode)
            # print(f"The shape of histograms is {h.shape}")
            histos += h
            texts += t
            candidate_views +=v
        tokens = clip.tokenize(texts).to(self.device)
        if self.args.debug:
            print(texts)
            print(tokens.shape)
        self.model.eval()
        img = self.gt_img.permute((0,3,1,2))
        logits_per_image, logits_per_text = self.model(self.preprocess(img), tokens)
        
        histos = torch.Tensor(histos)
        gt_histos = torch.Tensor(self.hist)
        d_histo = torch.sum((histos - gt_histos) ** 2, dim=1).cuda()

        # torch.nn.functional.cosine_similarity(histos, gt_histos,dim=1).cuda()
        dist = logits_per_image - d_histo

        probs= dist.softmax(dim = -1).detach().cpu()[0]
        view_id = torch.argmax(probs)

        # return a start view
        return candidate_views[view_id]


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

            room_token = f'a photo of {roomType}'
            for key,value in obj_dict.items(): 
                room_token += f',{value} {key}'
            semantic_tokens.append(room_token[:77])

        if args.debug:
            for t in semantic_tokens:
                print(t)
        text = clip.tokenize(semantic_tokens).to(self.device)
        self.model.eval()
        logits_per_image, logits_per_text = self.model(self.gt_img, text)
        print(f"logits_per_image shape is: {logits_per_image.shape}, {logits_per_image}")
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0]
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
        # for room_id in self.probe_idx[:3]:
        #     room = self.scene['rooms'][room_id]
        #     self.room_mesh_list.append(self.load_room_as_scene(room))
        #     labels = self.furnitureCluster(room)
        #     furniture_groups = [[] for _ in range(len(labels))]
        #     idx =0
        #     for obj in room['objList']:
        #         if 'coarseSemantic' not in obj or obj["coarseSemantic"] == 'Window' or obj['coarseSemantic'] == 'Door':
        #             continue
        #         furniture_groups[labels[idx]].append(obj)
        #         idx += 1
            
        #     group_tokens = []

        #     for g in furniture_groups:
        #         token = 'a sketch containing'

        #         for obj in g:
        #             token += f' {obj['coarseSemantic']}'
                
        #         group_tokens.append(token)
            
        #     group_t = clip.tokenize(group_tokens)
        #     logits_per_image, logits_per_text = self.model(self.sketch, group_t)
        #     probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0]
        #     print(probs)                                                                                

        
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
            
                self.gt_img = self.preprocess(Image.open(os.path.join(sketch_dir,token,img))).unsqueeze(0).to(self.args.device)
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


