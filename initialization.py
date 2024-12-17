import json
import os

import cv2
import config

import CLIP_.clip as clip


class Initializer:
    
    def __init__(self, args):
        self.dataset_dir = args.dataset_dir
        self.scene_name = args.scene_name
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device,jit=False)
        with open(os.path.join(self.dataset_dir,self.scene_name+'.json'),'r') as f:
            self.scene = json.load(f)
            f.close()
        self.args = args


    def initialize(self):
        semantic_tokens = []
        for room in self.scene['rooms']:
            objCount = 0
            obj_dict = {}
            roomType = room['roomType']
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
            for key, value in obj_dict:
                room_token += f', {value} {key}'
            
            semantic_tokens.append(room_token)

        if args.debug:
            for t in semantic_tokens:
                print(t)
            
        


                
                



if __name__ == '__main__':
    args = config.parse_arguments()


