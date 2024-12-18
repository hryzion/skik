import json
import os

from PIL import Image
import config

import CLIP_.clip as clip
import torch


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
                
                room_token += f', {value} {key}'
            
            semantic_tokens.append(room_token)

        if args.debug:
            for t in semantic_tokens:
                print(t)
        text = clip.tokenize(semantic_tokens).to(self.device)

        self.model.eval()
        # sketch_feature = self.model.encode_image(self.sketch)
        # text_feature = self.model.encode_text(text)

        logits_per_image, logits_per_text = self.model(self.sketch, text)

        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0]
        print(probs)
        
        probs = list(probs)

        new_p = probs[:]
        new_p.sort(reverse=True)


        self.probe_idx = []
        for p in new_p:
            if p < self.eps:
                break
            self.probe_idx.append(probs.index(p))
        
        
        
        


                
                



if __name__ == '__main__':
    args = config.parse_arguments()
    a = Initializer(args)
    a.initialize()


