import torch
import torch.nn as nn
print(torch.__version__)
import CLIP_.clip as clip
from PIL import Image


class CLIPLoss(nn.Module):
    def __init__(self, args):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device,jit=False)
        self.device = "cuda"
    
    def forward(self,sketch_view, sketch_input, mode):
        if self.calc_input:
            targets_ = self.preprocess(sketch_input).to(self.device)
            self.targets_features = self.model.encode_image(targets_).detach()
            self.calc_target = False
        
        sketch_feature = self.preprocess(sketch_view).to(self.device)
        loss_clip = 1. - torch.cosine_similarity(sketch_feature, self.targets_features)
        return loss_clip





device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device,jit=False)

image = preprocess(Image.open("./data/sketch/sketch1.jpg")).unsqueeze(0).to(device)
image2 = preprocess(Image.open("./data/sketch/sketch3.jpg")).unsqueeze(0).to(device)

text = clip.tokenize(["a bedroom", "a sketch", "a cat","a half camel",'a camel']).to(device)

model.eval()
image1_features = model.encode_image(image)
image2_features = model.encode_image(image2)
text_features = model.encode_text(text)

print(f"cosine distance: {torch.cosine_similarity(image1_features, image2_features)}")

logits_per_image, logits_per_text = model(image, text)
probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
