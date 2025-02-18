import torch
import torch.nn as nn
import collections
import CLIP_.clip as clip
from PIL import Image
from torchvision import transforms
from geomloss import SamplesLoss
import numpy as np


class LossFunc(nn.Module):
    def __init__(self, args = None):
        super(LossFunc, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_with_clip = args.train_with_clip
        self.clip_weight = args.clip_weight
        self.start_clip = args.start_clip

        self.clip_conv_loss = args.clip_conv_loss
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.clip_text_guide = args.clip_text_guide

        self.vae_loss = args.vae_loss
        self.orb_loss = args.orb_loss
        self.sample_loss = args.sample_loss
        self.l2_loss = args.l2_loss
        self.semantic_loss = args.semantic_loss

        self.args = args
        self.losses_to_apply = self.get_losses_to_apply()
        print(self.losses_to_apply)
        self.loss_mapper = {
            # 'clip':CLIPLoss(args),
            # "clip_conv_loss": CLIPConvLoss(args),
            'sample_loss' :SinkhornLoss(args),
            'orb_loss':ORBLoss(args),
            "l2" :L2Loss(args),
            "semantic_loss":SemanticLoss(args)
        }

    def get_losses_to_apply(self):
        losses_to_apply = []
        if self.train_with_clip and self.start_clip == 0:
            losses_to_apply.append("clip")
        if self.clip_conv_loss:
            losses_to_apply.append("clip_conv_loss")
        if self.clip_text_guide:
            losses_to_apply.append("clip_text")
        if self.vae_loss:
            losses_to_apply.append('vae')
        if self.orb_loss:
            losses_to_apply.append("orb_loss")
        if self.sample_loss:
            losses_to_apply.append("sample_loss")
        if self.l2_loss:
            losses_to_apply.append("l2")
        if self.semantic_loss:
            losses_to_apply.append("semantic_loss")
        return losses_to_apply
    
    def forward(self, sketches, targets,  mode="train"):
 
        losses_dict = dict.fromkeys(
            self.losses_to_apply, torch.tensor([0.0]).to(self.args.device))
        loss_coeffs = dict.fromkeys(self.losses_to_apply, 1.0)
        loss_coeffs["clip"] = self.clip_weight
        loss_coeffs["clip_text"] = self.clip_text_guide

        for loss_name in self.losses_to_apply:
            if loss_name in ["clip_conv_loss"]:
                conv_loss = self.loss_mapper[loss_name](
                    sketches, targets, mode)
                for layer in conv_loss.keys():
                    losses_dict[layer] = conv_loss[layer]
            elif loss_name == "l2":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets)
            elif loss_name in ["sample_loss", "semantic_loss"]:
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets
                )
            else:
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets, mode).mean()
            # loss = loss + self.loss_mapper[loss_name](sketches, targets).mean() * loss_coeffs[loss_name]

        for key in self.losses_to_apply:
            # loss = loss + losses_dict[key] * loss_coeffs[key]
            losses_dict[key] = losses_dict[key] * loss_coeffs[key]
        # print(losses_dict)
        return losses_dict
    

class L2Loss(nn.Module):
    def __init__(self, args = None):
        super(L2Loss, self).__init__()
        self.device = args.device
    def forward(self, img, target):
        return torch.sum((img - target)** 2) 

class SemanticLoss(nn.Module):
    def __init__(self, args = None):
        super(SemanticLoss, self).__init__()
        self.devive = args.device
    
    def forward(self, render_res,gt_res):
        render_semantic = render_res['semantics']
        gt_semantic = gt_res['semantics']
        return torch.sum((render_semantic - gt_semantic) ** 2) * 2
        


# TODO: Yirui to complete
class ORBLoss(nn.Module):

    def __init__(self, args = None):
        super(ORBLoss, self).__init__()
        self.device = args.device

        
    def forward(self,sketch, target):
        import cv2
        self.tensor_img1 = sketch
        self.tensor_img2 = target
        self.tensor_img1 = (self.rgb_to_grayscale(self.tensor_img1)*255).clamp(0,255)
        self.tensor_img2 = (self.rgb_to_grayscale(self.tensor_img2)*255).clamp(0,255)
        cv_img1 = self.tensor_to_np(self.tensor_img1).astype(np.uint8)
        cv_img2 = self.tensor_to_np(self.tensor_img2).astype(np.uint8)
        #现tensor_img形状为Tensor:(1,1,224,224)
        # 实际上会自动转化为灰度图再进行SIFT操作
        # cv_img1 = cv2.cvtColor(np_img1, cv2.COLOR_BGR2GRAY)
        # cv_img2 = cv2.cvtColor(np_img2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        #此处描述子用于match，不用于计算loss
        keypoints1, descriptors1 = orb.detectAndCompute(cv_img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(cv_img2, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # L2表示使用欧式距离，crossCheck=True确保一对匹配是双向一致的
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)  # 排序
        matches = matches[:10]  # 取前十个
        #下为两组匹配点的二维列表
        points1 = np.array(np.round([keypoints1[m.queryIdx].pt for m in matches]))
        points2 = np.array(np.round([keypoints2[m.trainIdx].pt for m in matches]))
        points1_tensor = torch.tensor(points1,dtype=torch.float32,device=self.device, requires_grad=True)
        points2_tensor = torch.tensor(points2,dtype=torch.float32,device=self.device, requires_grad=True)
        #计算两组点的误差
        loss_1 = torch.mean(torch.norm(points1_tensor - points2_tensor, dim=1))#匹配点的距离求平均
        loss_2 = torch.mean((points1_tensor - points2_tensor)**2)#两个张量的mse
        #如果认为匹配组合i的x,y方向的误差与匹配组合j的x,y方向的误差地位相等，可以采用loss_2即mse
        return loss_1

    def rgb_to_grayscale(tensor):
        """
        将彩色图像张量转换为灰度图，保持计算图以支持反向传播。

        参数:
            tensor (torch.Tensor): 形状为 (B, 3, H, W) 的输入 RGB 图像张量。

        返回:
            torch.Tensor: 形状为 (B, 1, H, W) 的灰度图张量。
        """
        # RGB 通道权重
        weights = torch.tensor([0.299, 0.587, 0.114], device=tensor.device, dtype=tensor.dtype)
        if tensor.shape == torch.Size([1,224,224,4]):#sketch和render的shape不同，在这里统一处理
            tensor = tensor[...,0:3]
            tensor = torch.permute(tensor,(0,3,1,2))
        grayscale = torch.tensordot(tensor, weights, dims=([1], [0]))
        # 增加一个通道维度 (B, H, W) -> (B, 1, H, W)
        grayscale = grayscale.unsqueeze(1)
        return grayscale

    def tensor_to_np(tensor):
        tensor = tensor[0]
        tensor = tensor.permute(1, 2, 0)
        nparray = tensor.detach().cpu().numpy()
        return nparray

class SinkhornLoss(nn.Module):
    def __init__(self, args = None):
        super(SinkhornLoss, self).__init__()
        self.device = args.device
        self.loss = SamplesLoss("sinkhorn", blur=0.05)
        self.res = args.res
        x =torch.linspace(0,1,self.res)
        y = torch.linspace(0,1,self.res)
        pts = torch.meshgrid(x, y)
        self.pos = torch.cat([pts[1][...,None],pts[0][...,None]],dim=2)[None, ...].to(self.device) # 1, H, W, 2
        
    def match_point(self, haspos, render_point_5d, gt_rgb, view):
        _ , h, w, c = render_point_5d.shape
        target_point_5d = torch.zeros((haspos.shape[0], h, w, 5), device=self.device)
        target_point_5d[..., :3] = torch.clamp(gt_rgb,0,1)
        target_point_5d[..., 3:] = render_point_5d[...,3:].clone().detach()
        target_point_5d = target_point_5d.reshape(-1, h*w, 5)
        render_point_5d_match = render_point_5d.clone().reshape(-1,h*w,5)
        render_point_5d_match.clamp_(0.0,1.0)
        # render_point_5d_match[...,:3] *= self.rgb_match_weight(view)
        # target_point_5d[...,:3] = target_point_5d[...,:3]*self.rgb_match_weight(view)
        # print(target_point_5d.shape)
        pointloss = self.loss(render_point_5d_match, target_point_5d)*self.res*self.res
        # print(pointloss)
        [g] = torch.autograd.grad(torch.sum(pointloss), [render_point_5d_match])
        # print(g)
        # g[...,:3]/=self.rgb_match_weight(view)
        
        return (render_point_5d-g.reshape(-1,h,w,5)).detach()

        

    def forward(self, render_res, gt_res, view  = 0): # 1 ,1, H, W
        new_match = True

        haspos = render_res["msk"]
        render_pos = (render_res["pos"]+1.0)/2.0
        render_rgb = render_res["images"]
        render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
        render_point_5d = torch.cat([render_rgb, render_pos], dim=-1)
        gt_rgb=gt_res["images"][view:view+1]
        
        match_point_5d = self.match_point(haspos,render_point_5d,gt_rgb,view)
        dist = match_point_5d - render_point_5d
        loss = torch.sum(dist**2)

        return loss

        

        

class CLIPLoss(nn.Module):
    def __init__(self, args = None):
        super(CLIPLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device,jit=False)
        self.calc_input = True
    
    def forward(self,sketch_view, sketch_input, mode = 'train'):
        if self.calc_input:
            targets_ = sketch_input.to(self.device)
            print("target_", targets_.shape)
            self.targets_features = self.model.encode_image(targets_).detach()
            self.calc_target = False
        
        sketch_feature = self.model.encode_image(sketch_view).to(self.device)
        # print('sketch feature:', sketch_feature)
        loss_clip = 1. - torch.cosine_similarity(sketch_feature, self.targets_features)
        return loss_clip


class CLIPFeatureMap(nn.Module):
    def __init__(self, clip_model) -> None:
        super().__init__()
        self.clip = clip_model
        self.featuremaps = None

        for i in range(12):
            self.clip.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i)
            )
    def make_hook(self, layer):
        def hook(module, input, output):
            if len(output.shape)==3:
                self.featuremaps[layer] = output.permute(1,0,2)
            else:
                self.featuremaps[layer] = output
        
        return hook

    def forward(self,x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps
    
def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    ret = []
    for x_conv, y_conv in zip(xs_conv_features, ys_conv_features):
        w,h = x_conv.shape[1],x_conv.shape[2]
        ret.append(w*h*torch.square(x_conv - y_conv).mean())

    return ret


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).sum() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]

def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    if "RN" in clip_model_name:
        return [torch.square(x_conv, y_conv, dim=1).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


class CLIPConvLoss(torch.nn.Module):
    def __init__(self, args = None):
        super(CLIPConvLoss, self).__init__()
        self.clip_model_name = args.clip_model_name
        assert self.clip_model_name in [
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "ViT-B/32",
            "ViT-B/16",
        ]

        self.clip_conv_loss_type = args.clip_conv_loss_type
        self.clip_fc_loss_type = "Cos"  # args.clip_fc_loss_type
        assert self.clip_conv_loss_type in [
            "L2", "Cos", "L1",
        ]
        assert self.clip_fc_loss_type in [
            "L2", "Cos", "L1",
        ]

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, clip_preprocess = clip.load(
            self.clip_model_name, args.device, jit=False)

        if self.clip_model_name.startswith("ViT"):
            self.visual_encoder = CLIPFeatureMap(self.model)

        else:
            self.visual_model = self.model.visual
            layers = list(self.model.visual.children())
            init_layers = torch.nn.Sequential(*layers)[:8]
            self.layer1 = layers[8]
            self.layer2 = layers[9]
            self.layer3 = layers[10]
            self.layer4 = layers[11]
            self.att_pool2d = layers[12]

        self.args = args

        self.img_size = clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ]) 
         # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.device = args.device
        self.num_augs = self.args.num_aug_clip

        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.clip_fc_layer_dims = None  # self.args.clip_fc_layer_dims
        self.clip_conv_layer_dims = None  # self.args.clip_conv_layer_dims
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.counter = 0

    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        if sketch.shape[1] == 1:
            sketch = sketch.repeat(1,3,1,1)
        conv_loss_dict = {}
        x = sketch.to(self.device)
        y = target.to(self.device)
        sketch_augs, img_augs = [self.normalize_transform(x)], [
            self.normalize_transform(y)]
        if mode == "train":
            for n in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([x, y]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        if self.clip_model_name.startswith("RN"):
            xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(
                xs.contiguous())
            ys_fc_features, ys_conv_features = self.forward_inspection_clip_resnet(
                ys.detach())

        else:
            xs_fc_features, xs_conv_features = self.visual_encoder(xs)
            ys_fc_features, ys_conv_features = self.visual_encoder(ys)

        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, self.clip_model_name)

        for layer, w in enumerate(self.args.clip_conv_layer_weights):
            if w:
                conv_loss_dict[f"clip_conv_loss_layer{layer}"] = conv_loss[layer] * w

        if self.clip_fc_loss_weight:
            # fc distance is always cos
            fc_loss = (1 - torch.cosine_similarity(xs_fc_features,
                       ys_fc_features, dim=1)).mean()
            conv_loss_dict["fc"] = fc_loss * self.clip_fc_loss_weight

        self.counter += 1
        return conv_loss_dict

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x
        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]






if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device,jit=False)
    import config
    import os
    import tqdm
    args = config.parse_arguments()

    loss_func = LossFunc(args)

    transform = transforms.ToTensor()

    sketch_dir = rf"./sketch_results"
    image_dir = rf'./sample_results'


    corr = 0
    mist = 0
    dif =0
    for i,room in (enumerate(os.listdir(sketch_dir))):
        for sketch in tqdm.tqdm(os.listdir(os.path.join(sketch_dir,room))):
            
            # if len(sketch.split('-')) < 2 or len(sketch.split('-')) > 4:
            #     continue
            ske_img = transform(Image.open(os.path.join(sketch_dir,room,sketch))).unsqueeze(0).to(device)
            ren_img = transform(Image.open(os.path.join(image_dir,room,sketch))).unsqueeze(0).to(device)
            loss_base_dict = loss_func(ske_img,ren_img)
            loss_base = sum(list(loss_base_dict.values()))
            for rendered in os.listdir(os.path.join(image_dir,room)):
                if rendered == 'path':
                    continue
                # if len(rendered.split('-')) < 2 or len(rendered.split('-')) > 4:
                #     continue
                if os.path.splitext(rendered)[1] != '.jpg':
                    continue
                if sketch == rendered:
                    continue
                
                image2=transform(Image.open(os.path.join(image_dir,room,rendered))).unsqueeze(0).to(device)
                
                loss_diff_dict=loss_func(ske_img,image2)
                
                # print(os.path.join(image_dir,room,rendered))
                loss_diff = sum(list(loss_diff_dict.values()))
                if loss_diff > loss_base:
                    corr +=1 
                else:
                    mist+=1
                
                dif += loss_diff.detach().cpu() - loss_base.detach().cpu()

                
        
        print(corr,mist)
        print(f"Accuracy: {corr/(corr+mist)}")
        print(f'average diff: {dif/(corr+mist)}')

    # image = transform(Image.open(rf'D:\zhx_workspace\sketch\PhotoSketch\Exp\PhotoSketch\SketchResults\0b324ba6-32f3-4ea8-b3d7-710bf86014dc\room2-wellAlignedShifted-3.png')).repeat(3,1,1).unsqueeze(0).to(device)

    # image2 = transform(Image.open(rf'D:\zhx_workspace\3DScenePlatformDev\sceneviewer\results\0b324ba6-32f3-4ea8-b3d7-710bf86014dc\room2-wellAlignedShifted-3.png')).unsqueeze(0).to(device)

    # image3 = transform(Image.open(rf'D:\zhx_workspace\3DScenePlatformDev\sceneviewer\results\0b324ba6-32f3-4ea8-b3d7-710bf86014dc\room4-threeWall_R_thin-0.png')).unsqueeze(0).to(device)

    # loss1 = loss_func(image,image2)
    # print(loss1)

    # loss2 = loss_func(image,image3)
    # print(loss2)





    # image = preprocess(Image.open("./data/sketch/sketch1.jpg")).unsqueeze(0).to(device)
    # image2 = preprocess(Image.open("./data/sketch/sketch3.jpg")).unsqueeze(0).to(device)

    # text = clip.tokenize(["a bedroom with a a bed in the center and chair around it, with a waredobe", "a living room with a dog", "a cat","a half camel",'a camel']).to(device)

    # print(image.shape)

    # model.eval()
    # image1_features = model.encode_image(image)
    # image2_features = model.encode_image(image2)
    # text_features = model.encode_text(text)

    # print(image1_features.shape)
    # print(f"cosine distance: {torch.cosine_similarity(image1_features, text_features)}")

    # logits_per_image, logits_per_text = model(image, text)

    # probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
