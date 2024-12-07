import torch
import torch.nn as nn
import collections
import CLIP_.clip as clip
from PIL import Image
from torchvision import transforms


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

        self.args = args
        self.losses_to_apply = self.get_losses_to_apply()
        print(self.losses_to_apply)
        self.loss_mapper = {
            'clip':CLIPLoss(args),
            "clip_conv_loss": CLIPConvLoss(args)
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
                    sketches, targets).mean()
            else:
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets, mode).mean()
            # loss = loss + self.loss_mapper[loss_name](sketches, targets).mean() * loss_coeffs[loss_name]

        for key in self.losses_to_apply:
            # loss = loss + losses_dict[key] * loss_coeffs[key]
            losses_dict[key] = losses_dict[key] * loss_coeffs[key]
        # print(losses_dict)
        return losses_dict


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
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
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
        ])  # clip normalisation
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
        #         y = self.target_transform(target).to(self.args.device)
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





def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    if "RN" in clip_model_name:
        return [torch.square(x_conv, y_conv, dim=1).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device,jit=False)

    image = preprocess(Image.open("./data/sketch/sketch1.jpg")).unsqueeze(0).to(device)
    image2 = preprocess(Image.open("./data/sketch/sketch3.jpg")).unsqueeze(0).to(device)

    text = clip.tokenize(["a bedroom with a a bed in the center and chair around it, with a waredobe", "a living room with a dog", "a cat","a half camel",'a camel']).to(device)

    print(image.shape)

    model.eval()
    image1_features = model.encode_image(image)
    image2_features = model.encode_image(image2)
    text_features = model.encode_text(text)

    print(image1_features.shape)
    print(f"cosine distance: {torch.cosine_similarity(image1_features, text_features)}")

    logits_per_image, logits_per_text = model(image, text)

    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
