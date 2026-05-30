import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms
from PIL import Image
import os
import argparse
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_images(folder, transform):
    imgs = []
    for f in os.listdir(folder):
        if f.lower().endswith(".png"):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img = transform(img)
            img = torch.from_numpy(np.array(img))  # H W C uint8
            img = img.permute(2,0,1)               # C H W
            imgs.append(img)
    # print(imgs)
    return torch.stack(imgs).to(device)


def main(args):

    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=50).to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        # transforms.ToTensor()
    ])

    print("Loading real images...")
    real = load_images(args.real, transform)

    print("Loading fake images...")
    fake = load_images(args.fake, transform)

    fid.update(real, real=True)
    fid.update(fake, real=False)

    kid.update(real, real=True)
    kid.update(fake, real=False)

    print("FID:", fid.compute().item())
    print("KID:", kid.compute()[0].item())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute FID and KID")
    parser.add_argument("--real", required=True, help="real image folder")
    parser.add_argument("--fake", required=True, help="fake image folder")

    args = parser.parse_args()

    main(args)
