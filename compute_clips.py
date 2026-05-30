import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from CLIP_.clip import clip


def load_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def compute_clip_scores(image_dir, text_dir, device="cuda"):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    scores = []
    valid_pairs = []

    for img_name in tqdm(image_files):
        img_path = os.path.join(image_dir, img_name)

        # 对应 json 文件
        base_name = os.path.splitext(img_name)[0]
        json_path = os.path.join(text_dir, base_name + ".json")

        if not os.path.exists(json_path):
            print(f"Missing json for {img_name}")
            continue

        # 读取图像
        image = load_image(img_path)
        if image is None:
            continue

        # 读取文本
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            text = data["description"]
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            continue

        # 预处理
        image_input = preprocess(image).unsqueeze(0).to(device)
        text = 'Semantic map for:'+text
        text_input = clip.tokenize([text[:100]]).to(device)

        
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity
        score = (image_features @ text_features.T).item()

        scores.append(score)
        valid_pairs.append((img_name, score))

    return scores, valid_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="clip_scores")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    scores, pairs = compute_clip_scores(
        args.image_dir,
        args.text_dir,
        args.device
    )

    scores = np.array(scores)

    print(f"\nTotal valid samples: {len(scores)}")
    print(f"Mean CLIP Score: {scores.mean():.6f}")
    print(f'Variance CLIP Score: {scores.var():.6f}')

    # 保存 numpy
    np.save(args.output_prefix + ".npy", scores)

    # 保存 txt（方便后续分析）
    with open(args.output_prefix + ".txt", "w") as f:
        for name, s in pairs:
            f.write(f"{name}\t{s:.6f}\n")

    print(f"Saved scores to {args.output_prefix}.npy and .txt")


if __name__ == "__main__":
    main()
