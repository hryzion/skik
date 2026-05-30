import cv2
import numpy as np


labels = np.load("../../dataset/semantics/train/gt/0_semantic.npy")

H, W, C = labels.shape
with open("../../dataset/yolo/labels/teset.txt","w+") as f:

    for i in  range(C):
        mask = labels[...,i]
        kernel = np.ones((7, 7), np.uint8)  # 5x5 的正方形核
        mask = (mask>0.5).astype(np.uint8)
        mask*=255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for j in range(len(contours)):
            info = f"{j}"
            for pt in contours[j]:
                info += f" {(pt[0][1]+0.0)/W} {(pt[0][0]+0.0)/H}"
            f.write(info)
        
f.close()
        # output_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
        # cv2.imwrite(f"test_{i}.png",output_image)
