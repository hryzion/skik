import cv2
import torch.nn.functional as F
def compute_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # 归一化并展平
    return hist

def cosine_distance(hist1, hist2):
    cosine_sim = F.cosine_similarity(hist1, hist2, dim=1)
    # 余弦距离 = 1 - 余弦相似度
    return 1 - cosine_sim.item()