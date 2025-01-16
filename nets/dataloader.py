import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import scipy.io
import numpy as np
from config import *

dataset_base_dir='../SketchyScene-7k'


class SKiSDataset(Dataset):
    def __init__(self, data_base_dir, mode = 'train',transform=None):
        self.data_paths = os.path.join(data_base_dir,mode)
        color_map_mat_path = os.path.join(data_base_dir, 'colorMapC46.mat')
        self.colorMap = scipy.io.loadmat(color_map_mat_path)['colorMap']
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像大小调整为 224x224
            transforms.ToTensor(),  # 将 PIL.Image 转换为 torch.Tensor
            # 可以添加其他转换操作，例如 Normalize
        ])
                

    def __len__(self):
        return len(os.listdir(os.path.join(self.data_paths, "DRAWING_GT")))  # 数据集大小

    def __getitem__(self, idx):
        image_name = 'L0_sample' + str(idx + 1) + '.png'
        image_path = os.path.join(self.data_paths,'DRAWING_GT',image_name)
        image = Image.open(image_path).convert('RGB')  # 加载图像并转换为RGB
        image = self.transform(image)
        
        mask_class_name = 'sample_' + str(idx + 1) + '_class.mat'
        mask_instance_name = 'sample_' + str(idx + 1) + '_instance.mat'

        class_base_dir = os.path.join(self.data_paths, 'CLASS_GT')
        instance_base_dir = os.path.join(self.data_paths, 'INSTANCE_GT')
        mask_class_path = os.path.join(class_base_dir, mask_class_name)
        mask_instance_path = os.path.join(instance_base_dir, mask_instance_name)

        INSTANCE_GT = scipy.io.loadmat(mask_instance_path)['INSTANCE_GT']
        INSTANCE_GT = np.array(INSTANCE_GT, dtype=np.uint8)  # shape=(750, 750)
        CLASS_GT = scipy.io.loadmat(mask_class_path)['CLASS_GT']  # (750, 750)
        CLASS_GT = np.array(CLASS_GT,dtype=np.uint8)
        
        lab_classes=np.zeros([num_classes,CLASS_GT.shape[0],CLASS_GT.shape[1]])
        for i in range(lab_classes.shape[0]):
            lab_classes[i,CLASS_GT == i] = 1
        # print(lab_classes.shape)




        # # print(np.max(INSTANCE_GT))  # e.g. 101
        # instance_count = np.bincount(INSTANCE_GT.flatten())
        # # print(instance_count.shape)  # e.g. shape=(102,)

        # instance_count = instance_count[1:]  # e.g. shape=(101,)
        # nonzero_count = np.count_nonzero(instance_count)  # e.g. 16
        # # print("nonzero_count", nonzero_count)  # e.g. shape=(102,)

        # mask_set = np.zeros([nonzero_count, INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.uint8)
        # class_id_set = np.zeros([nonzero_count], dtype=np.uint8)

        # real_instanceIdx = 0
        # for i in range(instance_count.shape[0]):
        #     if instance_count[i] == 0:
        #         continue

        #     instanceIdx = i + 1

        #     ## mask
        #     mask = np.zeros([INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.uint8)
        #     mask[INSTANCE_GT == instanceIdx] = 1
        #     mask_set[real_instanceIdx] = mask

        #     class_gt_filtered = CLASS_GT * mask
        #     class_gt_filtered = np.bincount(class_gt_filtered.flatten())
        #     class_gt_filtered = class_gt_filtered[1:]
        #     class_id = np.argmax(class_gt_filtered) + 1

        #     class_id_set[real_instanceIdx] = class_id

        #     real_instanceIdx += 1

        # mask_set = np.transpose(mask_set, (1, 2, 0))

        
        return image, lab_classes
