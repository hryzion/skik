import torch
import torch.nn as nn
from torchvision.models import googlenet
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,RandomResizedCrop,RandomHorizontalFlip
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import json
from utils import *
import tqdm


DATA_DIR = "../../dataset/views"
EPOCHS = 100

class PoseDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode = "train"):
        """
        Args:
            root_dir (str): 数据集根目录（包含每个类别的子文件夹）
            transform (callable): 可选的数据增强
        """
        self.img_root_dir = os.path.join(DATA_DIR, root_dir, 'img', mode)
        self.pose_root_dir = os.path.join(DATA_DIR, root_dir, 'views', mode)

        self.image_fps = os.listdir(self.img_root_dir)
        self.pose_fps = os.listdir(self.pose_root_dir)

        self.transform = transform
        if not transform:
            self.transform = ToTensor()

        # 收集所有图像路径和对应的标签
    def __len__(self):
        return len(self.image_fps)

    def __getitem__(self, idx):
        img_path =os.path.join(self.img_root_dir,self.image_fps[idx])
        image = Image.open(img_path)  # 统一转为RGB
        
        if self.transform:
            image = self.transform(image)

        pose_path = os.path.join(self.pose_root_dir, self.pose_fps[idx])
        with open(pose_path,'r') as f:
            pose_json = json.load(f)
        dir_pos= to_direction(pose_json['gt_view'])
        # print(dir_pos['position'], dir_pos['direction'])
        dir_pos = torch.cat((dir_pos['position'], dir_pos['direction']),dim=-1)

    
        return image,  dir_pos # 返回 tensor 图像和标签





class PoseNet(nn.Module):
    def __init__(self, pretrained=True, aux_loss=True):
        super().__init__()
        original_model = googlenet(pretrained=pretrained, aux_logits=True)
        
        # 特征提取主干网络（保留Inception模块）
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.conv2,
            original_model.conv3,
            original_model.inception3a,
            original_model.inception3b,
            original_model.inception4a,
            original_model.inception4b,
            original_model.inception4c,
            original_model.inception4d,
            original_model.inception4e,
            original_model.inception5a,
            original_model.inception5b
        )
        
        # 修改辅助分类器（输出7D姿态）
        # self.aux1 = self._make_aux(original_model.aux1)  # inception4a的输出
        # self.aux2 = self._make_aux(original_model.aux2)  # inception4d的输出
        
        # 修改主分类器（增加2048维全连接层）
        self.avgpool = original_model.avgpool
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, 2048)  # 新增全连接层
        self.fc2 = nn.Linear(2048, 6)     # 最终回归层
        
        self.aux_loss = aux_loss  # 是否使用辅助损失

    # def _make_aux(self, original_aux):
    #     """将辅助分类器转换为7D姿态回归器"""
    #     return nn.Sequential(
    #         original_aux.conv,
    #         nn.Flatten(),
    #         nn.Linear(original_aux.fc.in_features, 7)  # 修改输出维度
    #     )

    def forward(self, x):
        # 前向传播路径
        x = self.features(x)
        
        # 辅助分类器输出（仅在训练阶段保留）
        # aux1, aux2 = None, None
        # if self.training and self.aux_loss:
        #     aux1 = self.aux1(x)  # inception4a输出
        #     aux2 = self.aux2(x)  # inception4d输出
        
        # 主分类器路径
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)     # 新增的2048维层
        x = self.fc2(x)     # 最终7D输出
        
        # 训练时返回多个输出
     
        return x

def pose_postprocess(output):
    """测试阶段四元数标准化"""
    position = output[:, :3]
    orientation = output[:, 3:]
    orientation = nn.functional.normalize(orientation, p=2, dim=1)
    return torch.cat([position, orientation], dim=1)

# 数据预处理器定义
train_transform = Compose([
    Resize(256),                 # Step1: 缩放至256像素
    RandomResizedCrop(224),      # Step2: 随机裁剪
    RandomHorizontalFlip(),      # 数据增强
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = Compose([
    Resize(256),        # Step1: 缩放至256像素
    CenterCrop(224),    # Step2: 中心裁剪
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试阶段密集评估函数
def dense_evaluation(model, image):
    """生成128个裁剪并平均预测结果"""
    crops = []
    image = Resize(320)(image)  # 放大图像以保留细节
    
    # 生成128个裁剪区域
    for i in range(8):
        for j in range(8):
            crop = F.crop(image, i*32, j*32, 224, 224)
            crop = test_transform(crop)
            crops.append(crop)
    
    # 分批推断
    with torch.no_grad():
        crops = torch.stack(crops).to(device)
        outputs = model(crops)
        outputs = torch.stack([pose_postprocess(o) for o in outputs])
        return outputs.mean(dim=0)
    
# 定义损失函数（位置MSE + 四元数角度差）
def pose_loss(pred, target):
    position_loss = nn.MSELoss()(pred[:, :3], target[:, :3])
    
    orientation_loss = nn.MSELoss()(pred[:, 3:] / torch.norm(pred[:,3:]),target[:, 3:])
    return position_loss + 200 * orientation_loss  # 加权损失系数

def save_ckpt(model, optimizer, epoch, loss, save_dir="checkpoints", filename="model_ckpt.pth"):
    """保存 checkpoint"""
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
    save_path = os.path.join(save_dir, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)

    print(f"Checkpoint saved at epoch {epoch} -> {save_path}")



def validate_model(model, val_loader, criterion, device='cuda:0'):
    """在验证集上评估模型"""
    model.eval()  # 进入 eval 模式
    val_loss = 0.0
    ori_loss = 0.
    dir_loss = 0.
    with torch.no_grad():  # 关闭梯度计算
        for images, pose in val_loader:
            images, pose = images.to(device), pose.to(device)
            outputs = model(images)
            loss = criterion(outputs, pose)
            val_loss += loss.item()

            d_p = torch.sqrt(torch.sum((outputs[:,:3] - pose[:,:3])**2))
            nod = outputs[:,3:]/torch.norm(outputs[:,3:])
            dot_prod = torch.sum(nod * pose[:,3:],dim=1)
            d_d = sum(torch.acos(dot_prod) * 180 / np.pi)
            ori_loss += d_p
            dir_loss += d_d

    avg_val_loss = val_loss / len(test_dataset)
    avg_ori_loss = ori_loss / len(test_dataset)
    avg_dir_loss = dir_loss / len(test_dataset)
    print(f"average distance = {avg_ori_loss}m, average angle = {avg_dir_loss}d")
    model.train()  # 恢复 train 模式
    return avg_val_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda:1', save_dir="checkpoints"):
    
    # with torch.autograd.detect_anomaly():
        best_val_loss = float('inf')  # 记录最低的验证损失
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for images, pose in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

                images, pose = images.to(device), pose.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, pose)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        

            avg_train_loss = epoch_loss / len(train_dataset)
            avg_val_loss = validate_model(
                model, val_loader, criterion, device)  # 计算验证损失
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # **保存模型：如果验证损失下降**
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_ckpt(model, optimizer, epoch+1, avg_val_loss,
                        save_dir, filename="best_model.pth")

            # **每 5 轮保存一次 checkpoint**
            if (epoch + 1) % 5 == 0:
                save_ckpt(model, optimizer, epoch+1, avg_val_loss,
                        save_dir, filename=f"epoch_{epoch+1}.pth")


# 用法示例
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    for scene_tokens in os.listdir(DATA_DIR):
        train_dataset = PoseDataset(
            root_dir=scene_tokens,
            transform=train_transform,
        )

        test_dataset = PoseDataset(
            root_dir=scene_tokens,
            transform=test_transform,
            mode='val'
        )

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        valid_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # 初始化模型
        posenet_model = PoseNet(pretrained=True).to(device)
        criterion = pose_loss
        optimizer = torch.optim.Adam(posenet_model.parameters(), lr=1e-5)


        train_model(posenet_model, train_loader, valid_loader, 
                criterion, optimizer, EPOCHS, device=device)

    
    
