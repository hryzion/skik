import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from dataloader import SKiSDataset
from SKIS_model import SwinTransformerEncoder

from config import *






def main():


    train_dataset = SKiSDataset("../SketchyScene-7k/")
    val_dataset = SKiSDataset("../SketchyScene-7k/",'val')
    print('before load train data...')

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('after load train data...')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # (224,224 --> 750, 750)
    skis_model = SwinTransformerEncoder()
    optimizer = optim.Adam(skis_model.parameters(), lr=learing_rate)
    criterion = nn.CrossEntropyLoss()
    # 训练模型
    device = torch.device( 'cpu')
    print(device)
    skis_model.to(device)

    running_loss = 0
    val_loss = []
    print('-----Begin Training-----')
    for epoch in range(num_epochs):
        # train
        skis_model.train()
        for i, (img, label) in enumerate(train_loader):
            
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            predict=skis_model(img)
            loss=criterion(label,predict)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss+=loss.item()

            if i % 50 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, loss:{loss.item()}')
               
        print(f'Epoch: {epoch+1}/{num_epochs}, loss: {running_loss/len(train_loader)}')


if __name__ == "__main__":
    main()