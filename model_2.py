import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import sys
import os
from PIL import Image

'''
this is the code modified by ChatGPT based on model.py
now we are examing how this could be different from our code.
'''

class SingleInputResNet(nn.Module):
    def __init__(self):
        super(SingleInputResNet, self).__init__()
        # 加载预训练的 ResNet-50 模型
        self.resnet = models.resnet50(pretrained=True)
        
        # 冻结模型的所有参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 修改 ResNet-50 的输入通道数
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 获取 ResNet-50 的全连接层输入特征数
        num_ftrs = self.resnet.fc.in_features
        
        # 定义新的全连接层，输出维度为 2，即两个类别
        self.fc = nn.Linear(num_ftrs, 2)  # 输入特征数为 num_ftrs，输出维度为 2

    def forward(self, input):
        # 传递输入图像到 ResNet-50 模型，直到最后一个卷积层的输出
        x = self.resnet.conv1(input)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # 应用全局平均池化
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)  # 展平特征向量
        
        # 使用新的全连接层进行分类
        output = self.fc(x)
        return output


def initialize_model():
    model = SingleInputResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, criterion, optimizer


def train_model(model, dataloaders, criterion, optimizer, num_epochs=24):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            optimizer.zero_grad() 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        
        print(f'Epoch {epoch}/{num_epochs - 1} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


if __name__ == "__main__":
    model, criterion, optimizer = initialize_model()
    input = torch.randn(1, 4, 224, 224)
