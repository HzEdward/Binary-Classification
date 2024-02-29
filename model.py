# 最基本pytorch的resnet实现
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import sys
import os
from PIL import Image

def initialize_model():
    resnet50 = models.resnet50(pretrained=True)

    #! 冻结模型的所有参数，避免在训练过程中更新它们
    for param in resnet50.parameters():
        param.requires_grad = False

    # 1.全连接层, 2048 -> 2
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, 2)
    # 2.损失函数: Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()
    # 3.优化器: Adam Optimizer, lr是学习率
    optimizer = torch.optim.Adam(resnet50.fc.parameters(), lr=0.001)

    return resnet50, criterion, optimizer


def get_dataloaders():
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 调整图像大小以匹配ResNet的输入尺寸
        transforms.ToTensor(), # 将图像转换成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 归一化
    ])

    train_dataset = datasets.ImageFolder(root='data_new/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='data_new/valid_new', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return {'train': train_loader, 'val': val_loader}

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
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

# validation 1: Given the images with labels, calculate the loss and accuracy of the model
def validate_model(model, dataloaders, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['val']:
        with torch.no_grad(): # 确保在推断时不计算梯度
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(dataloaders['val'].dataset)
    total_acc = running_corrects.double() / len(dataloaders['val'].dataset)
    print(f'Val Loss: {total_loss:.4f} Acc: {total_acc:.4f}')

# validation 2
# Without label, just classify the images
# output the predictions in the form of dictionary, with image name as key and predicted class as value
def classify_images(model, dataloader):  
    model.eval()  
    predictions = {}
    for inputs, filenames in dataloader['val']:  
        # 迭代遍历验证数据加载器 dataloader['val'] 中的每个批次
        # print("filenames: ", filenames) # tensor([0, 0, 0, 0, 1, 1, 1])
        with torch.no_grad():  # 确保在推断时不计算梯度
            outputs = model(inputs) 
            _, preds = torch.max(outputs, 1)    
            for filename, pred in zip(filenames, preds):                
                train_dataset = dataloader['val'].dataset
                predictions[filename] = train_dataset.classes[pred.item()]
    print("Predictions: ", predictions)

    return predictions


if __name__ == "__main__":
    # 1. 初始化模型
    model, criterion, optimizer = initialize_model()
    # 2. 获取数据加载器
    dataloaders = get_dataloaders()
    # 3. 训练模型
    train_model(model, dataloaders, criterion, optimizer, num_epochs=25)
    # 4. 验证模型
    validate_model(model, dataloaders, criterion)
    # 5. 分类图像
    classify_images(model, dataloaders)
