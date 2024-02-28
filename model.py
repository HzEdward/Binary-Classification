import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

def initialize_model():
    resnet50 = models.resnet50(pretrained=True)

    #! 冻结了模型的大部分参数，除了最后一层全连接层的参数
    for param in resnet50.parameters():
        param.requires_grad = False

    # 获取全连接层(fc)的输入特征数量
    # output: 2048 
    # (在 ResNet-50 中，最后一个卷积层的输出特征数是 2048)
    num_ftrs = resnet50.fc.in_features
    
    # 全连接层: Input参数+Output分类数量
    resnet50.fc = nn.Linear(num_ftrs, 2)

    # 损失函数：Entroy Loss
    criterion = nn.CrossEntropyLoss()

    # 优化器：Adam Optimizer
    optimizer = torch.optim.Adam(resnet50.fc.parameters(), lr=0.001)

    return resnet50, criterion, optimizer

def DataLoader():
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小以匹配ResNet的输入尺寸
        transforms.ToTensor(),  # 将图像转换成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    # 创建数据集
    train_dataset = datasets.ImageFolder(root='./train', transform=transform)
    val_dataset = datasets.ImageFolder(root='./valid', transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)