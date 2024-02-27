import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

def initialize_model():
    # 加载已经完成预训练的ResNet-50模型
    resnet50 = models.resnet50(pretrained=True)

    # 冻结模型的所有参数，避免在训练过程中更新它们
    for param in resnet50.parameters():
        param.requires_grad = False

    # 获取全连接层(fc)的输入特征数量
    num_ftrs = resnet50.fc.in_features

    # 替换全连接层以适应新的分类任务
    # 假设我们的新任务只有2个类别
    resnet50.fc = nn.Linear(num_ftrs, 2)
    print(resnet50.fc)

    # 打印修改后的模型结构
    print(resnet50)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet50.fc.parameters(), lr=0.001)

    # 接下来，你可以使用自定义的数据加载器来训练和测试模型
    # 注意：由于我们冻结了模型的大部分参数，仅全连接层的参数会在训练过程中更新



def Dataloader():
    # 你可以使用自定义的数据加载器来训练和测试模型
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

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

        print(f'Epoch {epoch}/{num_epochs - 1} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

def validate_model(model, dataloaders, criterion):
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['val']:
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(dataloaders['val'].dataset)
    total_acc = running_corrects.double() / len(dataloaders['val'].dataset)
    print(f'Val Loss: {total_loss:.4f} Acc: {total_acc:.4f}')


if __name__ == "__main__":
    # 初始化模型、损失函数和优化器
    model, criterion, optimizer = initialize_model()

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    train_model(model, dataloaders, criterion, optimizer, num_epochs=25)
    validate_model(model, dataloaders, criterion)

     