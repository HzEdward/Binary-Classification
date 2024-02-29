import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

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

    # 创建数据集, train_dataset和val_dataset是以
    # train_dataset = datasets.ImageFolder(root='data_new/train', transform=transform)
    # val_dataset = datasets.ImageFolder(root='data_new/valid', transform=transform)

    train_dataset = CustomImageFolder(root='data_new/train', transform=transform)
    val_dataset = CustomImageFolder(root='data_new/valid', transform=transform)

    # 创建数据加载器
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

# validation 1
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


# 还是有问题
def classify_images(model, dataloader, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions = {}

    for inputs, _, paths in dataloader['val']:  # 假设dataloader同时返回图像、占位符（忽略的标签）和图像路径
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for pred, path in zip(preds, paths):
                class_name = class_names[pred]  # 根据预测索引获取类名
                if class_name not in predictions:
                    predictions[class_name] = [path]
                else:
                    predictions[class_name].append(path)

    print("Predictions: ", {k: v[:5] for k, v in predictions.items()})  # 打印每类的前5个预测

    return predictions

class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]  # 获取图像路径
        return (*original_tuple, path)  # 返回图像数据、标签和路径


# validation 2
# def classify_images(model, dataloader):
#     model.eval()  
#     predictions = []
#     for inputs,_ in dataloader['val']:  # 假设dataloader仅加载图像，没有标签
#         with torch.no_grad():  # 确保在推断时不计算梯度
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             predictions.extend(preds.tolist())  # 将预测结果保存到列表中
#     print("Predictions: ", predictions[:5])

#     return predictions



if __name__ == "__main__":
    # 1. 初始化模型
    model, criterion, optimizer = initialize_model()
    # 2. 获取数据加载器
    dataloaders = get_dataloaders()
    # 3. 训练模型
    train_model(model, dataloaders, criterion, optimizer, num_epochs=25)
    # 4. 验证模型
    validate_model(model, dataloaders, criterion)

    class_names = dataloaders['train'].dataset.classes

    # 5. 分类图像
    classify_images(model, dataloaders, class_names)
