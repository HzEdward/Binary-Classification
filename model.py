import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import sys
import os
from PIL import Image
from dataloader import *

'''
这是模型的主入口，包含以下功能：
    1. initialize_model: 初始化模型、标准和优化器
    2. train_model: 训练模型
    3. valid_model: 验证模型
    4. resume_training: 从检查点恢复训练
    5. test_model: 在测试集上测试模型
    创建检查点(create_checkpoint): 为模型创建检查点

note: 
* the dataloader is defined in dataloader.py。在本文件中Dataset读取被注释了，实际上这是一个可以工作的例子。
* __getitem__中一定要用0和1来作为返回值,否则不符合内部操作规定

在运行程序时注意：
* 为了使用GPU，需要将模型和数据转移到GPU上，设置GPU.device("cuda:0"), 否则运行效果会比较慢


'''

# class SegmentationDataset(Dataset):
#     def __init__(self, root_dir, transform=None, transform_segmentation=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.transform_segmentation = transform_segmentation
#         self.samples = []
        
#         for label in ("correct_label", "mislabelled"):
#             label_dir = os.path.join(root_dir, label)
#             for folder in os.listdir(label_dir):
#                 if folder.startswith('.'):
#                     continue
#                 else:
#                     folder_path = os.path.join(label_dir, folder)
#                     rgb_image_path = os.path.join(folder_path, "original.jpeg")
#                     segmentation_image_path = os.path.join(folder_path, "gd.jpeg")
#                     self.samples.append((rgb_image_path, segmentation_image_path, label))
                
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         rgb_path, segmentation_path, label = self.samples[idx]
#         rgb_image = Image.open(rgb_path).convert("RGB")
#         segmentation_image = Image.open(segmentation_path).convert("L")
        
#         if self.transform:
#             rgb_image = self.transform(rgb_image)
#         if self.transform_segmentation:
#             segmentation_image = self.transform_segmentation(segmentation_image)
        
#         label = 0 if label == "correct_label" else 1
#         images = torch.cat([rgb_image, segmentation_image], dim=0)

#         return images, label

# def get_dataloaders():
#     transform_rgb = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
    
#     #* segmentation is 1 channel, so we only need to normalize it
#     transform_segmentation = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])
    
#     train_dataset = SegmentationDataset(root_dir='data_simu/train', transform=transform_rgb, transform_segmentation=transform_segmentation)
#     val_dataset = SegmentationDataset(root_dir='data_simu/valid', transform=transform_rgb, transform_segmentation=transform_segmentation)

#     # note: batch size is 32
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
#     return {'train': train_loader, 'val': val_loader}

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

def train_model(model, dataloaders, criterion, optimizer, num_epochs=30):
    print("Training started!")
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
        if epoch_acc == 1:
            break
    print("Training finished!")

# write a valid function to test the model
def valid_model(model, dataloaders, criterion):
    print("Validation started!")
    model.eval()
    running_loss = 0
    running_corrects = 0

    for inputs, labels in dataloaders['val']:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders['val'].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)
    
    print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print("Validation finished!")

'''
the function using the checkpoint to resume training
'''
def resume_training(model, optimizer, checkpoint, dataloaders, criterion, num_epochs=25):
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")
    print("Training started!")

    for epoch in range(start_epoch, num_epochs):
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
        
    print("Training finished!")

'''
the function to test the model on the testset by using the checkpoint
'''
def test_model(model, dataloader, checkpoint):
    model.load_state_dict(checkpoint['model'])
    model.eval()
    running_corrects = 0

    for inputs, labels in dataloaders['val']:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)
    print(f'Test Acc: {epoch_acc:.4f}')
    print("Testing finished!")

def create_checkpoint(model, optimizer, epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    #save checkpoint inside logs folder
    torch.save(checkpoint, 'logs/checkpoint.pth')
    return checkpoint

if __name__ == "__main__":
    # change cpu to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model, criterion, optimizer = initialize_model()
    dataloaders = get_dataloaders()
    train_model(model, dataloaders, criterion, optimizer)
    valid_model(model, dataloaders, criterion)

    

        



    


    
    

