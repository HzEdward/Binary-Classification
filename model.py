import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import sys
import os
from PIL import Image

class DualInputResNet(nn.Module):
    def __init__(self):
        super(DualInputResNet, self).__init__()
        # 加载预训练的ResNet-50模型
        self.resnet = models.resnet50(pretrained=True)
        
        # 冻结模型的所有参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 获取ResNet-50的全连接层输入特征数
        num_ftrs = self.resnet.fc.in_features
        
        # 定义新的全连接层，输出维度为2，即两个类别
        self.fc = nn.Linear(num_ftrs * 2, 2)  # * 2 是因为有两个输入图像
        
    def forward(self, input1, input2):
        # 分别传递两个输入图像到ResNet-50模型
        output1 = self.resnet(input1)
        output2 = self.resnet(input2)
        
        # 将两个输出特征连接起来
        combined_output = torch.cat((output1, output2), dim=1)
        
        # 使用新的全连接层进行分类
        output = self.fc(combined_output)
        
        return output
    
class ModifiedResNet50(nn.Module):
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        original_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(original_model.children())[1:-2]
        )
        
        self.classifier = nn.Linear(2048, 2)
        
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_filenames[idx])

        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root='data_simu/train', transform=transform)
    val_dataset= CustomDataset(root_dir=os.path.join('data', 'valid'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return {'train': train_loader, 'val': val_loader}

def initialize_model():
    model = DualInputResNet()
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
            
            # 分别取出两个输入图像
            input1 = inputs[:, :3, :, :]  # 第一个图像
            input2 = inputs[:, 3:, :, :]  # 第二个图像
            
            outputs = model(input1, input2)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        
        print(f'Epoch {epoch}/{num_epochs - 1} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

def classify_images(model, dataloader):  
    model.eval()  
    predictions = {}
    for inputs in dataloader['val']:  
        with torch.no_grad(): 
            # 分别取出两个输入图像
            input1 = inputs[:, :3, :, :]  # 第一个图像
            input2 = inputs[:, 3:, :, :]  # 第二个图像
            
            outputs = model(input1, input2) 
            _, preds = torch.max(outputs, 1)
            
            preds = ["panda" if pred.item() == 1 else "man" for pred in preds]

            for filename, pred in zip(filename, preds):  
                predictions[filename] = pred

    print("Predictions: ", predictions)

    return predictions

if __name__ == "__main__":
    model, criterion, optimizer = initialize_model()

    dataloaders = get_dataloaders()

    train_model(model, dataloaders, criterion, optimizer, num_epochs=24)

    classify_images(model, dataloaders)
