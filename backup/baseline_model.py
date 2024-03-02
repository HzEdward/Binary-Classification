import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
from PIL import Image
import sys

def initialize_model():
    resnet50 = models.resnet50(pretrained=True)

    for param in resnet50.parameters():
        param.requires_grad = False

    num_ftrs = resnet50.fc.in_features
    
    resnet50.fc = nn.Linear(num_ftrs, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet50.fc.parameters(), lr=0.001)

    return resnet50, criterion, optimizer

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

        return image, self.image_filenames[idx]

# 需要进行修改，因为现在数据库的读入是不同的
def get_dataloaders():
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 调整图像大小以匹配ResNet的输入尺寸
        transforms.ToTensor(), # 将图像转换成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 归一化
    ])

    train_dataset = datasets.ImageFolder(root='data_new/train', transform=transform)
    val_dataset= CustomDataset(root_dir=os.path.join('data', 'valid'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return {'train': train_loader, 'val': val_loader}

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

def classify_images(model, dataloader):  
    model.eval()  
    predictions = {}
    for inputs, filenames in dataloader['val']:  
        with torch.no_grad(): 
            outputs = model(inputs) 
            #! 在此处已经获取了每组的预测值
            # torch.max()返回两个值，第一个是最大值，第二个是最大值的索引
            _, preds = torch.max(outputs, 1) 
            preds = ["panda" if pred.item() == 1 else "man" for pred in preds]

            for filename, pred in zip(filenames, preds):  
                predictions[filename] = pred
                # predictions[filename] = dataloader['val'].dataset.classes[pred.item()]

    print("Predictions: ", predictions)

    return predictions


if __name__ == "__main__":

    model, criterion, optimizer = initialize_model()

    dataloaders = get_dataloaders()

    train_model(model, dataloaders, criterion, optimizer, num_epochs=24)

    classify_images(model, dataloaders)
