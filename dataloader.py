import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# 定义数据集
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        for label in ("correct_label", "mislabelled"):
            label_dir = os.path.join(root_dir, label)
            # 已经到了最底层的层级，即correct_label和mislabelled文件夹中
            for folder in os.listdir(label_dir):
                folder_path = os.path.join(label_dir, folder)
                rgb_image_path = os.path.join(folder_path, "original.jpeg")  # 文件名必须是original.jpeg
                segmentation_image_path = os.path.join(folder_path, "gd.jpeg")  # 文件名必须是gd.jpeg
                self.samples.append((rgb_image_path, segmentation_image_path, label))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
    #TODO: __getitem__方法的作用是包括，通过读取图片路径和标签，对图片进行transform，返回图片和标签
        # 读取图片，每次读取图片会读取三个路径，分别是rgb图片路径、分割图片路径和标签
        rgb_path, segmentation_path, label = self.samples[idx]
        # rgb图会被转换成RGB格式
        rgb_image = Image.open(rgb_path).convert("RGB")
        # 分割图会被转换成灰度图
        segmentation_image = Image.open(segmentation_path).convert("L")
        
        # 如果有transform，就对图片进行transform
        if self.transform:
            rgb_image = self.transform(rgb_image)
            segmentation_image = self.transform(segmentation_image)
        
        if label =="correct_label":
            label = 0
        elif label == "mislabelled":
            label = 1

        # 是否需要这一步呢
        images = torch.cat([rgb_image, segmentation_image], dim=0)
        
        return images, label

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = SegmentationDataset(root_dir='data_simu/train', transform=transform)
    val_loader = SegmentationDataset(root_dir='data_simu/valid', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=32, shuffle=False)
    print("DataLoader is ready!")
    
    return {'train': train_loader, 'val': val_loader}


if __name__ == "__main__":
    dataloaders = get_dataloaders()

    for i, (inputs, labels) in enumerate(dataloaders['train']):
        print(inputs.shape, labels)
        if i == 0:
            break


