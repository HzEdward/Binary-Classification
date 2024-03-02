import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import sys

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)

        self.images = []  # 存储图像路径
        self.labels = []  # 存储标签（0或1）

        #delete if there is a '.DS_Store' in self.classes
        if '.DS_Store' in self.classes:
            self.classes.remove('.DS_Store')

        # 遍历类别文件夹
        # for idx, class_name in enumerate(self.classes):
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            class_label = idx  # 类别标签为类别文件夹的索引, labelled是1，unlabelled是0
            print("class_dir: ", class_dir)
            print("class_label: ", class_label)
            for subdir in os.listdir(class_dir):
                subdir_path = os.path.join(class_dir, subdir)
                if os.path.isdir(subdir_path):  # 确保子路径是文件夹
                    print("subdir_path: ", subdir_path) 
                    for filename in os.listdir(subdir_path):
                        if filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.jpg'): 
                            img_path = os.path.join(subdir_path, filename)
                            print("img_path: ", img_path)
                            self.images.append(img_path)
                            print("self.images: ", self.images)
                            self.labels.append(class_label)
                            print("self.labels: ", self.labels)
                            print("one loop done")
                            sys.exit()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# 如果这里多了个函数
def get_dataloaders(root_dir, batch_size=32, shuffle=True, num_workers=4):
    # 图像预处理设置
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小以匹配ResNet的输入尺寸
        transforms.ToTensor(),  # 将图像转换成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    # 创建训练数据集
    train_dataset = CustomDataset(root_dir=os.path.join(root_dir, 'train'), transform=transform)
    
    # 创建验证数据集
    val_dataset = CustomDataset(root_dir=os.path.join(root_dir, 'valid'), transform=transform)

    # 创建训练数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # 创建验证数据加载器
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {'train': train_loader, 'val': val_loader}

if __name__ == '__main__':
    dataloaders = get_dataloaders(root_dir='data_simu')



