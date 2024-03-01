import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)

        self.images = []  # 存储图像路径
        self.labels = []  # 存储标签（0或1）

        # 遍历类别文件夹
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            class_label = idx  # 类别标签为类别文件夹的索引
            # 遍历每个类别文件夹中的子文件夹
            for subdir in os.listdir(class_dir):
                subdir_path = os.path.join(class_dir, subdir)
                if os.path.isdir(subdir_path):  # 确保子路径是文件夹
                    # 读取子文件夹中的图像文件
                    for filename in os.listdir(subdir_path):
                        if filename.endswith('.jpg') or filename.endswith('.png'):  # 假设只有图像文件
                            img_path = os.path.join(subdir_path, filename)
                            self.images.append(img_path)
                            self.labels.append(class_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

from torch.utils.data import DataLoader

def get_dataloaders(root_dir, batch_size=32, shuffle=True, num_workers=4):
    # 图像预处理
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
