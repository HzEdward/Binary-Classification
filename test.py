import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys

def hidden_files_removal(folder):
    return [folder for folder in folder if not folder.startswith('.')]

class train_CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.correct_label =  os.path.join(root_dir, 'correct_label')
        self.mislabeled = os.path.join(root_dir, 'mislabelled')

        self.correct_pair_folders = os.listdir(self.correct_label)
        self.mislabeled_pair_folders = os.listdir(self.mislabeled)
        
        # 移除隐藏文件，如 .DS_Store
        self.correct_pair_folders = hidden_files_removal(self.correct_pair_folders)
        self.mislabeled_pair_folders = hidden_files_removal(self.mislabeled_pair_folders)


    def __getitem__(self, idx):
        pair_folder_name = self.pair_folders[idx]
        pair_folder_path = os.path.join(self.root_dir, pair_folder_name)
        # need to create two folder name, one is for correct label, one is for mislabelled
        
        # 读取 RGB 图像
        image_path = os.path.join(pair_folder_path, "original.jpeg")
        image = Image.open(image_path).convert("RGB")
        
        # 读取标签图像
        label_path = os.path.join(pair_folder_path, "gd.jpeg")
        label = Image.open(label_path).convert("L")  # 转换为灰度图像
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
    
class valid_CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pair_folders = os.listdir(root_dir)

        # 移除隐藏文件，如 .DS_Store
        self.pair_folders = [folder for folder in self.pair_folders if not folder.startswith('.')]

    def __len__(self):
        return len(self.pair_folders)

    def __getitem__(self, idx):
        pair_folder_name = self.pair_folders[idx]
        pair_folder_path = os.path.join(self.root_dir, pair_folder_name)
        
        # 读取 RGB 图像
        image_path = os.path.join(pair_folder_path, "original.jpeg")
        image = Image.open(image_path).convert("RGB")
        
        # 读取标签图像
        label_path = os.path.join(pair_folder_path, "gd.jpeg")
        label = Image.open(label_path).convert("L")  # 转换为灰度图像
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label    



def get_dataloaders(root_dir="data_simu"):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小以匹配ResNet的输入尺寸
        transforms.ToTensor(),  # 转换图像为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    #* 创建训练集和验证集的数据集对象, train_dataset 和 valid_dataset都是CustomDataset的实例
    train_dataset = train_CustomDataset(root_dir=os.path.join(root_dir, 'train'), transform=transform)
    # output the image pair in the correct_label and mislabelled folder
    valid_dataset = valid_CustomDataset(root_dir=os.path.join(root_dir, 'valid'), transform=transform)

    # 创建训练集和验证集的数据加载器，train_loader 和 valid_loader都是DataLoader的实例
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    return {'train': train_loader, 'valid': valid_loader}

if __name__ == '__main__':
    # 获取数据加载器
    dataloaders = get_dataloaders()
    print(dataloaders['train'].__len__())

