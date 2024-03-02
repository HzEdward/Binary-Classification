import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 数据集的根目录，应该包含correct_label和mislabelled两个子目录
        transform: 应用于图像的预处理和增强
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 加载数据
        for label in ("correct_label", "mislabelled"):
            label_dir = os.path.join(root_dir, label)
            for folder in os.listdir(label_dir):
                folder_path = os.path.join(label_dir, folder)
                rgb_image_path = os.path.join(folder_path, "rgb.png")  # 假设文件名为rgb.png
                segmentation_image_path = os.path.join(folder_path, "segmentation.png")  # 假设文件名为segmentation.png
                self.samples.append((rgb_image_path, segmentation_image_path, label))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, segmentation_path, label = self.samples[idx]
        rgb_image = Image.open(rgb_path).convert("RGB")
        segmentation_image = Image.open(segmentation_path).convert("RGB")  # 也作为RGB图像加载
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
            segmentation_image = self.transform(segmentation_image)
        
        # 标签编码
        label = 0 if label == "correct_label" else 1
        
        # 将两个图像堆叠在一起作为模型的输入
        images = torch.cat([rgb_image, segmentation_image], dim=0)
        
        return images, label
