import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, transform_segmentation=None):
        self.root_dir = root_dir
        self.transform = transform
        self.transform_segmentation = transform_segmentation
        self.samples = []
        
        for label in ("correct_label", "mislabelled"):
            label_dir = os.path.join(root_dir, label)
            for folder in os.listdir(label_dir):
                if folder.startswith('.'):
                    continue
                else:
                    folder_path = os.path.join(label_dir, folder)
                    rgb_image_path = os.path.join(folder_path, "original.jpeg")
                    segmentation_image_path = os.path.join(folder_path, "gd.jpeg")
                    self.samples.append((rgb_image_path, segmentation_image_path, label))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, segmentation_path, label = self.samples[idx]
        rgb_image = Image.open(rgb_path).convert("RGB")
        segmentation_image = Image.open(segmentation_path).convert("L")
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
        if self.transform_segmentation:
            segmentation_image = self.transform_segmentation(segmentation_image)
        
        label = 0 if label == "correct_label" else 1
        images = torch.cat([rgb_image, segmentation_image], dim=0)

        return images, label

def get_dataloaders():
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    #* segmentation is 1 channel, so we only need to normalize it
    transform_segmentation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #TODO： check if this is the correct normalization
    ])
    
    train_dataset = SegmentationDataset(root_dir='data_simu/train', transform=transform_rgb, transform_segmentation=transform_segmentation)
    val_dataset = SegmentationDataset(root_dir='data_simu/valid', transform=transform_rgb, transform_segmentation=transform_segmentation)

    # note: batch size is 32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return {'train': train_loader, 'val': val_loader}

if __name__ == "__main__":
    dataloaders = get_dataloaders()

    for i, (image, labels) in enumerate(dataloaders['train']):
        print("image shape:", image.shape)
        print("labels:", labels)
        if i == 0:
            break

    print("=====================================")
    for i, (image, labels) in enumerate(dataloaders['val']):
        print("image shape:", image.shape)

        print("labels:", labels)
        if i == 0:
            break
        
    print("=====================================")
    for inputs, labels in dataloaders['train']:
        print(inputs.shape)
        print(labels)
        break