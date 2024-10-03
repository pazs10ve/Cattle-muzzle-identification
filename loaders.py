import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2

img_width = 640
img_height = 640
batch_size = 4

transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((img_width, img_height))
])
    


def visualize_cropped_images(data_loader):
    images, labels = next(iter(data_loader))
    print(labels)
    print(images.shape, labels.shape)

    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        img = images[i].permute(1, 2, 0)  
        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.show()



class YoloCroppingDataset(Dataset):
    def __init__(self, root_dir, model, transform=None, target_transform = None):
        self.root_dir = root_dir
        self.model = model
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = [(os.path.join(root_dir, subfolder, fname), subfolder) 
                            for subfolder in os.listdir(root_dir) 
                            for fname in os.listdir(os.path.join(root_dir, subfolder))]
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        results = self.model(img, verbose = False)

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.data.cpu().numpy()  
            x1, y1, x2, y2, _, _ = boxes[0] 
            cropped_img = img.crop((x1, y1, x2, y2)) 

            if self.transform:
                cropped_img = self.transform(cropped_img)

            
            return cropped_img, torch.tensor(int(label))
        
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(int(label))


def get_loaders(root_dir : str, 
                model : YOLO,
                batch_size : int,
                train_ratio : int = 0.8, 
                val_ratio : int = 0.2, 
                transform : v2.Transform = None, 
                target_transform : v2.Transform = None):
    

    dataset = YoloCroppingDataset(root_dir=root_dir,
                                  model=model, 
                                  transform=transform, 
                                  target_transform=target_transform)
    if train_ratio:
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size

    if not train_ratio and val_ratio:
        val_size = int(val_ratio * len(dataset))
        train_size = len(dataset) - val_size


    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
    



"""
img_width = 224
img_height = 224
batch_size = 4
root_dir = 'cattle muzzle'
model = 'yolov10_fine_tuned.pt'
transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((img_width, img_height))
])


train_loader, val_loader = get_loaders(root_dir = root_dir, 
                                       model = model, 
                                       batch_size = batch_size, 
                                       transform = transform)

visualize_cropped_images(train_loader)
visualize_cropped_images(val_loader)

"""

