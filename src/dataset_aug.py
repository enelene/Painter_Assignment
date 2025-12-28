import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

class AugmentedDataset(Dataset):
    def __init__(self, root_monet, root_photo, mode='train'):
        self.monet_files = sorted(glob.glob(os.path.join(root_monet, "*.jpg")))
        self.photo_files = sorted(glob.glob(os.path.join(root_photo, "*.jpg")))
        
        # 1. Resize to 286 (bigger than 256)
        # 2. Randomly crop to 256
        # 3. Random Flip
        self.transform = transforms.Compose([
            transforms.Resize(286, Image.BICUBIC),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # For testing, we don't crop, we just resize standardly
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.mode = mode

    def __getitem__(self, index):
        monet_idx = index % len(self.monet_files)
        monet_path = self.monet_files[monet_idx]
        photo_path = self.photo_files[index % len(self.photo_files)]
        
        monet_img = Image.open(monet_path).convert("RGB")
        photo_img = Image.open(photo_path).convert("RGB")
        
        # Apply transforms
        if self.mode == 'train':
            monet_img = self.transform(monet_img)
            photo_img = self.transform(photo_img)
        else:
            monet_img = self.test_transform(monet_img)
            photo_img = self.test_transform(photo_img)
            
        return {"monet": monet_img, "photo": photo_img}

    def __len__(self):
        return max(len(self.monet_files), len(self.photo_files))