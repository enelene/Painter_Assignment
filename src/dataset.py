import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class UnpairedDataset(Dataset):
    def __init__(self, root_monet, root_photo, mode='train'):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.monet_files = sorted(glob.glob(os.path.join(root_monet, "*.jpg")))
        self.photo_files = sorted(glob.glob(os.path.join(root_photo, "*.jpg")))
        
        self.mode = mode

    def __getitem__(self, index):
        monet_idx = index % len(self.monet_files)
        
        monet_path = self.monet_files[monet_idx]
        photo_path = self.photo_files[index % len(self.photo_files)]
        
        monet_img = Image.open(monet_path).convert("RGB")
        photo_img = Image.open(photo_path).convert("RGB")
        
        return {
            "monet": self.transform(monet_img), 
            "photo": self.transform(photo_img)
        }

    def __len__(self):
        return max(len(self.monet_files), len(self.photo_files))