import os
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
import glob

class KariCloudDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=False, patch_size=800, patch_stride=400, patch_root='./data/patches'):
        self.root = Path(root)
        self.train = train
        task = 'train' if train else 'val'
        self.img_dir = self.root/f'{task}'/'images'
        self.img_files = sorted(self.img_dir.glob('*.tif'))
        self.transforms = get_transforms(train)

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # check patch directories
        patch_dir = os.path.join(patch_root, task)
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir, exist_ok=True)

        # check if patches already exist. if not, create patches
        self.img_patch_file_list = []
        for img_file in self.img_files:
            img_patch_pattern = os.path.join(patch_dir, 'img_%g_%g_%s_*.pt' % (patch_size, patch_stride, os.path.basename(img_file).rsplit('.tif')[0]))
            label_patch_pattern = os.path.join(patch_dir, 'label_%g_%g_%s_*.pt' % (patch_size, patch_stride, os.path.basename(img_file).rsplit('.tif')[0]))
            img_patch_files = glob.glob(img_patch_pattern)
            label_patch_files = glob.glob(label_patch_pattern)
            if len(img_patch_files) == 0 or len(label_patch_files) == 0 or len(img_patch_files) != len(label_patch_files):
                img_patch_files, _ = self.create_img_label_patches(img_file, patch_size, patch_stride, patch_dir)
            self.img_patch_file_list += img_patch_files

    def __getitem__(self, idx):
        img_patch_file= self.img_patch_file_list[idx]
        label_patch_file = img_patch_file.replace('img_', 'label_')
        img = torch.load(img_patch_file)   # 0-1 normalization
        label = torch.load(label_patch_file)
        img, label = self.transforms(img, label)
        return img, label, img_patch_file

    def __len__(self):
        return len(self.img_patch_file_list)
    
    def create_img_label_patches(self, img_file, patch_size=256, patch_stride=256, out_dir='./data/patches'):   
        img_patch_files, label_patch_files = [], []
        print('creating patches for %s...' % img_file, end=' ')
        # image patches
        if not os.path.exists(img_file):
                raise Exception('%s not found' % img_file)
        img = open_geotiff(img_file)  # (H, W, C), RGB+NIR (4 bands), 14-bit image
        h, w = img.shape[:2]  

        # numpy arrays to tensors
        img = torch.from_numpy(img.transpose(2, 0, 1)).to(dtype=torch.int16)  # (H, W, C) to (C, H, W)

        pad_h = int((np.ceil(h / patch_stride) - 1) * patch_stride + patch_size - h)
        pad_w = int((np.ceil(w / patch_stride) - 1) * patch_stride + patch_size - w)
        padded_img = F.pad(img, pad=[0, pad_w, 0, pad_h])
        patches = padded_img.unfold(1, patch_size, patch_stride).unfold(2, patch_size, patch_stride) # [C, NH, NW, patch_size, patch_size]
        
        for y in range(patches.shape[1]):
            for x in range(patches.shape[2]):
                img_patch_file = os.path.join(out_dir, 'img_%g_%g_%s_%g_%g.pt' %
                                                (patch_size, patch_stride,
                                                os.path.basename(img_file).rsplit('.tif')[0], y, x))
                torch.save(patches[:, y, x, :, :].contiguous(), img_patch_file)  # (C, patch_size, patch_size)
                img_patch_files.append(img_patch_file)
        
        print('success')

        # label patches
        label_file = str(img_file).replace('images', 'labels').replace('.tif', '.png')
        print('creating patches for %s...' % label_file, end=' ')
        if not os.path.exists(label_file):
            raise Exception('%s not found' % label_file)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)  
        label = torch.from_numpy(label).to(dtype=torch.uint8)  # tensor of (H, W) belongs to {0, 1, 2, 3}
    
        pad_h = int((np.ceil(h / patch_stride) - 1) * patch_stride + patch_size - h)
        pad_w = int((np.ceil(w / patch_stride) - 1) * patch_stride + patch_size - w)
        padded_label = F.pad(label, pad=[0, pad_w, 0, pad_h])
        patches = padded_label.unfold(0, patch_size, patch_stride).unfold(1, patch_size, patch_stride) # [NH, NW, patch_size, patch_size]

        for y in range(patches.shape[0]):
            for x in range(patches.shape[1]):
                label_patch_file = os.path.join(out_dir, 'label_%g_%g_%s_%g_%g.pt' % 
                                                (patch_size, patch_stride, os.path.basename(label_file).rsplit('.png')[0], y, x))
                torch.save(patches[y, x, :, :].contiguous(), label_patch_file)
                label_patch_files.append(label_patch_file)    # (patch_size, patch_size)
        print('success')
        return img_patch_files, label_patch_files
    
class ImageAug:
    def __init__(self, train):
        if train:
            self.aug = A.Compose([A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.5),
                                  A.ShiftScaleRotate(p=0.5),
                                  A.RandomBrightnessContrast(p=0.3),
                                  ToTensorV2()])
        else:
            self.aug = ToTensorV2()
        
    def __call__(self, img, label):
        img = img.permute(1, 2, 0).numpy().astype(dtype=np.float32) / (2 ** 14 - 1)  # KOMPSAT-3/3A (14-bit)
        label = label.numpy()
        transformed = self.aug(image=img, mask=label)
        return transformed['image'], transformed['mask']

def get_transforms(train):
    transforms = ImageAug(train)
    return transforms


def open_geotiff(img_file):
    with rasterio.open(img_file) as f:
        img = f.read()  # (C, H, W)
        img = img.transpose(1,2,0).astype(np.float32)  # (H, W, C), RGB+NIR (4 bands), 14-bit image
    return img
    


