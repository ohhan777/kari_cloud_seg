import sys
import os
import cv2
import numpy as np
import torch
import rasterio

def make_color_label(label):
    class_colors = [        # RGB color for each class
        [0, 0, 0],          # 0: clear sky (background)
        [255, 0, 0],        # 1: thick cloud
        [0, 255, 0],        # 2: thin cloud
        [255, 255, 0]       # 3: cloud shadow
    ]
    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)  # (H, W, 3) shape
    for i, class_color in enumerate(class_colors):
        color_label[label == i] = class_color
    return color_label


def plot_image(img, label=None, save_file='image.png', alpha=0.3):
    # if img is a tensor, convert to a numpy array
    if torch.is_tensor(img):  # input: (4, H, W) tensor, RGB+NIR, 0-1 normalized
        img = img[:3,:,:].mul(255.0).clamp(0.0, 255.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    else:
        # (H, W, 4), 14-bit numpy image (RGB) -> (H, W, 3), 8-bit numpy image (RGB)
        img = (img[:,:,:3]/(2**14-1)*255.0).astype(np.uint8)  
    
    img = histogram_stretch(img)  # histogram stretch for better visualization
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB to BGR for cv2

    if label is not None:
        # if label_img is tensor, convert to cv2 image
        if torch.is_tensor(label):
            label = label.cpu().numpy().astype(np.uint8)
            
        color_label = make_color_label(label)
        color_label = cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR)  # RGB to BGR for cv2
        
        img = cv2.addWeighted(img, 1.0, color_label, alpha, 0) # overlays image and label

    # save image
    cv2.imwrite(save_file, img)


def open_geotiff(img_file):
    with rasterio.open(img_file) as f:
        img = f.read()  # (C, H, W)
        img = img.transpose(1,2,0).astype(np.float32)  # (H, W, C), RGB+NIR (4 bands)
    return img  

def histogram_stretch(img, lower_percentile=2, upper_percentile=98):
    stretched_img = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[2]):
        band = img[:, :, i]
        non_zero_pixels = band[band > 0]
        if non_zero_pixels.size > 0:
            min_val = np.percentile(non_zero_pixels, lower_percentile)
            max_val = np.percentile(non_zero_pixels, upper_percentile)
            if max_val > min_val:
                stretched_band = np.clip((band - min_val) * 255 / (max_val - min_val), 0, 255)
                stretched_img[:, :, i] = stretched_band.astype(np.uint8)
        else:
            stretched_img[:, :, i] = band
    return stretched_img

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    img_file = 'data/kari-cloud-small/val/images/CLD00057_MS4_K3_NIA0057.tif'
    label_file = img_file.replace('images', 'labels').replace('.tif', '.png')

    img = open_geotiff(img_file)  # (H, W, C) numpy array, RGB+NIR order, 14-bit image
    
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    
    color_label = make_color_label(label)
    color_label = cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR)
    cv2.imwrite('outputs/label1.png', color_label)
    
    plot_image(img, save_file='outputs/image1.png')
    plot_image(img, label, save_file='outputs/image+label1.png')