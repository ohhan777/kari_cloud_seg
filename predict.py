import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.plots import plot_image
import argparse
from torchvision import models
from utils.kari_cloud_dataset import open_geotiff

def predict(opt):   
    # Model initialization
    model = models.segmentation.deeplabv3_resnet101(num_classes=4)
    model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 

    # GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Load weights
    if not os.path.exists(opt.weight):
        raise FileNotFoundError(f"Model weight file not found: {opt.weight}")
    try:
        checkpoint = torch.load(opt.weight)
        model.load_state_dict(checkpoint['model'])
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # Input processing
    try:
        img = open_geotiff(opt.input)  # (H, W, C), RGB+NIR (4 bands), 14-bit image
    except Exception as e:
        print(f"Error opening input file: {e}")
        return
    
    h, w = img.shape[:2]        
    img = img.astype(np.float32).transpose(2, 0, 1) / (2**14 - 1)
    img = torch.from_numpy(img)  # (C, H, W), float32, 0-1 normalized

    # Unfold image to patches
    patch_size, patch_stride = opt.patch_size, opt.patch_stride
    pad_h = int((np.ceil(h / patch_stride) - 1) * patch_stride + patch_size - h)
    pad_w = int((np.ceil(w / patch_stride) - 1) * patch_stride + patch_size - w)
    padded_img = F.pad(img, pad=[0, pad_w, 0, pad_h])
    patches = padded_img.unfold(1, patch_size, patch_stride).unfold(2, patch_size, patch_stride)  # [C, NH, NW, H, W]

    # Create output directory
    os.makedirs(opt.output, exist_ok=True)
    
    print('Predicting...')
    pred_patches = []

    # Prediction loop with tqdm progress bar
    with torch.no_grad():
        model.eval()
        for y in tqdm(range(patches.shape[1]), desc="Processing patches"):
            imgs = patches[:, y, :, :, :].permute(1, 0, 2, 3).to(device)
            preds = model(imgs)['out']
            pred_patches.append(preds.cpu())
            
            # Periodically clear CUDA cache
            if (y + 1) % 10 == 0:
                torch.cuda.empty_cache()
    
    # Fold patches to image
    pred_patches = torch.cat(pred_patches, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    pred_patches = pred_patches.contiguous().view(1, 4, -1, patch_size * patch_size)
    pred_patches = pred_patches.permute(0, 1, 3, 2)
    pred_patches = pred_patches.contiguous().view(1, 4 * patch_size * patch_size, -1)
    ph, pw = padded_img.shape[1], padded_img.shape[2]
    out = F.fold(pred_patches, output_size=(ph, pw), kernel_size=patch_size, stride=patch_stride)
    recovery_mask = F.fold(torch.ones_like(pred_patches), output_size=(ph, pw), kernel_size=patch_size, stride=patch_stride)
    out /= recovery_mask

    pred = torch.argmax(out, dim=1).squeeze(0)[:h, :w]  # (H, W) 
    save_file = os.path.join(opt.output, os.path.basename(opt.input).replace('.tif', '_pred.png'))
    plot_image(img, pred, save_file)
    print('Done')

    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Input image path')
    parser.add_argument('--output', '-o', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--weight', '-w', default='weights/ohhan_cloud_adam_ce_best.pth', help='Weight file path')
    parser.add_argument('--patch-size', type=int, default=800, help='Patch size')
    parser.add_argument('--patch-stride', type=int, default=400, help='Patch stride')
    opt = parser.parse_args()

    predict(opt)