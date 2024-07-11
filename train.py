import os
import argparse
import torch
from utils.kari_cloud_dataset import KariCloudDataset
from utils.metrics import ConfusionMatrix
from utils.loss import ce_loss
import torch.optim as optim
import time
import wandb
from pathlib import Path
from torchvision import models
from utils.plots import plot_image

def train(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    name = opt.name

    # wandb settings
    # wandb.init(id=opt.name, resume='allow', project=Path(__file__).parent.stem)
    # wandb.config.update(opt)

    # Train dataset
    train_dataset = KariCloudDataset('./data/kari-cloud-small', train=True, 
                                     patch_size=800, patch_stride=400, patch_root='./data/patches')
    # Train dataloader
    num_workers = min([os.cpu_count(), batch_size, 16])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)

    # Validation dataset
    val_dataset = KariCloudDataset('./data/kari-cloud-small', train=False,
                                   patch_size=800, patch_stride=400, patch_root='./data/patches')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    
    # Network model
    num_classes = 4 
    model = models.segmentation.deeplabv3_resnet101(num_classes=num_classes)
    model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  
    
    # GPU-support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:   # multi-GPU
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
      
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # loading a weight file (if exists)
    weight_file = Path('weights')/(name + '.pth')
    best_accuracy = 0.0
    start_epoch, end_epoch = (0, epochs)
    if os.path.exists(weight_file):
        checkpoint = torch.load(weight_file)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print('resumed from epoch %d' % start_epoch)

    (img, label_img, img_file) = train_dataset[0]  # for debugging
    
    confusion_matrix = ConfusionMatrix(num_classes)

    # training/validation
    for epoch in range(start_epoch, end_epoch):
        print('epoch: %d/%d' % (epoch+1, end_epoch))
        t0 = time.time()
        # training
        epoch_loss = train_one_epoch(train_dataloader, model, optimizer, device)
        t1 = time.time()
        print('loss=%.4f (took %.2f sec)' % (epoch_loss, t1-t0))
        lr_scheduler.step(epoch_loss)
        # validation
        val_epoch_loss = val_one_epoch(val_dataloader, model, confusion_matrix, device)
        val_epoch_iou = confusion_matrix.get_iou()
        val_epoch_mean_iou = confusion_matrix.get_mean_iou()
        val_epoch_pix_accuracy = confusion_matrix.get_pix_acc()
        
        print('[validation] loss=%.4f, mean iou=%.4f, pixel accuracy=%.4f' % 
              (val_epoch_loss, val_epoch_mean_iou, val_epoch_pix_accuracy))
        print('class IoU: [' + ', '.join([('%.4f' % (x)) for x in val_epoch_iou]) + ']')
        
        # saving the best status into a weight file
        if val_epoch_pix_accuracy > best_accuracy:
             best_weight_file = Path('weights')/(name + '_best.pth')
             best_accuracy = val_epoch_pix_accuracy
             state = {'model': model.state_dict(), 'epoch': epoch, 'best_accuracy': best_accuracy}
             torch.save(state, best_weight_file)
             print('best accuracy=>saved\n')

        # saving the current status into a weight file
        state = {'model': model.state_dict(), 'epoch': epoch, 'best_accuracy': best_accuracy}
        torch.save(state, weight_file)
        # wandb logging
        # wandb.log({'train_loss': epoch_loss, 'val_loss': val_epoch_loss, 'val_mean_iou': val_epoch_mean_iou, 'val_accuracy': val_epoch_pix_accuracy})
        
def train_one_epoch(train_dataloader, model, optimizer, device):
    model.train()
    losses = [] 
    for i, (imgs, targets, _) in enumerate(train_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)['out']     # forward, preds: (B, C, H, W)
        loss = ce_loss(preds, targets) # calculates the iteration loss  
        optimizer.zero_grad()   # zeros the parameter gradients
        loss.backward()         # backward
        optimizer.step()        # update weights
        print('\t iteration: %d/%d, loss=%.4f' % (i, len(train_dataloader)-1, loss))    
        losses.append(loss.item())
    return torch.tensor(losses).mean().item()


def val_one_epoch(val_dataloader, model, confusion_matrix, device):
    model.eval()
    losses = []
    total = 0
    for i, (imgs, targets, img_file) in enumerate(val_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            preds = model(imgs)['out']   # forward, preds: (B, C, H, W)
            loss = ce_loss(preds, targets)
            losses.append(loss.item())
            preds = torch.argmax(preds, dim=1)  # (B, H, W)
            confusion_matrix.process_batch(preds, targets)
            total += preds.size(0)
            # sample images (overwrite the previous ones)
            if i == 0:
                for j in range(min(5, preds.size(0))):  
                    save_file = os.path.join('outputs', 'val_%d.png' % (j))
                    plot_image(imgs[j], preds[j], save_file)
                
    avg_loss = torch.tensor(losses).mean().item()
    
    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=250, help='target epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--name', default='ohhan_cloud_adam_ce', help='name for the run')

    opt = parser.parse_args()

    train(opt)
