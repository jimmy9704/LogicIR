import os
import sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import prepare_data, Dataset
from utils.utils import *

import torchvision.utils as tutils
from difflogic import KeepMSBPlanes

from tqdm import tqdm
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="DnLogic")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=12, help="Training batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=40000, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
# logic
parser.add_argument('--grad-factor', type=float, default=2.)
parser.add_argument('--connections', type=str, default='random', choices=['random', 'unique'])
parser.add_argument('--seed', type=int, default=0, help='seed (default: 0)')
parser.add_argument('--channels', type=int, default=256, help='num of channels')
parser.add_argument("--implementation", type=str, default='cuda', help="implementation mode recommend cuda")
parser.add_argument('--log_every', type=int, default=400, help='how many steps to wait before logging training status')
parser.add_argument("--load_from", type=str, default='None')

parser.add_argument("--tiny", type=bool, default=False, help='use tiny model or not')

opt = parser.parse_args()

def main():
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    if opt.tiny:
        from models import LogicNetTiny as LogicNet
    else:
        from models import LogicNet
    model = LogicNet(opt, channels=8).cuda()
    if opt.load_from != 'None':
        model.load_state_dict(torch.load(f"{opt.load_from}"))

    print(model)

    criterion = nn.MSELoss(size_average=False)
    criterion.cuda()
    keep_msb = KeepMSBPlanes(num_msb=4)
    keep_msb.cuda()
    # Optimizer

    optimizer = optim.Adam(
        [param for name, param in model.named_parameters() if name != 'alpha'], 
        lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)

    step = 0
    best_psnr = 0 
    for epoch in range(opt.epochs):
        torch.cuda.empty_cache()
        if step < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in tqdm(enumerate(loader_train, 0), total=len(loader_train), desc="Training"):
            # training step
            model.train(mode=True)
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            
            imgn_train = img_train + noise
            imgn_train = torch.clamp(imgn_train, 0., 1.)

            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())

            out_train = imgn_train + model(imgn_train)

            loss = criterion(out_train, img_train) / (imgn_train.size()[0]*2)
            loss_msb = criterion(out_train, keep_msb(img_train)) / (imgn_train.size()[0]*2)

            loss = loss + 0.005 * loss_msb
            
            loss.backward()
            optimizer.step()
            if step % (opt.log_every) == 0:
                with torch.no_grad():
                    out_train = torch.clamp(out_train, 0., 1.)
                    psnr_train = batch_PSNR(out_train, img_train, 1.)
                writer.add_scalar('PSNR on training data with train mode', psnr_train, step)
                
                with torch.no_grad():
                    model.train(mode=False)
                    out_train = imgn_train + model(imgn_train)
                    out_train = torch.clamp(out_train, 0., 1.)
                    psnr_train = batch_PSNR(out_train, img_train, 1.)
                writer.add_scalar('loss', loss.item(), step)
                print('loss: ', loss.item())
                writer.add_scalar('PSNR on training data', psnr_train, step)
                print('PSNR on training data', psnr_train)

                Img = tutils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
                Imgn = tutils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
                Irecon = tutils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
                tutils.save_image(Img, os.path.join(opt.outf,'gt.png'))      
                tutils.save_image(Imgn, os.path.join(opt.outf,'noisy.png'))    
                tutils.save_image(Irecon, os.path.join(opt.outf,'recon.png'))

            if step % (opt.log_every*10) == 0:
                with torch.no_grad():
                    model.train(mode=False)
                    # validate
                    psnr_val = 0
                    for k in range(len(dataset_val)):
                        img_val = torch.unsqueeze(dataset_val[k], 0)
                        noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                        imgn_val = img_val + noise
                        imgn_val = torch.clamp(imgn_val, 0., 1.)

                        img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)       
                        out_val = sliding_window_processing(model, imgn_val, patch_size=192, overlap=20)
                        psnr_val += batch_PSNR(out_val, img_val, 1.)

                        Img = tutils.make_grid(img_val.data, nrow=1, normalize=True, scale_each=True)
                        Imgn = tutils.make_grid(imgn_val.data, nrow=1, normalize=True, scale_each=True)
                        Irecon = tutils.make_grid(out_val.data, nrow=1, normalize=True, scale_each=True)
                        tutils.save_image(Img, os.path.join(opt.outf,'gt_val.png'))    
                        tutils.save_image(Imgn, os.path.join(opt.outf,'noisy_val.png'))   
                        tutils.save_image(Irecon, os.path.join(opt.outf,'recon_val.png'))

                    psnr_val /= len(dataset_val)
                    print("\n[step %d] PSNR_val: %.4f" % (step, psnr_val))
                    writer.add_scalar('PSNR on validation data', psnr_val, step)
                    # save model
                    torch.save(model.state_dict(), os.path.join(opt.outf, f'net_{step}.pth'))
                    if psnr_val > best_psnr:
                        best_psnr = psnr_val
                        print(f"New best PSNR_val: {best_psnr:.4f}. Saving model...")
                        torch.save(model.state_dict(), os.path.join(opt.outf, f'net_best.pth'))
                    
            step += 1

if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(data_path='data', patch_size=64, stride=10, aug_times=1)
    main()
