import os
import sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
import torch
from torch.autograd import Variable
from utils.utils import *
import random
import shutil
from tqdm import tqdm
import cv2
import torchvision.transforms.functional as TF

sys.path.append(os.path.join(os.getcwd(), 'experiments'))
from dataset import Dataset_test

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="DnLogic")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
# logic
parser.add_argument('--grad-factor', type=float, default=2.)
parser.add_argument('--connections', type=str, default='random', choices=['random', 'unique'])
parser.add_argument('--seed', type=int, default=0, help='seed (default: 0)')
parser.add_argument('--channels', type=int, default=256, help='num of channels')
parser.add_argument('--epoch', type=int, default=1, help='epoch for validation')
parser.add_argument("--implementation", type=str, default='cuda', help="Using ste mode for training")
parser.add_argument("--rotation", action='store_true', help="Enable rotation")
parser.add_argument("--test_file", type=str, default='')
parser.add_argument("--test_data", type=str, default='Set12')

parser.add_argument("--tiny", type=bool, default=False, help='use tiny model or not')

opt = parser.parse_args()

def main():
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    # Load dataset
    print('Loading dataset ...\n')
    dataset_val = Dataset_test(f'data/{opt.test_data}')
    print("# of validation samples: %d\n" % int(len(dataset_val)))

    # Build model
    if opt.tiny:
        from models import LogicNetTiny as LogicNet
    else:
        from models import LogicNet    
    model = LogicNet(opt, channels=8).cuda()
    model.load_state_dict(torch.load(f"{opt.test_file}"),strict=True)

    with torch.no_grad():
        model.train(mode=False)
    
        # validate
        psnr_val = 0
        ssim_val = 0
        for k in tqdm(range(len(dataset_val))):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise
            imgn_val = torch.clamp(imgn_val, 0., 1.)
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)       
            out_val = sliding_window_processing(model, imgn_val, opt.rotation, patch_size=192, overlap=20)
            
            cur_psnr = batch_PSNR(out_val, img_val, 1.)
            cur_ssim = cal_ssim(out_val, img_val)
            psnr_val += cur_psnr
            ssim_val += cur_ssim

            print(f"PSNR_{k}: {cur_psnr}, SSIM_{k}: {cur_ssim}")
            cv2.imwrite(f'./results/gt_test_{k}.png', dataset_val[k].squeeze(0).numpy()*255.0)
            cv2.imwrite(f'./results/recon_test_{k}.png', out_val.squeeze(0).squeeze(0).cpu().numpy()*255.0)
            cv2.imwrite(f'./results/noisy_test_{k}.png', imgn_val.squeeze(0).squeeze(0).cpu().numpy()*255.0)

        psnr_val /= len(dataset_val)
        print("\nPSNR_val: %.4f" % (psnr_val))
        ssim_val /= len(dataset_val)
        print("\nSSIM_val: %.4f" % (ssim_val))

if __name__ == "__main__":
    test_module_path = "results"
    if os.path.exists(test_module_path):
        shutil.rmtree(test_module_path)
    os.makedirs(test_module_path, exist_ok=True)
    main()