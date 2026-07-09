import math
import torch
import torch.nn as nn
import numpy as np
# from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from scipy import signal
import cv2
import torchvision.transforms.functional as TF

def rotation_ensemble(noisy_patch, model):
    outputs = []

    for angle in [0, 90, 180, 270]:
        rotated_input = TF.rotate(noisy_patch, angle)
        rotated_output = model(rotated_input)
        restored_output = TF.rotate(rotated_output, -angle)
        outputs.append(restored_output)

    out_patch = torch.stack(outputs, dim=0).mean(dim=0)
    return out_patch

def sliding_window_processing(model, imgn, rotation=False, patch_size=512, overlap=2):
    stride = patch_size - overlap 
    _, _, h, w = imgn.size()
    output = torch.zeros_like(imgn)  
    count_map = torch.zeros_like(imgn)  
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch_h = min(patch_size, h - y) 
            patch_w = min(patch_size, w - x)

            noisy_patch  = torch.zeros((1, 1, patch_size, patch_size)).cuda()
            
            noisy_patch[:, :, :patch_h, :patch_w] = imgn[:, :, y:y+patch_h, x:x+patch_w]
            
            with torch.no_grad():
                noisy_patch = torch.clamp(noisy_patch, 0, 1)
                if rotation:
                    out_patch = noisy_patch + model.alpha*rotation_ensemble(noisy_patch, model)
                else:
                    out_patch = noisy_patch + model.alpha*model(noisy_patch)
                out_patch = torch.clamp(out_patch, 0., 1.)
            
            out_patch = out_patch[:, :, :patch_h, :patch_w]

            output[:, :, y:y+patch_h, x:x+patch_w] += out_patch
            count_map[:, :, y:y+patch_h, x:x+patch_w] += 1

    output /= count_map
    return output

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def cal_ssim(img1, img2):
    img1 = 255*img1.squeeze().squeeze().cpu().numpy()
    img2 = 255*img2.squeeze().squeeze().cpu().numpy()
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T

    M, N = np.shape(img1)

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))
