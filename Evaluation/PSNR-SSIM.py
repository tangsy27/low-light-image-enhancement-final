import os, glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from math import exp

def collect_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)

def load_img_01(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0  # [0,1]
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # 1x3xHxW
    return t

# -------- EvLight PSNR --------
def psnr_evlight(pred, gt, eps=1e-10):
    mse = torch.mean((pred - gt) ** 2) + eps
    return (-10.0) * torch.log10(mse)

# -------- EvLight SSIM (same as their code) --------
def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                          for x in range(window_size)], dtype=torch.float32)
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D = gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_evlight(img1, img2, window_size=11):
    # img1,img2: 1xCxHxW, range [0,1]
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def eval_folder(pred_dir, gt_dir):
    gt_files = collect_images(gt_dir)
    gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_files}

    pred_files = collect_images(pred_dir)

    psnrs, ssims = [], []
    missing = 0

    for p in pred_files:
        key = os.path.splitext(os.path.basename(p))[0]
        if key not in gt_map:
            missing += 1
            continue

        pred = load_img_01(p)
        gt = load_img_01(gt_map[key])

        if pred.shape != gt.shape:
            pred = F.interpolate(pred, size=(gt.shape[2], gt.shape[3]), mode="bilinear", align_corners=False)

        psnrs.append(psnr_evlight(pred, gt).item())
        ssims.append(ssim_evlight(pred, gt).item())

    print(f"Pred: {pred_dir}")
    print(f"GT  : {gt_dir}")
    print(f"Matched: {len(psnrs)} images, Missing GT for {missing} preds")
    print(f"Mean PSNR (EvLight-style, MAX=1): {sum(psnrs)/len(psnrs):.4f}")
    print(f"Mean SSIM (EvLight-style, RGB, win=11): {sum(ssims)/len(ssims):.4f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    args = ap.parse_args()
    eval_folder(args.pred_dir, args.gt_dir)
