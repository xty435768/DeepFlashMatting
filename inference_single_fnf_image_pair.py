import argparse
import os
from model_new.model_multidecoder_transformer import FlashCueExtractor
import torch
from model_new.model_multidecoder_transformer import FlashCueExtractor
import mediapy
from PIL import Image
import cv2
import numpy as np
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from model_new.model_multidecoder_transformer import MattingRefineNew

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash-img-path', type=str, default='demo_img_in/f.jpg')
    parser.add_argument('--noflash-img-path', type=str, default='demo_img_in/nf.jpg')
    parser.add_argument('--save-path', type=str, default='demo_img_out/out.jpg')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/best_model.pth')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    device = 'cuda'
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

    fce = FlashCueExtractor(device=device)
    flash_frame = np.array(Image.open(args.flash_img_path).convert("RGB"))
    flash_frame_tensor = ToTensor()(flash_frame).unsqueeze(0).cuda()
    noflash_frame = np.array(Image.open(args.noflash_img_path).convert("RGB"))
    flash_trimap = fce(flash_frame, noflash_frame, downsample_ratio=0.05, threshold=0.7)
    noflash_frame_tensor = ToTensor()(noflash_frame).unsqueeze(0).cuda()
    green_screen_background = torch.tensor([116, 237, 160]).view(1, 3, 1, 1).cuda() / 255.0

    model = MattingRefineNew().cuda()
    weight = torch.load(args.checkpoint_path)['state_dict']
    model.load_state_dict(weight)
    model.eval()

    with torch.no_grad():
        output = model(flash_frame_tensor, noflash_frame_tensor, flash_trimap)
        alpha_hr, fg_hr = output['upsample_out'][:2]
        com = alpha_hr * fg_hr + (1 - alpha_hr) * green_screen_background
        vis = torch.cat([noflash_frame_tensor, com], dim=3)
        save_image(vis, args.save_path)
    print(f"Saved result to {args.save_path}")
    