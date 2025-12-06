from PIL import Image
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math
from transformers import Sam2Processor, Sam2Model

import torchvision
from model_new.dinov2.config_dinov2 import *
from model_new.dinov2.dino_model_wrapper import DinoVisionTransformerWrapper
from model_new.trimap_decoder import TrimapDecoder
from model_new.detail_decoder import MattingDecoder
from model_new.matting_sparse_refiner import SimpleSparseMattingWithForeground
from model_new.fcm import FCM
from model_new.cc_utils import *
import kornia

def compute_composition_alpha(pred_trimap_logit, pred_detail_pha):
    pred_trimap_argmax = torch.argmax(pred_trimap_logit, dim=1, keepdim=True)
    pred_trimap_unknown_region = (pred_trimap_argmax == 1).float()
    pred_trimap_fg_region = (pred_trimap_argmax == 2).float()
    pred_comp_pha = pred_trimap_fg_region + pred_detail_pha * pred_trimap_unknown_region
    return pred_comp_pha

def fast_color_histogram(image, color_stride=16):
    hist = torch.arange((256 // color_stride) ** 3).reshape(256 // color_stride, 256 // color_stride,
                                                            256 // color_stride)
    H, W, C = image.shape
    image = torch.tensor(image, dtype=torch.long) // color_stride
    index = image.reshape(-1, C).T

    hist = torch.bincount(hist[index[0], index[1], index[2]], minlength=(256 // color_stride) ** 3)
    return hist


def get_flash_ratio(cv2_img, flash_hist, noflash_hist, color_stride=16, type='flash'):
    image = torch.tensor(cv2_img, dtype=torch.long) // color_stride
    unit = 256 // color_stride
    image = image[:, :, 0] * unit * unit + image[:, :, 1] * unit + image[:, :, 2]
    assert type in ['flash', 'noflash']
    if type == 'flash':
        return torch.maximum((flash_hist[image] - noflash_hist[image]) / flash_hist[image],
                             torch.zeros(image.shape))
    else:
        return torch.maximum((noflash_hist[image] - flash_hist[image]) / noflash_hist[image],
                             torch.zeros(image.shape))


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2).cuda()

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def get_flash_ratio_batch(flash_img_cv2, noflash_img_cv2, gaussian_blur, zeta=0.0, norm=True, target_device='cuda'):
    assert len(flash_img_cv2.shape) == 3 and flash_img_cv2.dtype == np.uint8 and flash_img_cv2.shape[2] == 3, "flash image must be HWC uint8 format"
    assert len(noflash_img_cv2.shape) == 3 and noflash_img_cv2.dtype == np.uint8 and noflash_img_cv2.shape[2] == 3, "noflash image must be HWC uint8 format"
    flash_img_hist = fast_color_histogram(flash_img_cv2)
    noflash_img_hist = fast_color_histogram(noflash_img_cv2)
    flash_img_ratio = gaussian_blur(
        torch.maximum(
            get_flash_ratio(flash_img_cv2, flash_img_hist, noflash_img_hist, type='flash').unsqueeze(0).unsqueeze(0),
            torch.Tensor([zeta])).to(target_device))
    noflash_img_ratio = gaussian_blur(
        torch.maximum(
            get_flash_ratio(noflash_img_cv2, flash_img_hist, noflash_img_hist, type='noflash').unsqueeze(0).unsqueeze(0),
            torch.Tensor([zeta])).to(target_device))
    if norm:
        flash_img_ratio = (flash_img_ratio - flash_img_ratio.min()) / (flash_img_ratio.max() - flash_img_ratio.min())
        noflash_img_ratio = (noflash_img_ratio - noflash_img_ratio.min()) / (noflash_img_ratio.max() - noflash_img_ratio.min())
    return flash_img_ratio, noflash_img_ratio


class FlashCueExtractor(nn.Module):
    def __init__(self, device, trimap_kernel=45):
        super().__init__()
        self.device = device
        self.gaussian_blur = get_gaussian_kernel().to(device)
        self.sam_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)
        self.sam_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
        self.trimap_gen_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (trimap_kernel, trimap_kernel))
    
    def get_trimap_gt_one_hot(self, alpha):
        alpha = np.array(alpha)
        max_value, min_value = alpha.max(), alpha.min()
        fg_and_unknown = np.array(np.not_equal(alpha, min_value).astype(np.float32))
        fg = np.array(np.equal(alpha, max_value).astype(np.float32))
        dilate = cv2.dilate(fg_and_unknown, self.trimap_gen_kernel, iterations=1)
        erode = cv2.erode(fg, self.trimap_gen_kernel, iterations=1)
        trimap = erode * 2 + (dilate - erode)
        return torch.from_numpy(np.equal(trimap, np.arange(3)[:, None, None]).astype(np.float32))

    def sam2_forward(self, image, points):
        labels = [[[1] * len(points)]]  # all positive points
        inputs = self.sam_processor(images=image, input_points=[[points]], input_labels=labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        masks = self.sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        best_mask_idx = torch.argmax(outputs['iou_scores'], dim=-1)[0].cpu().item()
        mask = masks[:, best_mask_idx, :, :]
        return mask
    
    def flash_ratio_map(self, flash_img, noflash_img, zeta=0.0, norm=True, threshold=0.7, downsample_ratio=0.05):
        flash_ratio_map = get_flash_ratio_batch(flash_img, noflash_img, self.gaussian_blur, zeta=zeta, norm=norm, target_device=self.device)[0]
        flash_ratio_map[flash_ratio_map < threshold] = 0.
        flash_ratio_map = kornia.filters.median_blur(flash_ratio_map, (5,5))
        flash_ratio_cues = find_centroids(flash_ratio_map > 0.0)[0].cpu().numpy()
        flash_ratio_cues = flash_ratio_cues * (1 / downsample_ratio)
        flash_ratio_cues = flash_ratio_cues[:, ::-1] if len(flash_ratio_cues.shape) > 1 else flash_ratio_cues[::-1][np.newaxis, :]
        return flash_ratio_cues.astype(np.long).tolist()

    def forward(self, flash_img, noflash_img, downsample_ratio=0.05, threshold=0.7):
        flash_frame_lr = cv2.resize(flash_img, fx=downsample_ratio, fy=downsample_ratio, dsize=(0,0))
        noflash_frame_lr = cv2.resize(noflash_img, fx=downsample_ratio, fy=downsample_ratio, dsize=(0,0))
        flash_cues = self.flash_ratio_map(flash_frame_lr, noflash_frame_lr, downsample_ratio=downsample_ratio, threshold=threshold)
        flash_mask = self.sam2_forward(Image.fromarray(flash_img), flash_cues)
        flash_trimap = self.get_trimap_gt_one_hot(flash_mask[0].cpu().numpy()).unsqueeze(0).to(self.device)
        return flash_trimap



class MattingBaseNew(nn.Module):
    def __init__(self, backbone='dinov2_small', backbone_init_weight=None, fcm_dim=256):
        super().__init__()
        assert backbone in ['dinov2_base', 'dinov2_small']
        backbone_config = globals()[backbone]()
        self.backbone_dim = backbone_config['embed_dim']
        self.backbone_nf = DinoVisionTransformerWrapper(original_config=backbone_config, pretrained_weight=backbone_init_weight)
        self.backbone_f = DinoVisionTransformerWrapper(original_config=backbone_config, new_input_channels=6, pretrained_weight=backbone_init_weight)
        self.backbone_img_size = backbone_config['img_size']
        self.backbone_feature_size = backbone_config['img_size'] // backbone_config['patch_size']
        self.backbone_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.fcm = FCM(self.backbone_dim, encoder_dim=256, image_size=self.backbone_feature_size, nf_transformer_layer_num=2, f_transformer_layer_num=1)
        self.trimap_decoder = TrimapDecoder(feature_channel=fcm_dim, target_resolution=[64, 128], target_channels=[128, 64], output_channels=3)
        self.matting_decoder = MattingDecoder(downsample_in=[6, 32, 64, 128], upsample_in=[fcm_dim, 128, 64, 32, 32])
        
    def forward(self, flash, noflash, flash_trimap):
        flash_backbone = self.backbone_resize(flash)
        noflash_backbone = self.backbone_resize(noflash)
        flash_trimap_backbone = self.backbone_resize(flash_trimap)

        flash_context = self.backbone_f(torch.cat([self.backbone_normalize(flash_backbone), flash_trimap_backbone], dim=1))
        noflash_context = self.backbone_nf(self.backbone_normalize(noflash_backbone))
        noflash_context = self.fcm(flash_context, noflash_context, None, None)
        noflash_trimap_logit = self.trimap_decoder(noflash_context)
        pred_result = self.matting_decoder(noflash, noflash_trimap_logit, noflash_context)
        noflash_trimap_logit = F.interpolate(noflash_trimap_logit, size=pred_result['alpha'].shape[-2:], mode='bilinear', align_corners=False)
        return {**pred_result, 'trimap_logit': noflash_trimap_logit}
    
    def backbone_resize(self, tensor):
        b, c, h, w = tensor.shape
        if h != self.backbone_img_size or w != self.backbone_img_size:
            return F.interpolate(tensor, size=(self.backbone_img_size, self.backbone_img_size), mode='bilinear', align_corners=False)
        return tensor


class MattingRefineNew(MattingBaseNew):
    def __init__(self, backbone='dinov2_small', fcm_dim=256):
        super().__init__(backbone=backbone, fcm_dim=fcm_dim)
        self.sparse_upsampler = SimpleSparseMattingWithForeground()

    def forward(self, flash, noflash, flash_trimap, forward_sparse_upsampler=True):
        flash_backbone = self.backbone_resize(flash)
        noflash_backbone = self.backbone_resize(noflash)
        flash_trimap_backbone = self.backbone_resize(flash_trimap)

        flash_context = self.backbone_f(torch.cat([self.backbone_normalize(flash_backbone), flash_trimap_backbone], dim=1))
        noflash_context = self.backbone_nf(self.backbone_normalize(noflash_backbone))
        noflash_context = self.fcm(flash_context, noflash_context, None, None)
        noflash_trimap_logit = self.trimap_decoder(noflash_context)
        pred_result = self.matting_decoder(noflash_backbone, noflash_trimap_logit, noflash_context)
        noflash_trimap_logit = F.interpolate(noflash_trimap_logit, size=pred_result['alpha'].shape[-2:], mode='bilinear', align_corners=False)
        base_model_output = {**pred_result, 'trimap_logit': noflash_trimap_logit}
        
        if forward_sparse_upsampler:
            pred_composition_alpha = compute_composition_alpha(noflash_trimap_logit, pred_result['alpha'])
            upsample_output = self.sparse_upsampler(torch.cat([pred_composition_alpha, pred_result['fg_raw']], dim=1), pred_result['hidden'], noflash)
            return {**base_model_output, 'upsample_out': upsample_output}
        else:
            return base_model_output
