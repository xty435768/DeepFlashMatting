import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True

class MaxPooledBasedErodeOperation(nn.MaxPool2d):
    def __init__(self, kernel_size, stride = None, padding = 0, dilation = 1, return_indices = False, ceil_mode = False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def forward(self, input):
        result = super().forward(-input)
        return -result

class SimpleSparseMattingWithForeground(nn.Module):
    def __init__(self, detail_feature_lr_dim=32, sparse_mask_dilation_kernel=11):
        super(SimpleSparseMattingWithForeground, self).__init__()
        self.dilation_kernel = sparse_mask_dilation_kernel
        self.dilate_op = nn.MaxPool2d(self.dilation_kernel, stride=1, padding=self.dilation_kernel // 2)
        self.erode_op_5 = MaxPooledBasedErodeOperation(kernel_size=5, stride=1, padding=2)
        self.erode_op_7 = MaxPooledBasedErodeOperation(kernel_size=7, stride=1, padding=3)
        channels = [24, 16, 12]
        self.conv_2x_to_1x = spconv.SparseSequential(
            spconv.SparseConv2d(detail_feature_lr_dim + 3 + 4, channels[0], kernel_size=3, padding=1, bias=True, indice_key='spconv0'), # 3 表示原图，4 表示alpha1+fg3
            nn.BatchNorm1d(channels[0]),
            nn.LeakyReLU(),
            spconv.SparseConv2d(channels[0], channels[1], kernel_size=3, padding=1, bias=True, indice_key='spconv1'),
            nn.BatchNorm1d(channels[1]),
            nn.LeakyReLU(),
        )
        self.alpha_head_2x = spconv.SparseSequential(
            spconv.SparseConv2d(channels[1], 4, kernel_size=1, padding=0, bias=False, indice_key='spconv2'),
        )
        self.conv_1x_to_out = spconv.SparseSequential(
            spconv.SparseConv2d(channels[1] + 3 + 4, channels[2], kernel_size=3, padding=1, bias=True, indice_key='spconv2'),
            nn.BatchNorm1d(channels[2]),
            nn.LeakyReLU(),
            spconv.SparseConv2d(channels[2], 4, kernel_size=1, padding=0, bias=False, indice_key='spconv3'),
        )

    def forward(self, alpha_fg_lr, detail_feature_lr, img_hr):
        hr_longest_length = max(img_hr.shape[-2], img_hr.shape[-1])
        assert alpha_fg_lr.shape[1] == 4 # 确保通道上是alpha1+fg3
        if hr_longest_length > 1024:
            img_hr_2x = F.interpolate(img_hr, scale_factor=0.5, mode='bilinear', align_corners=False)
        else:
            img_hr_2x = img_hr
        img_hr_1x = img_hr
        
        alpha_fg_2x = F.interpolate(alpha_fg_lr, size=img_hr_2x.shape[-2:], mode='bilinear', align_corners=False)
        detail_feature_2x = F.interpolate(detail_feature_lr, size=img_hr_2x.shape[-2:], mode='bilinear', align_corners=False)
        mask_2x = self.get_mask_from_alpha(alpha_fg_2x)
        sparse_input_2x, indices_2x = self.prepare_input_for_spconv(img_hr_2x, alpha_fg_2x, detail_feature_2x, mask_2x)
        if sparse_input_2x.shape[0] == 0:
            alpha_fg_1x = F.interpolate(alpha_fg_lr, size=img_hr_1x.shape[-2:], mode='bilinear', align_corners=False)
            alpha_fg_2x = F.interpolate(alpha_fg_lr, size=img_hr_2x.shape[-2:], mode='bilinear', align_corners=False)
            mask_1x = torch.zeros_like(alpha_fg_1x[:,:1,:,:], device=alpha_fg_1x.device)
            mask_2x = torch.zeros_like(alpha_fg_2x[:,:1,:,:], device=alpha_fg_2x.device)
            return alpha_fg_1x[:,:1,:,:], torch.clamp(alpha_fg_1x[:,1:,:,:] + img_hr_1x, 0., 1.), \
                   alpha_fg_2x[:,:1,:,:], torch.clamp(alpha_fg_2x[:,1:,:,:] + img_hr_2x, 0., 1.), \
                    mask_1x, mask_2x
        sparse_input_2x = spconv.SparseConvTensor(sparse_input_2x, indices_2x.int(), img_hr_2x.shape[2:], img_hr.shape[0])
        detail_feature_2x_out = self.conv_2x_to_1x(sparse_input_2x)
        alpha_fg_2x_out = self.alpha_head_2x(detail_feature_2x_out).dense()
        alpha_2x_out = alpha_fg_2x_out[:,:1,:,:].clamp_(0., 1.)
        fg_2x_out = alpha_fg_2x_out[:,1:,:,:]
        mask_2x_for_composition = self.erode_op_5(mask_2x)
        alpha_2x_out = mask_2x_for_composition * alpha_2x_out + (1 - mask_2x_for_composition) * alpha_fg_2x[:,:1,:,:]
        fg_2x_out = mask_2x_for_composition * fg_2x_out + (1 - mask_2x_for_composition) * alpha_fg_2x[:,1:,:,:]
        alpha_fg_1x = F.interpolate(torch.cat([alpha_2x_out, fg_2x_out], dim=1), size=img_hr_1x.shape[-2:], mode='bilinear', align_corners=False)
        fg_2x_out = torch.clamp(fg_2x_out + img_hr_2x, 0., 1.)
        detail_feature_1x = F.interpolate(detail_feature_2x_out.dense(), size=img_hr_1x.shape[-2:], mode='bilinear', align_corners=False)
        mask_1x = self.get_mask_from_alpha(alpha_fg_1x)
        sparse_input_1x, indices_1x = self.prepare_input_for_spconv(img_hr_1x, alpha_fg_1x, detail_feature_1x, mask_1x)
        if sparse_input_1x.shape[0] == 0:
            alpha_1x = F.interpolate(alpha_2x_out, size=img_hr_1x.shape[-2:], mode='bilinear', align_corners=False)
            fg_1x = F.interpolate(fg_2x_out, size=img_hr_1x.shape[-2:], mode='bilinear', align_corners=False)
            mask_1x = torch.zeros_like(alpha_fg_1x[:,:1,:,:], device=alpha_1x.device)
            return alpha_1x, fg_1x, alpha_2x_out, fg_2x_out, mask_1x, mask_2x
        sparse_input_1x = spconv.SparseConvTensor(sparse_input_1x, indices_1x.int(), img_hr_1x.shape[2:], img_hr.shape[0])
        alpha_fg_1x_out = self.conv_1x_to_out(sparse_input_1x).dense()
        alpha_1x_out = alpha_fg_1x_out[:,:1,:,:].clamp_(0., 1.)
        mask_1x_for_composition = self.erode_op_7(mask_1x)
        alpha_1x_out = mask_1x_for_composition * alpha_1x_out + (1 - mask_1x_for_composition) * alpha_fg_1x[:,:1,:,:]
        fg_1x_out = mask_1x_for_composition * alpha_fg_1x_out[:,1:,:,:] + (1 - mask_1x_for_composition) * alpha_fg_1x[:,1:,:,:]
        fg_1x_out = torch.clamp(fg_1x_out + img_hr_1x, 0., 1.)

        return alpha_1x_out, fg_1x_out, alpha_2x_out, fg_2x_out, mask_1x, mask_2x
    
    def get_mask_from_alpha(self, alpha):
        if alpha.shape[1] == 4: # alpha1+fg3
            mask = torch.logical_and(alpha[:,:1,:,:]>0.01, alpha[:,:1,:,:]<0.99).float()
        else:
            mask = torch.logical_and(alpha>0.01, alpha<0.99).float()
        mask = self.dilate_op(mask)
        return mask
    
    def prepare_input_for_spconv(self, img_batch, alpha_batch, feature_batch, mask):
        x = torch.cat([img_batch, alpha_batch, feature_batch], dim=1)
        indices = torch.where(mask.squeeze(1)>0)
        x = x.permute(0,2,3,1)
        x = x[indices]
        indices = torch.stack(indices, dim=1)

        return x, indices



if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import ToTensor, ToPILImage
    device= 'cuda'
    img = torch.randn(1, 3, 1600, 1600).to(device)
    alpha_lr = ToTensor()(Image.open('alpha.png').convert('L')).to(device).unsqueeze(0)
    detail_feature_lr = torch.randn(1, 32, 518, 518).to(device)
    fg_lr = torch.randn(1, 3, 518, 518).to(device)
    model = SimpleSparseMattingWithForeground().to(device)
    out = model(torch.cat([alpha_lr, fg_lr], dim=1), detail_feature_lr, img)
    for k in out:
        print(k.shape)
