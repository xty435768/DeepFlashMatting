import logging
import torch
from model_new.dinov2.dino_model import DinoVisionTransformer
from einops import rearrange

logger = logging.getLogger("dinov2")
class DinoVisionTransformerWrapper(DinoVisionTransformer):
    def __init__(self, original_config, new_input_channels=3, pretrained_weight=None):
        self.new_input_channels = new_input_channels
        assert new_input_channels >= 3
        if new_input_channels != 3:
            original_config['in_chans'] = new_input_channels
        super().__init__(**original_config)
        if pretrained_weight is not None:
            self.load_official_pretrained_weight(pretrained_weight)
        del self.norm
    
    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            raise NotImplementedError("DinoVisionTransformerWrapper does not support list input")

        x = self.prepare_tokens_with_masks(x, masks)

        for blk_index, blk in enumerate(self.blocks):
            x = blk(x)
        
        h = int(x.shape[1] ** 0.5)

        return rearrange(x[:, self.num_register_tokens + 1 :], "b (h w) d -> b d h w", h=h)

    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret

    def load_official_pretrained_weight(self, checkpoint):
        if self.new_input_channels != 3:
            patch_embed_weight = checkpoint['patch_embed.proj.weight']
            with torch.no_grad():
                self.patch_embed.proj.weight[:, :3] = patch_embed_weight
            checkpoint['patch_embed.proj.weight'] = self.patch_embed.proj.weight
            logger.info(f"Loaded pretrained weight for patch_embed.proj.weight with shape {patch_embed_weight.shape}")
        self.load_state_dict(checkpoint)