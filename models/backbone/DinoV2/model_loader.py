import torch
from torch import nn
import numpy as np

class DINOv2Wrapper(nn.Module):
    """
    Minimal wrapper to mimic CLIP.encode_image interface
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def encode_image(self, x, features_list):
        blocks = [i - 1 for i in features_list]  # 1-index → 0-index
        m = self.model
        B = x.shape[0]

        x = m.patch_embed(x)
        cls = m.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + m.pos_embed
        #x = m.pos_drop(x)

        patch_tokens = []

        for i, blk in enumerate(m.blocks):
            x = blk(x)
            if i in blocks:
                patch_tokens.append(x[:, 1:, :].contiguous())

        x = m.norm(x)
        global_feat = x[:, 0, :]

        return global_feat, None, patch_tokens
    

    def crop_to_36x36(self, tokens):
        """
        Robust cropping for DINOv2 tokens.
        Handles L = 1369, 1368, or any near-square length.
        """
        B, L, C = tokens.shape

        H = int(np.floor(np.sqrt(L)))   # e.g. 37 for 1369 or 1368
        usable = H * H                  # largest square <= L

        # drop extra tokens if any (e.g. 1368 -> use first 1369? no, use 36x36 later)
        tokens = tokens[:, :usable, :]

        # reshape to grid
        tokens = tokens.view(B, H, H, C)

        # crop to DictAS expected size
        tokens = tokens[:, :36, :36, :]

        return tokens.reshape(B, 36 * 36, C)
