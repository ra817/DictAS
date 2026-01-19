import os
import cv2
import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from models.backbone.clip.model_CLIP import Load_CLIP
from models.utils import norm_patch, setup_seed, normalize


#TRANSFORMS
def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform_test(n_px):
    return Compose([
        Resize((n_px, n_px)),
        CenterCrop((n_px, n_px)),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        ),
    ])


#LOAD IMAGES
def load_images_from_dir(dir_path):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    return sorted([
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(valid_exts)
    ])

def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_path, exist_ok=True)


    # Load CLIP backbone ONLY
    # -----------------------------
    model_CLIP, _, _ = Load_CLIP(
        args.image_size,
        args.pretrained_path,
        device=device
    )
    model_CLIP.eval().to(device)

    # -----------------------------
    # Load images
    # -----------------------------
    transform_image = _transform_test(args.image_size)
    image_paths = load_images_from_dir(args.sequence_dir)

    assert len(image_paths) > 0, "No images found"

    for idx, img_path in enumerate(image_paths):
        print(f"\n[{idx+1}/{len(image_paths)}] {os.path.basename(img_path)}")

        # Load image
        img_pil = Image.open(img_path)
        img_tensor = transform_image(img_pil).unsqueeze(0).to(device)

        # -----------------------------
        # Extract CLIP features
        # -----------------------------
        with torch.no_grad():
            _, _, patch_tokens = model_CLIP.encode_image(
                img_tensor,
                args.features_list
            )

        # -----------------------------
        # Inspect features
        # -----------------------------
        for layer_id, feat in zip(args.features_list, patch_tokens):
            # feat shape: [B, N, C]
            feat_np = feat.squeeze(0).cpu().numpy()

            print(f"Layer {layer_id}")
            print(f"  Shape      : {feat_np.shape}")
            print(f"  Min value  : {feat_np.min():.6f}")
            print(f"  Max value  : {feat_np.max():.6f}")
            print(f"  Mean value : {feat_np.mean():.6f}")
            print(f"  Std value  : {feat_np.std():.6f}")

            # Optional: save features
            save_name = f"{os.path.basename(img_path).split('.')[0]}_layer{layer_id}.npy"
            np.save(os.path.join(args.save_path, save_name), feat_np)

        print("-" * 60)



if __name__ == "__main__":

    parser = argparse.ArgumentParser("DictAS Online Inference")
    parser.add_argument("--sequence_dir", type=str,default="demo/pcb/normal_image_support")
    parser.add_argument("--save_path", type=str, default="./results/pcb_results_3")
    parser.add_argument("--checkpoint_path", type=str,default="checkpoints/dict/clip_base/train_mvtec/epoch_5.pth")
    parser.add_argument("--config_path", type=str,default="checkpoints/backbone/clip/Vit-B-16.json")
    parser.add_argument("--pretrained_path", type=str,default="checkpoints/backbone/clip/VIT_b_16_clip.pt")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--features_list", type=int, nargs="+",default=[2, 5, 7, 11])
    parser.add_argument("--sigm", type=int, default=1)
    parser.add_argument("--scale_list", type=int, nargs="+", default=[1])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=222)

    args = parser.parse_args()
    torch.cuda.set_device(args.device_id)
    setup_seed(args.seed)
    test(args)







