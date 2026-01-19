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

from models.dictionary.DictAS import MyDictionary
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


#MAIN TEST
def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_path, exist_ok=True)

    #BACKBONE LAYERS
    model_CLIP, _, _ = Load_CLIP(args.image_size,args.pretrained_path,device=device)
    model_CLIP.eval().to(device)

    #DICTIONARY LAYERS
    with open(args.config_path, "r") as f:
        model_configs = json.load(f)

    Mymodel = MyDictionary(model_configs, args).to(device)
    Mymodel.eval()

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    Mymodel.load_state_dict(checkpoint["Mymodel"])


    #TEST DATA
    transform_image = _transform_test(args.image_size)
    image_paths = load_images_from_dir(args.sequence_dir)
    assert len(image_paths) >= 2, "Need at least 2 images"

    #DICTIONARY MEMORY
    patch_good_tokens_memory = None


    for idx, img_path in enumerate(image_paths):
        print(f"[{idx+1}/{len(image_paths)}] {os.path.basename(img_path)}")

        #Image loading
        img_pil = Image.open(img_path)
        img_tensor = transform_image(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, patch_tokens = model_CLIP.encode_image(img_tensor, args.features_list)
            patch_tokens = [norm_patch(p, True) for p in patch_tokens]
            patch_tokens = [Mymodel.Value_Generator(p) for p in patch_tokens]

        #Init memory
        if idx == 0:
            patch_good_tokens_memory = patch_tokens
            print("Dictionary initialized")
            continue

        #Test
        with torch.no_grad():
            anomaly_map_list, Retrived_list_ClS = Mymodel(patch_tokens, patch_good_tokens_memory, mode="test")

            anomaly_maps = []
            for amap in anomaly_map_list:
                amap = F.interpolate(amap.unsqueeze(1), size=args.image_size, mode="bilinear", align_corners=True).squeeze()
                anomaly_maps.append(amap)

            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=0), dim=0).cpu().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigm)
            print(anomaly_map.shape)

        # num_anomalous_pixel = np.sum(anomaly_map>0.1)
        # print(num_anomalous_pixel)
        count = np.count_nonzero(anomaly_map.ravel() > 0.1)
        print(count)

        print(anomaly_map)
        print(anomaly_map.min())
        print(anomaly_map.max())
        print(anomaly_map.mean())

        # #VISUALIZATION
        raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_h, orig_w = raw_img.shape[:2]

        # Ensure anomaly map is 2D
        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map.squeeze()

        #Resize anomaly map (use NEAREST for masks if you want sharper edges)
        anomaly_resized = cv2.resize(anomaly_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        #Binary mask
        binary_mask_vis = np.zeros((orig_h, orig_w), dtype=np.uint8)
        binary_mask_vis[anomaly_resized > 0.1] = 255
        

        #Convert to 3 channels for concatenation
        binary_mask_vis = cv2.cvtColor(binary_mask_vis, cv2.COLOR_GRAY2BGR)

        #Concatenate
        gap = np.ones((orig_h, 10, 3), dtype=np.uint8) * 255
        vis_concat = np.concatenate([raw_img, gap, binary_mask_vis], axis=1)

        #Save
        save_name = f"binary_result_{idx:03d}.png"
        cv2.imwrite(os.path.join(args.save_path, save_name), vis_concat)

        if idx==20:break


        #Update memory
        updated_memory = []
        for old, new in zip(patch_good_tokens_memory, patch_tokens):
            updated_memory.append(torch.cat([old, new], dim=0))
        patch_good_tokens_memory = updated_memory



if __name__ == "__main__":

    parser = argparse.ArgumentParser("DictAS Online Inference")
    parser.add_argument("--sequence_dir", type=str,default="/SOLUTION/Defect_detection_pcb/dataset/Demo/training_arch_4")
    parser.add_argument("--save_path", type=str, default="./results/pcb_results_4")
    parser.add_argument("--checkpoint_path", type=str,default="checkpoints/dict/clip_base/train_mvtec/epoch_5.pth")
    parser.add_argument("--config_path", type=str,default="checkpoints/backbone/clip/Vit-B-16.json")
    parser.add_argument("--pretrained_path", type=str,default="checkpoints/backbone/clip/VIT_b_16_clip.pt")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--features_list", type=int, nargs="+",default=[2, 5, 7, 11])
    parser.add_argument("--sigm", type=int, default=6)
    parser.add_argument("--scale_list", type=int, nargs="+", default=[1])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=222)

    args = parser.parse_args()
    torch.cuda.set_device(args.device_id)
    setup_seed(args.seed)
    test(args)
