import torch.nn as nn
import torch
from encoders.r2d2_encoder.export_image_embeddings import get_pretrained_model
import torchvision.transforms as tvf
from encoders.sp_encoder.export_image_embeddings import SuperPoint
from vggt.models.vggt import VGGT
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, feature_type):
        super(FeatureExtractor, self).__init__()
        self.feature_type = feature_type
        if feature_type == "sp":
            print("Loading SuperPoint model...")
            self.model = SuperPoint().cuda().eval()
            self.feature_dim = 256
        elif feature_type == "r2d2":
            print("Loading R2D2 model...")
            self.model = get_pretrained_model()
            self.feature_dim = 128
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
            self.norm_image = tvf.Compose([tvf.Normalize(mean=mean, std=std)])
        elif feature_type == "vggt":
            print("Loading VGGT model...")
            self.model = VGGT()
            self.model.load_state_dict(torch.load("vggt_weight/VGGT-1B.pt"))
            self.feature_dim = 128
        else:
            raise ValueError("Foundation model not supported")
        

    @torch.no_grad()
    def forward(self, image):
        if self.feature_type == "sp":
            features, scores = self.model(image)
            return {
                "feature_map": features,
                "scores": scores
            }
        elif self.feature_type == "r2d2":
            image = self.norm_image(image)
            features, repeatability, reliability = self.model(image)
            return {
                "feature_map": features,
                "repeatability": repeatability,
                "reliability": reliability
            }
        elif self.feature_type == "vggt":
            image = preprocess_image_tensor(image[0], mode="pad")
            aggregated_tokens_list, ps_idx = self.model.aggregator(image) # [B, S, 1374, 2048] 5
            depth_feature_map, depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, image, ps_idx) # [B, S, 518, 518, 1]  [B, S, 518, 518]
            return {
                "feature_map": depth_feature_map,
                "depth_map": depth_map,
                "depth_conf": depth_conf
            }


def preprocess_image_tensor(image: torch.Tensor, mode: str = "crop", target_size: int = 518) -> torch.Tensor:
    """
    Preprocess a single image tensor to match the behavior of
    vggt.utils.load_fn.load_and_preprocess_images for one image.

    Args:
        image: Tensor with shape (C, H, W). Values expected in [0,1].
        mode: "crop" or "pad". Matches load_and_preprocess_images behavior.
        target_size: Target width (and height for pad) in pixels.

    Returns:
        Tensor with shape (3, H_out, W_out)
    """
    if image.dim() != 3:
        raise ValueError("Expected image tensor with shape (C, H, W)")

    # Ensure 3 channels (drop alpha if present)
    if image.shape[0] >= 3:
        img = image[:3, ...]
    else:
        # If grayscale, repeat to 3 channels
        img = image.repeat(3 // image.shape[0], 1, 1)

    _, height, width = img.shape

    if mode not in ["crop", "pad"]:
        raise ValueError("mode must be either 'crop' or 'pad'")

    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
    else:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

    # Resize (expects NCHW)
    img_resized = F.interpolate(
        img.unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False
    ).squeeze(0)

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img_resized = img_resized[:, start_y : start_y + target_size, :]

    if mode == "pad":
        h_padding = target_size - img_resized.shape[1]
        w_padding = target_size - img_resized.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            img_resized = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), value=1.0)
    img_resized = img_resized.clamp(0.0, 1.0)
    if img_resized.dim() == 3:
        img_resized = img_resized.unsqueeze(0).unsqueeze(0)
    return img_resized
