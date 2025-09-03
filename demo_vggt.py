from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_point_conf_distribution(point_conf):
    point_conf = point_conf.cpu().numpy()
    point_conf = point_conf.reshape(-1)
    plt.hist(point_conf, bins=100)
    plt.savefig("point_conf_distribution.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT()
model.load_state_dict(torch.load("vggt_weight/VGGT-1B.pt"))
model.to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["kn_church-2.jpg"]  
images = load_and_preprocess_images(image_names).to(device)
# 保存model.aggregator的参数
torch.save(model.aggregator.state_dict(), "vggt_weight/aggregator_params.pt")
# 保存model.depth_head的参数
torch.save(model.depth_head.state_dict(), "vggt_weight/depth_head_params.pt")
# 保存model.point_head的参数
torch.save(model.point_head.state_dict(), "vggt_weight/point_head_params.pt")
# 保存model.camera_head的参数
torch.save(model.camera_head.state_dict(), "vggt_weight/camera_head_params.pt")

images = images[None]  # add batch dimension
print(images.shape)
depth_time = 0
point_time = 0
aggregator_time = 0
start_time = time()
for i in tqdm(range(1)):
    with torch.no_grad():
        t0 = time()
        with torch.amp.autocast(dtype=dtype, device_type=device):
            
            aggregated_tokens_list, ps_idx = model.aggregator(images) # [B, S, 1374, 2048] 5
            
        # Predict Cameras
        # pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        t1 = time()
        # Predict Depth Maps
        depth_feature_map, depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx) # [B, S, 518, 518, 1]  [B, S, 518, 518]
        t2 = time()
        # Predict Point Maps
        point_feature_map, point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx) # [B, S, 518, 518, 3]  [B, S, 518, 518]
        # 获取point_conf的分布
        get_point_conf_distribution(depth_conf)
        print(depth_conf.max(), depth_conf.min())
        t3 = time()
        depth_time += t2 - t1
        point_time += t3 - t2
        aggregator_time += t1 - t0
        del aggregated_tokens_list, ps_idx, depth_map, depth_conf, point_map, point_conf
end_time = time()
print(f"Total time: {(end_time - start_time)/100}")
print(f"Depth time: {depth_time/100}, Point time: {point_time/100}")
print(f"Aggregator time: {aggregator_time/100}")
