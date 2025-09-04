import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from gaussian_renderer import get_render_visible_mask, render_gsplat
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state, seed_everything
from utils.graphics_utils import focal2fov, fov2focal
from utils.image_utils import get_resolution_from_longest_edge
from utils.loss_utils import *

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import pickle

import torch.nn.functional as F

from encoders.feature_extractor import FeatureExtractor
from scene.kpdetector import KpDetector

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_FOUND = True
except ImportError:
    SKLEARN_FOUND = False

try:
    from skimage import color
    SKIMAGE_FOUND = True
except ImportError:
    SKIMAGE_FOUND = False


def calculate_delta_e(rgb1, rgb2):
    """
    计算两个RGB颜色之间的ΔE差异
    
    Args:
        rgb1: 第一组RGB颜色 tensor，形状为 (N, 3)，值范围 [0, 1]
        rgb2: 第二组RGB颜色 tensor，形状为 (N, 3)，值范围 [0, 1]
    
    Returns:
        delta_e: ΔE值 tensor，形状为 (N,)，值越小表示颜色越相似
    """
    if SKIMAGE_FOUND:
        # 使用 skimage 进行色彩空间转换
        # 转换为 numpy 并调整形状用于 skimage
        rgb1_np = rgb1.detach().cpu().numpy()
        rgb2_np = rgb2.detach().cpu().numpy()
        
        # skimage 需要形状为 (..., 3) 的数组
        lab1 = color.rgb2lab(rgb1_np[..., None, :]).squeeze()
        lab2 = color.rgb2lab(rgb2_np[..., None, :]).squeeze()
        
        # 计算 ΔE76
        delta_e = torch.sqrt(
            torch.sum((torch.tensor(lab1, device=rgb1.device) - torch.tensor(lab2, device=rgb1.device)) ** 2, dim=-1)
        )
    else:
        # 如果没有 skimage，使用简化的计算方法（直接在RGB空间计算欧几里得距离）
        print("Warning: skimage not found, using RGB Euclidean distance as approximation")
        delta_e = torch.sqrt(torch.sum((rgb1 - rgb2) ** 2, dim=-1))
    
    return delta_e


def get_sampled_gaussian(gaussians: GaussianModel, idx_sampled):
    sampled_gaussians = GaussianModel(gaussians.max_sh_degree)
    sampled_gaussians._xyz = gaussians._xyz[idx_sampled]
    sampled_gaussians._loc_feature = gaussians._loc_feature[idx_sampled]
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    return sampled_gaussians


@torch.no_grad()
def calculate_match_score(
    gaussians: GaussianModel,
    gt_image,               # image 3xHxW
    gt_depth,               # depth HxW
    gt_depth_confidence,    # depth confidence HxW
    pose,
    K,
    render_visible_mask=None,   # 设置gaussian球的可见性
    img_mask=None,              # 设置图片中像素位置的可见性
):
    xyz = gaussians.get_xyz
    gs_rgb = gaussians.get_features[:,:3].squeeze()
    
    # project gaussians to image space
    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=-1)
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_homo = xyz_cam / depths

    xy = (K @ xyz_cam_homo)[:2].long()
    
    # 在图片中会显示的gaussian点
    in_mask = (
        (xy[0] >= 0)
        & (xy[0] < gt_image.shape[2])
        & (xy[1] >= 0)
        & (xy[1] < gt_image.shape[1])
    )

    if render_visible_mask is not None: # 与传入的可见性求并集
        visible_mask = in_mask & render_visible_mask
    else:
        visible_mask = in_mask

    if img_mask is not None:            # 设置在可见像素中的Gaussian球
        visible_xy = xy[:, in_mask]
        img_mask_expand = torch.zeros_like(visible_mask, dtype=torch.bool)
        img_mask_expand[in_mask] = img_mask[0, visible_xy[1], visible_xy[0]]
        visible_mask = visible_mask & img_mask_expand

    xy = xy[:, visible_mask]
    depths = depths[visible_mask]
    gs_rgb = gs_rgb[visible_mask]
    
    # 计算Gaussian球特征与图片颜色的ΔE差异
    im_rgb = gt_image[:, xy[1], xy[0]].T  # 转置为 (N_visible, 3)
    rgb_delta_e = calculate_delta_e(gs_rgb, im_rgb)
    
    # 计算Gaussian球深度与图片深度的差异
    depth_diff = torch.abs(depths - gt_depth[xy[1], xy[0]])
    
    rgb_sigma = 10.0  
    rgb_score = torch.exp(-rgb_delta_e / rgb_sigma)
    
    depth_sigma = 1.0  
    depth_score = torch.exp(-depth_diff / depth_sigma)
    
    return rgb_score, depth_score, visible_mask

def generate_gt_map(
    gaussians: GaussianModel,
    gt_feature_map,
    idx_sampled,
    pose,
    K,
    render_visible_mask=None,
):
    if render_visible_mask is not None:
        render_visible_mask = render_visible_mask[idx_sampled]
        idx_sampled = idx_sampled[render_visible_mask]
    sampled_xyz = gaussians.get_xyz[idx_sampled]

    gt_map = torch.zeros(
        (1, gt_feature_map.shape[1], gt_feature_map.shape[2]),
        device=gt_feature_map.device,
    )
    
    xyz_homo = torch.cat(
        [sampled_xyz, torch.ones(sampled_xyz.shape[0], 1, device=sampled_xyz.device)],
        dim=-1,
    )
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_norm = xyz_cam / depths

    xy = (K @ xyz_cam_norm)[:2].long()

    in_mask = (
        (xy[0] >= 0)
        & (xy[0] < gt_feature_map.shape[2])
        & (xy[1] >= 0)
        & (xy[1] < gt_feature_map.shape[1])
    )

    xy_pos = xy[:, in_mask]

    gt_map[:, xy_pos[1], xy_pos[0]] = 1

    return gt_map


def random_knn_score(points, npoints, score, k=32):
    # points xyz坐标
    # npoints 采样后数量
    # score 每个点的分数
    # k k近邻数量
    sampled_idx = torch.randperm(points.shape[0])[:npoints]
    sampled_points = points[sampled_idx]
    points = points.cpu()
    sampled_points = sampled_points.cpu()
    
    # 批量处理距离计算以避免内存问题
    batch_size = 1000  # 每批处理1000个采样点
    knn_idx_list = []
    
    for i in range(0, len(sampled_points), batch_size):
        batch_end = min(i + batch_size, len(sampled_points))
        batch_sampled = sampled_points[i:batch_end]
        
        # 计算当前批次的距离
        batch_dist = torch.cdist(batch_sampled, points) # 计算采样点到所有点的最近距离
        batch_knn_idx = torch.topk(batch_dist, k, largest=False, dim=-1)[1] # 选出最近的k个点
        knn_idx_list.append(batch_knn_idx)  
    
    # 合并所有批次的结果
    knn_idx = torch.cat(knn_idx_list, dim=0).cuda()

    # knn select
    knn_score = score[knn_idx]  # (npoints, k)
    score_knn_sort_idx = torch.argsort(         # 每组中分数排序
        knn_score, descending=True, dim=-1
    )  # (npoints, k)

    final_sampled_idx = set()

    for i in range(npoints):
        for j in score_knn_sort_idx[i]:         # 选择每组中第1/2/3...高的点
            idx = knn_idx[i, j].item()  
            if idx not in final_sampled_idx: 
                final_sampled_idx.add(idx)  
                break  

    return torch.tensor(list(final_sampled_idx)).cuda()


def random_knn_score_efficient(points, npoints, score, k=32):
    """
    内存高效的 KNN 实现，使用 sklearn 的 NearestNeighbors
    """
    if not SKLEARN_FOUND:
        print("Warning: sklearn not found, falling back to batched torch implementation")
        return random_knn_score(points, npoints, score, k)
    
    sampled_idx = torch.randperm(points.shape[0])[:npoints]
    sampled_points = points[sampled_idx]
    
    # 转换为 numpy 进行高效的近邻搜索
    points_np = points.cpu().detach().numpy()
    sampled_points_np = sampled_points.cpu().detach().numpy()
    
    # 使用 sklearn 的 NearestNeighbors，内存效率更高
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    nbrs.fit(points_np)
    
    # 批量查找最近邻
    distances, knn_idx = nbrs.kneighbors(sampled_points_np)
    knn_idx = torch.from_numpy(knn_idx).cuda()
    
    # knn select
    knn_score = score[knn_idx]  # (npoints, k)
    score_knn_sort_idx = torch.argsort(
        knn_score, descending=True, dim=-1
    )  # (npoints, k)

    final_sampled_idx = set()

    for i in range(npoints):
        for j in score_knn_sort_idx[i]:
            idx = knn_idx[i, j].item()  
            if idx not in final_sampled_idx: 
                final_sampled_idx.add(idx)  
                break  

    return torch.tensor(list(final_sampled_idx)).cuda()


def random_knn_score_color_diverse(points, npoints, score, colors, k=32, color_weight=0.5, use_lab=True):
    """
    基于KNN的评分+颜色多样性联合采样。
    - points: (N,3) 空间坐标
    - npoints: 目标采样数
    - score: (N,) 每点分数（越大越好）
    - colors: (N,3) RGB ∈[0,1]
    - k: 每个种子对应的空间近邻数
    - color_weight: 颜色多样性权重，越大越偏好多样性
    - use_lab: 若可用，使用LAB空间计算ΔE
    """
    if color_weight <= 0:
        return random_knn_score_efficient(points, npoints, score, k)

    sampled_idx = torch.randperm(points.shape[0])[:npoints]
    sampled_points = points[sampled_idx]

    # 近邻搜索（优先用 sklearn，更省内存）
    if SKLEARN_FOUND:
        points_np = points.detach().cpu().numpy()
        sampled_points_np = sampled_points.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
        nbrs.fit(points_np)
        _, knn_idx_np = nbrs.kneighbors(sampled_points_np)
        knn_idx = torch.from_numpy(knn_idx_np).to(points.device)
    else:
        points_cpu = points.detach().cpu()
        sampled_points_cpu = sampled_points.detach().cpu()
        batch_size = 1000
        knn_idx_list = []
        for i in range(0, len(sampled_points_cpu), batch_size):
            batch_end = min(i + batch_size, len(sampled_points_cpu))
            batch_sampled = sampled_points_cpu[i:batch_end]
            batch_dist = torch.cdist(batch_sampled, points_cpu)
            batch_knn_idx = torch.topk(batch_dist, k, largest=False, dim=-1)[1]
            knn_idx_list.append(batch_knn_idx)
        knn_idx = torch.cat(knn_idx_list, dim=0).to(points.device)

    # 预计算颜色空间
    colors = colors.clamp(0, 1)
    if use_lab and SKIMAGE_FOUND:
        colors_np = colors.detach().cpu().numpy()
        lab_np = color.rgb2lab(colors_np)
        lab_colors = torch.tensor(lab_np, device=points.device, dtype=colors.dtype)
    else:
        lab_colors = colors

    final_sampled_idx = set()

    for i in range(npoints):
        group = knn_idx[i]  # (k,)
        s = score[group]
        # 组内归一化，避免尺度偏置
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-8)

        c = lab_colors[group].float()
        if k > 1:
            dmat = torch.cdist(c, c, p=2)                 # (k,k)
            contrast = dmat.sum(dim=1) / (k - 1)          # 平均ΔE，越大越“与众不同”
        else:
            contrast = torch.zeros_like(s, dtype=torch.float32, device=s.device)
        c_norm = (contrast - contrast.min()) / (contrast.max() - contrast.min() + 1e-8)

        obj = (1.0 - color_weight) * s_norm + color_weight * c_norm
        order = torch.argsort(obj, descending=True)
        for j in order:
            idx = group[j].item()
            if idx not in final_sampled_idx:
                final_sampled_idx.add(idx)
                break

    return torch.tensor(list(final_sampled_idx), device=points.device)


def matching_oriented_sample(
    scene,
    gaussians,
    feature_extractor,
    render_visible_masks,
    masks=None, # 使用Mask2Former生成的 stuff_mask(前景物体)、sky_mask(天空)、undistort_mask(未失真)
    num=16384,
    k=32,
    weight=0.5,
    color_diverse_weight=0.0,
):
    viewpoint_stack = scene.getTrainCameras().copy()
    rgb_score_sum = torch.zeros(    # 每个Gaussian可见次数的和
        gaussians.get_xyz.shape[0], dtype=torch.float32, device="cuda"
    )
    depth_score_sum = torch.zeros(    # 每个Gaussian可见次数的和
        gaussians.get_xyz.shape[0], dtype=torch.float32, device="cuda"
    )
    score_num = torch.zeros(    # 每个Gaussian分数的和
        gaussians.get_xyz.shape[0], dtype=torch.int, device="cuda"
    )
    fine_resolution = (         # 图像的size
        viewpoint_stack[0].original_image.shape[1],
        viewpoint_stack[0].original_image.shape[2],
    )

    for viewpoint_cam in tqdm(viewpoint_stack, desc="Match Score"):
        gt_image = viewpoint_cam.original_image.cuda()  # 图片  
        
        gt_depth = feature_extractor(gt_image[None])["depth_map"]  # 图片深度
        gt_depth_confidence = feature_extractor(gt_image[None])["depth_conf"]  # 图片深度置信度
        
        gt_image = F.interpolate(                 # 将图片特征插值到与图像相同的size
            gt_image,
            size=(fine_resolution[0], fine_resolution[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        
        gt_depth = F.interpolate(                 # 将图片depth插值到与图像相同的size
            gt_depth,
            size=(fine_resolution[0], fine_resolution[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        
        gt_depth_confidence = F.interpolate(                 # 将图片depth confidence插值到与图像相同的size
            gt_depth_confidence,
            size=(fine_resolution[0], fine_resolution[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)


        # 外参
        viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
        focalX = fov2focal(viewpoint_cam.FoVx, gt_image.shape[2])
        focalY = fov2focal(viewpoint_cam.FoVy, gt_image.shape[1])
        # print("focal:", focalX, focalY)
        # 内参
        K = torch.tensor(
            [
                [focalX, 0.0, gt_image.shape[2] / 2],
                [0.0, focalY, gt_image.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        # 获取可见的Gaussian椭球mask'
        if render_visible_masks.get(viewpoint_cam.image_name, None) is None:
            render_visible_mask = get_render_visible_mask(
                gaussians,
                viewpoint_cam,
                gt_image.shape[2],
                gt_image.shape[1],
            )
            render_visible_masks[viewpoint_cam.image_name] = render_visible_mask
        if masks is not None:
            object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
            distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]
            mask = object_mask & distort_mask   # 去除天空、移动物体和失真区域，保留前景物体
            img_mask = (
                F.interpolate(
                    mask[None].float(),
                    size=(gt_image.shape[1], gt_image.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                > 0.5
            )
        else:
            img_mask = None
        # 获取每个Gaussian球与图片特征的余弦相似度分数和mask
        image_color_score, image_depth_score, mask = calculate_match_score(
            gaussians,
            gt_image,
            gt_depth,
            viewmat,
            K,
            render_visible_mask=render_visible_masks[viewpoint_cam.image_name],
            img_mask=img_mask,
        )
        score_num[mask] += 1
        rgb_score_sum[mask] += image_color_score
        depth_score_sum[mask] += image_depth_score
        
    score_avg = weight * rgb_score_sum + (1 - weight) * depth_score_sum
    
    score_num[score_num == 0] = 1  # avoid divide by zero
    score_avg = score_avg / score_num # 每个点的平均分数（若该点从未出现为0）      

    # knn采样（可选颜色多样性）
    if color_diverse_weight > 0.0:
        gs_colors = gaussians.get_features[:,:3].squeeze().clamp(0, 1)
        sampled_idx = random_knn_score_color_diverse(
            gaussians.get_xyz, num, score_avg, gs_colors, k=k, color_weight=color_diverse_weight
        )
    else:
        sampled_idx = random_knn_score_efficient(gaussians.get_xyz, num, score_avg, k=k)
    sampled_idx = torch.unique(sampled_idx)
    return sampled_idx, score_avg, score_num
