#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterization_2dgs

from scene.gaussian_model import GaussianModel


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def get_render_visible_mask(
    pc: GaussianModel, pose, fovx, fovy, width, height, **rasterize_args
):
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    focal_length_x = width / (2 * tanfovx)
    focal_length_y = height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, width / 2.0],
            [0, focal_length_y, height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    rotations = pc.get_rotation

    colors = pc.get_features  # [N, K, 3]
    sh_degree = pc.active_sh_degree
    viewmat = pose  # [4, 4]

    scales = pc.get_scaling
    if scales.shape[-1] == 2:
        scales = torch.cat(
            [
                scales,
                torch.ones(
                    scales.shape[0], 1, device=scales.device, dtype=scales.dtype
                ),
            ],
            dim=-1,
        )  # (n, 2) -> (n, 3)
        # print(colors.shape)
        colors, alphas, normals, surf_normals, distort, median_depth, info = (
            rasterization_2dgs(
                means=means3D,  # [N, 3]
                quats=rotations,  # [N, 4]
                scales=scales,  # [N, 3]
                opacities=opacity.squeeze(-1),  # [N,]
                colors=colors,
                viewmats=viewmat[None],  # [1, 4, 4]
                Ks=K[None],  # [1, 3, 3]
                width=width,
                height=height,
                packed=False,
                sh_degree=sh_degree,
                render_mode="RGB",
                **rasterize_args
            )
        )
        # [1, H, W, 3] -> [3, H, W]
        colors.sum().backward()
        render_visible_mask = means3D.grad.norm(dim=-1) > 0
        means3D.grad.zero_()
        # print("visible_filter_num:", (info['radii'] > 0).sum())
    else:
        colors, render_alphas, info = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=colors,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=width,
            height=height,
            packed=False,
            sh_degree=sh_degree,
            **rasterize_args
        )

        colors.sum().backward()
        render_visible_mask = means3D.grad.norm(dim=-1) > 0
        means3D.grad.zero_()

    # print("render visible:", render_visible_mask.sum())
    return render_visible_mask


def render_gsplat(
    viewpoint_camera,
    pc: GaussianModel,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    rgb_only=False,
    norm_feat_bf_render=True,
    near_plane=0.01,
    far_plane=10000,
    longest_edge=640,
    **rasterize_args
):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    if scales.shape[1] == 2:
        return render_gsplat_2dgs(
            viewpoint_camera,
            pc,
            bg_color,
            scaling_modifier,
            override_color,
            rgb_only,
            norm_feat_bf_render,
            near_plane,
            far_plane,
            longest_edge,
            **rasterize_args
        )

    rotations = pc.get_rotation
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)  # [4, 4]
    if override_color is not None:
        colors = override_color  # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features  # [N, K, 3]
        sh_degree = pc.active_sh_degree

    if bg_color is None:
        bg_color = torch.zeros(3, device="cuda")

    width, height = viewpoint_camera.image_width, viewpoint_camera.image_height
    max_edge = max(width, height)
    if max_edge > longest_edge:
        scaling_modifier *= longest_edge / max_edge
        width, height = int(width * scaling_modifier), int(height * scaling_modifier)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = width / (2 * tanfovx)
    focal_length_y = height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, width / 2.0],
            [0, focal_length_y, height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    # render color
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=width,
        height=height,
        packed=False,
        sh_degree=sh_degree,
        near_plane=near_plane,
        far_plane=far_plane,
        **rasterize_args
    )
    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    color = rendered_image
    radii = info["radii"].squeeze(0)  # [N,]
    visible_mask = radii > 0
    try:
        info["means2d"].retain_grad()  # [1, N, 2]
    except:
        pass

    # render feature map
    if rgb_only is False:
        loc_feature = pc.get_loc_feature[visible_mask].squeeze()
        if norm_feat_bf_render:
            loc_feature = F.normalize(loc_feature, p=2, dim=-1)

        feat_map, alphas, meta = rasterization(
            means=means3D[visible_mask],  # [N, 3]
            quats=rotations[visible_mask],  # [N, 4]
            scales=scales[visible_mask],  # [N, 3]
            opacities=opacity.squeeze(-1)[visible_mask],  # [N,]
            colors=loc_feature,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=width,
            height=height,
            packed=False,
            near_plane=near_plane,
            far_plane=far_plane,
            **rasterize_args
        )
        feat_map = feat_map[0].permute(2, 0, 1)
        feat_map = F.normalize(feat_map, p=2, dim=0)
    else:
        feat_map = None

    return {
        "render": color,
        "feature_map": feat_map,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "alphas": render_alphas,
    }


def render_gsplat_2dgs(
    viewpoint_camera,
    pc: GaussianModel,
    bg_color=None,
    scaling_modifier=1.0,
    override_color=None,
    rgb_only=False,
    norm_feat_bf_render=True,
    near_plane=0.01,
    far_plane=10000,
    longest_edge=640,
    **rasterize_args
):
    """
    Render the scene.
    """
    width, height = viewpoint_camera.image_width, viewpoint_camera.image_height
    max_edge = max(width, height)
    if max_edge > longest_edge:
        scaling_modifier *= longest_edge / max_edge
        width, height = int(width * scaling_modifier), int(height * scaling_modifier)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = width / (2 * tanfovx)
    focal_length_y = height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, width / 2.0],
            [0, focal_length_y, height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    scales = torch.cat(
        [
            scales,
            torch.ones(scales.shape[0], 1, device=scales.device, dtype=scales.dtype),
        ],
        dim=-1,
    )  # (n, 2) -> (n, 3)
    rotations = pc.get_rotation
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)  # [4, 4]
    if override_color is not None:
        colors = override_color  # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features  # [N, K, 3]
        sh_degree = pc.active_sh_degree

    if bg_color is None:
        bg_color = torch.zeros(4, device="cuda")

    # expand to 4 channels
    if bg_color.shape[0] == 3:
        bg_color = torch.cat([bg_color, bg_color[:1]], dim=0)

    colors, alphas, normals, surf_normals, distort, median_depth, info = (
        rasterization_2dgs(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=colors,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=width,
            height=height,
            packed=False,
            sh_degree=sh_degree,
            backgrounds=bg_color[None],
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode="RGB+ED",
            **rasterize_args
        )
    )
    # [1, H, W, 3] -> [3, H, W]
    rendered_image = colors[0].permute(2, 0, 1)
    color = rendered_image[:3]
    depth = rendered_image[3:]
    radii = info["radii"].squeeze(0)  # [N,]
    visible_mask = radii > 0

    try:
        info["gradient_2dgs"].retain_grad()  # [1, N, 2]
    except:
        pass

    if rgb_only is False:
        loc_feature = pc.get_loc_feature[visible_mask].squeeze()
        if norm_feat_bf_render:
            loc_feature = F.normalize(loc_feature, p=2, dim=-1)

        feat_map, _, _, _, _, _, _ = rasterization_2dgs(
            means=means3D[visible_mask],  # [N, 3]
            quats=rotations[visible_mask],  # [N, 4]
            scales=scales[visible_mask],  # [N, 3]
            opacities=opacity.squeeze(-1)[visible_mask],  # [N,]
            colors=loc_feature.squeeze(),
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=width,
            height=height,
            packed=False,
            tile_size=8,
            near_plane=near_plane,
            far_plane=far_plane,
            **rasterize_args
        )

        feat_map = feat_map[0].permute(2, 0, 1)
        feat_map = F.normalize(feat_map, p=2, dim=0)
    else:
        feat_map = None

    return {
        "render": color,
        "rend_alpha": alphas,
        "rend_normal": normals,
        "surf_normal": surf_normals,
        "rend_dist": distort,
        "rend_median": median_depth,
        "feature_map": feat_map,
        "viewspace_points": info["gradient_2dgs"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "surf_depth": depth,
    }


def render_from_pose_gsplat(
    pc: GaussianModel,
    pose,
    fovx,
    fovy,
    width,
    height,
    bg_color=None,
    render_mode="RGB+ED",
    rgb_only=False,
    norm_feat_bf_render=True,
    near_plane=0.01,
    far_plane=10000,
    **rasterize_args
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    colors = pc.get_features  # [N, K, 3]
    sh_degree = pc.active_sh_degree

    if scales.shape[1] == 2:
        return render_from_pose_gsplat_2dgs(
            pc,
            pose,
            fovx,
            fovy,
            width,
            height,
            bg_color,
            render_mode,
            rgb_only,
            norm_feat_bf_render,
            near_plane,
            far_plane,
            **rasterize_args
        )

    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    focal_length_x = width / (2 * tanfovx)
    focal_length_y = height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, width / 2.0],
            [0, focal_length_y, height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    if bg_color is None:
        bg_color = torch.zeros(3, device="cuda")

    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=pose[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(width),
        height=int(height),
        packed=False,
        sh_degree=sh_degree,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode=render_mode,
        **rasterize_args
    )
    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    color = rendered_image[:3]
    if rendered_image.shape[0] == 4:
        depth = rendered_image[3:]
    else:
        depth = None
    radii = info["radii"].squeeze(0)  # [N,]
    visible_mask = radii > 0

    try:
        info["means2d"].retain_grad()  # [1, N, 2]
    except:
        pass

    if rgb_only is False:
        loc_feature = pc.get_loc_feature[visible_mask].squeeze()
        if norm_feat_bf_render:
            loc_feature = F.normalize(loc_feature, p=2, dim=-1)

        feat_map, alphas, meta = rasterization(
            means3D[visible_mask],
            rotations[visible_mask],
            scales[visible_mask],
            opacity.squeeze(-1)[visible_mask],
            loc_feature,
            pose[None],
            K[None],
            int(width),
            int(height),
            packed=False,
            near_plane=near_plane,
            far_plane=far_plane,
            **rasterize_args
        )
        feat_map = feat_map[0].permute(2, 0, 1)
        feat_map = F.normalize(feat_map, p=2, dim=0)
    else:
        feat_map = None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": color,
        "feature_map": feat_map,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "alphas": render_alphas,
        "depth": depth,
    }


def render_from_pose_gsplat_2dgs(
    pc: GaussianModel,
    pose,
    fovx,
    fovy,
    width,
    height,
    bg_color=None,
    render_mode="RGB+ED",
    rgb_only=False,
    norm_feat_bf_render=True,
    near_plane=0.01,
    far_plane=10000,
    **rasterize_args
):
    """
    Render the scene.
    """
    means3D = pc.get_xyz
    opacity = pc.get_opacity.squeeze()
    scales = pc.get_scaling
    scales = torch.cat(
        [
            scales,
            torch.ones(scales.shape[0], 1, device=scales.device, dtype=scales.dtype),
        ],
        dim=-1,
    )  # (n, 2) -> (n, 3)
    rotations = pc.get_rotation

    focalx = fov2focal(fovx, width)
    focaly = fov2focal(fovy, height)
    K = torch.tensor(
        [[focalx, 0, width / 2], [0, focaly, height / 2], [0, 0, 1]]
    ).cuda()

    colors = pc.get_features  # [N, K, 3]
    sh_degree = pc.active_sh_degree

    if bg_color is None:
        bg_color = torch.zeros(4, device="cuda")

    colors, alphas, normals, surf_normals, distort, median_depth, info = (
        rasterization_2dgs(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=colors,
            viewmats=pose[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=int(width),
            height=int(height),
            packed=False,
            sh_degree=sh_degree,
            backgrounds=bg_color[None],
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode=render_mode,
            **rasterize_args
        )
    )
    # [1, H, W, 3] -> [3, H, W]
    rendered_image = colors[0].permute(2, 0, 1)
    color = rendered_image[:3]
    depth = rendered_image[3:]
    radii = info["radii"].squeeze(0)  # [N,]
    visible_mask = radii > 0

    try:
        info["gradient_2dgs"].retain_grad()  # [1, N, 2]
    except:
        pass

    if rgb_only is False:
        loc_feature = pc.get_loc_feature.squeeze()
        if norm_feat_bf_render:
            loc_feature = F.normalize(loc_feature, p=2, dim=-1)

        feat_map, _, _, _, _, _, _ = rasterization_2dgs(
            means=means3D[visible_mask],  # [N, 3]
            quats=rotations[visible_mask],  # [N, 4]
            scales=scales[visible_mask],  # [N, 3]
            opacities=opacity.squeeze(-1)[visible_mask],  # [N,]
            colors=loc_feature.squeeze(),
            viewmats=pose[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=int(width),
            height=int(height),
            packed=False,
            tile_size=8,
            near_plane=near_plane,
            far_plane=far_plane,
            **rasterize_args
        )
        feat_map = feat_map[0].permute(2, 0, 1)
        feat_map = F.normalize(feat_map, p=2, dim=0)
    else:
        feat_map = None

    return {
        "render": color,
        "rend_alpha": alphas,
        "rend_normal": normals,
        "surf_normal": surf_normals,
        "rend_dist": distort,
        "rend_median": median_depth,
        "feature_map": feat_map,
        "viewspace_points": info["gradient_2dgs"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
    }
