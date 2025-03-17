import numpy as np
import torch
import cv2
import pycolmap
import poselib
import math

def solve_pose(p2d, p3d, K, solver="poselib", reprojection_error=8.0, confidence=0.9999, max_iterations=100000, min_iterations=1000):
    match_num = p2d.shape[0]
    if match_num < 4:
        print("[SKIP] No enough matches")
        return np.eye(4, dtype=np.float32), np.array([])
    

    if solver == "opencv":
        success, rvec, tvec, inliers = cv2.solvePnPRansac(p3d, 
                                                          p2d, 
                                                          K, 
                                                          distCoeffs=np.zeros((4,1)), 
                                                          reprojectionError=reprojection_error, 
                                                          confidence=confidence)
        if success:
            w2c = np.eye(4)
            cv2.Rodrigues(rvec, w2c[:3, :3])
            w2c[:3, -1] = tvec.flatten()
            w2c = w2c.astype(np.float32)
            inliers = np.array(inliers.flatten())
            return w2c, inliers
            
    elif solver == "colmap":
        camera = pycolmap.Camera(
            model="PINHOLE",
            width=int(K[0, 2] * 2),
            height=int(K[1, 2] * 2),
            params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        )
        estimation_ops = pycolmap.AbsolutePoseEstimationOptions()
        estimation_ops.ransac.max_error = reprojection_error
        estimation_ops.ransac.confidence = confidence
        refine_ops = pycolmap.AbsolutePoseRefinementOptions()

        res = pycolmap.absolute_pose_estimation(p2d, p3d, camera, estimation_ops, refine_ops)

        if res is not None:
            w2c = res['cam_from_world'].matrix()
            w2c = np.array(w2c).contiguous()
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0).astype(np.float32)
            inliers = res["inliers"]
            indices = np.where(inliers)[0]
            inliers = indices.reshape(-1, 1).astype(np.int32)
            return w2c, np.array(inliers.flatten())
        
    elif solver == "poselib":
        camera = {'model': 'PINHOLE', 'width': int(K[0, 2] * 2), 'height': int(K[1, 2] * 2), 
                  'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}
        
        max_reproj_error = reprojection_error
        confidence = confidence
        
        pose, info = poselib.estimate_absolute_pose(p2d, p3d, camera, 
                                                {   'max_iterations': max_iterations,
                                                    'min_iterations': min_iterations,
                                                    'max_reproj_error': max_reproj_error,
                                                    'success_prob': confidence}, 
                                                {'verbose': False})

        if info['num_inliers'] > 0:
            w2c = pose.Rt
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0).astype(np.float32)
            inliers = info["inliers"]
            indices = np.where(inliers)[0]
            inliers = indices.reshape(-1, 1).astype(np.int32)
            return w2c, inliers.flatten()

    return np.eye(4, dtype=np.float32), np.array([])

def cal_pose_error(pred_w2c, gt_w2c):
    """
    Calculate the pose error between the predicted pose and the ground truth pose.
    """
    pred_R = pred_w2c[:3, :3]
    pred_t = np.linalg.inv(pred_w2c)[:3, -1]
    gt_R = gt_w2c[:3, :3]
    gt_t = np.linalg.inv(gt_w2c)[:3, -1]

    # calculate angle error
    r_err = np.matmul(gt_R, np.transpose(pred_R))
    r_err = cv2.Rodrigues(r_err)[0]
    # Extract the angle.
    ae = np.linalg.norm(r_err) * 180 / math.pi

    # calculate translation error
    te = np.linalg.norm(pred_t - gt_t) * 100

    return ae, te


def compute_reprojection_error(points_3D, points_2D, camera_matrix, w2c):
    """
    Compute the reprojection error between the 3D points and the 2D points.
    """
    projection_matrix = camera_matrix @ w2c[:3, :]
    projected_points = projection_matrix @ torch.cat([points_3D, torch.ones((points_3D.shape[0], 1), device=points_3D.device)], dim=1).t()
    projected_points = projected_points[:2, :] / projected_points[2, :]
    projected_points = projected_points.t()
    reprojection_error = torch.linalg.norm(points_2D - projected_points, dim=1)
    return reprojection_error.mean()


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

def render_path_spiral(views, focal=30, zrate=0.5, rots=2, N=120):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    # poses = np.stack([np.concatenate([view.R.T, view.T[:, None]], 1) for view in views], 0)
    c2w = poses_avg(poses)
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    rads = np.percentile(np.abs(poses[:, :3, 3]), 90, 0)
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(z, up, c)
        # render_pose[:3] =  np.array([[ 9.9996626e-01, -7.5253481e-03, -3.2866236e-03, -5.6992844e-02],
        #             [-7.7875191e-03, -9.9601853e-01, -8.8805482e-02, -2.9015102e+00],
        #             [-2.6052459e-03,  8.8828087e-02, -9.9604356e-01, -2.3510060e+00]])
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses


def spherify_poses(views):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)

    p34_to_44 = lambda p: np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        render_pose = np.eye(4)
        render_pose[:3] = p
        #render_pose[:3, 1:3] *= -1
        new_poses.append(render_pose)

    new_poses = np.stack(new_poses, 0)
    print(new_poses.shape)
    return new_poses
