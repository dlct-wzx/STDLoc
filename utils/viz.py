import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import torch
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None]*2
    c = x*np.array([[0, 1., 0]]) + (2-x)*np.array([[1., 0, 0]])
    return np.clip(c, 0, 1)


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors='lime', ps=4):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    axes = plt.gcf().axes
    for a, k, c in zip(axes, kpts, colors):
        a.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0)


def plot_matches(kpts0, kpts1, color=None, scores=None, lw=1.5, ps=4, indices=(0, 1), a=1.):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        scores: score of each match
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        if scores is None:
            color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
        else:
            color = [list(*cm_RdGn(x)) for x in scores]
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        fig.lines += [matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1, transform=fig.transFigure, c=color[i], linewidth=lw,
            alpha=a)
            for i in range(len(kpts0))]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    # print(len(color), color[0])
    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches_w_gt_point(kpts0, kpts1, kpts0_gt_in_2, color=None, scores=None, lw=1.5, ps=4, indices=(0, 1), a=1.):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        scores: score of each match
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        if scores is None:
            color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
        else:
            color = [list(*cm_RdGn(x)) for x in scores]
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        
        fig.lines += [matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1, transform=fig.transFigure, c=color[i], linewidth=lw,
            alpha=a)
            for i in range(len(kpts0))]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    # print(len(color), color[0])
    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def add_text(idx, text, pos=(0.01, 0.99), fs=15, color='w',
             lcolor='k', lwidth=2, ha='left', va='top'):
    ax = plt.gcf().axes[idx]
    t = ax.text(*pos, text, fontsize=fs, ha=ha, va=va,
                color=color, transform=ax.transAxes)
    if lcolor is not None:
        t.set_path_effects([
            path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
            path_effects.Normal()])


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches='tight', pad_inches=0, **kw)
    
    
def draw_keypoint(image, kpts, scores=None, ps=4):
    """Draw keypoint on image
    Args:
        image: RGB, HWC
        kps: N*2
        color: RGB
    """
    plot_images([image])
    
    if scores is None:
        colors='lime'
    else:
        colors = [list(*cm_RdGn(x)) for x in scores]
    plot_keypoints([kpts], colors, ps=ps)
    

def draw_matches(image0, image1, kpts0, kpts1, scores, lw=1.5, 
                 ps=4, alpha=1.0):
    """Draw keypoint on image
    Args:
        image: RGB, HWC
        kps: N*2
        color: RGB
    """
    plot_images([image0, image1])
    colors = [list(*cm_RdGn(x)) for x in scores]
    plot_matches(kpts0, kpts1, colors, lw=lw, ps=ps, a=alpha)



def slerp(r1, r2, t):
    """Spherical interpolation between two rotations.
    Args:
        r1, r2: 3x3 rotation matrices.
        t: interpolation factor.
    Returns:
        3x3 interpolated rotation matrix.
    """
    # r1 = R.from_matrix(r1)
    # r2 = R.from_matrix(r2)
    times = []
    for i in range(t):
        times.append(i)
    key_times = [times[0], times[-1]]
    slerp = Slerp(key_times, R.from_matrix([r1, r2]))
    inter_r = slerp(times)
    return inter_r


def interpolate_pose(pose1, pose2, t):
    """Interpolate between two poses.
    Args:
        pose1, pose2: 4x4 transformation matrices.
        t: interpolation factor.
    Returns:
        4x4 interpolated transformation matrix.
    """
    device = pose1.device
    pose1 = pose1.cpu().numpy()
    pose2 = pose2.cpu().numpy()
    r1, r2 = pose1[:3, :3], pose2[:3, :3]
    t1, t2 = pose1[:3, 3], pose2[:3, 3]

    # interpolate rotation
    rs = slerp(r1, r2, t)

    # interpolate translation
    ts = []
    for t in np.linspace(0, 1, t):
        position = (1 - t) * t1 + t * t2
        ts.append(position)

    # compose the interpolated pose
    poses = []
    for r_t, t_t in zip(rs, ts):
        inter_pose = np.eye(4)
        inter_pose[:3, :3] = r_t.as_matrix()
        inter_pose[:3, 3] = t_t
        poses.append(torch.tensor(inter_pose, device=device, dtype=torch.float32))

    return poses

def stitch_images(image_a, image_b, mask_left=None, mask_right=None):
    if mask_left is None:
        print(image_a.shape)
        width, height = image_a.shape[:2]
        mask_left = np.zeros((width, height), dtype=bool)

        a = width/height
        for x in range(width):
            for y in range(height):
                if x > y*a:  
                    mask_left[x, y] = True
                else:  
                    continue
        mask_right = ~mask_left
    
    stitched_image = np.zeros_like(image_a)
    stitched_image[mask_left] = image_a[mask_left]
    stitched_image[mask_right] = image_b[mask_right]
    return stitched_image, mask_left, mask_right
    
