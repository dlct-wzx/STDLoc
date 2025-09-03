import torch
import torch.nn as nn

def simple_nms(scores, nms_radius: int):
    """
    Fast Non-maximum suppression to remove nearby points
    """
    assert nms_radius >= 0

    # 定义一个局部最大池化操作，用于寻找局部极大值
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    # 创建一个与scores形状相同的全零张量
    zeros = torch.zeros_like(scores)
    # 找到每个位置是否为其邻域内的最大值，生成布尔掩码
    max_mask = scores == max_pool(scores)
    # 进行两轮抑制，进一步去除邻域内的非极大值点
    for _ in range(2):
        # 通过池化扩展极大值掩码，得到被抑制的区域
        supp_mask = max_pool(max_mask.float()) > 0
        # 在被抑制区域内将分数置零，其他区域保持原分数
        supp_scores = torch.where(supp_mask, zeros, scores)
        # 在未被抑制的区域重新寻找极大值
        new_max_mask = supp_scores == max_pool(supp_scores)
        # 更新极大值掩码，保留新找到的极大值且不在抑制区域内
        max_mask = max_mask | (new_max_mask & (~supp_mask))

    # 只保留极大值点的分数，其他位置为零
    res = torch.where(max_mask, scores, zeros)
    return res

class KpDetector(torch.nn.Module):
    def __init__(self, in_dim):
        super(KpDetector, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim, 128, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_map):
        x = self.cnn(feat_map)
        x = self.sigmoid(x)
        return x