from torch.backends.cuda import matmul

from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)

        # q:BxNxD,K:BxNxKxD
        # q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        q, key, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        # pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        # BxNxKxD
        pos_enc = self.fc_delta(knn_xyz)
        # 8x1024x16x3
        epsilon = 1e-6
        weights = 1.0 / (dists.gather(2, knn_idx) + epsilon)
        weighted_avg = torch.sum(pos_enc * weights.unsqueeze(-1), dim=2) / torch.sum(weights, dim=2, keepdim=True)
        score = torch.matmul(q + weighted_avg, (key + weighted_avg).transpose(-2, -1)) / np.sqrt(key.size(-1))
        map = F.softmax(score, dim=-1)
        # attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        #
        # attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        attn = torch.matmul(map, v + weighted_avg)

        # res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(attn) + pre
        return res, attn
