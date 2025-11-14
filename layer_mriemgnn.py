import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import SpectroRiemannianKernel
from manifolds.euclidean import EuclideanManifold
from manifolds.hyperboloid import HyperboloidManifold

class MRiemGNNLayer(nn.Module):
    """
    多关系多空间核方法GNN层
    """

    def __init__(self, in_dim, out_dim, relation_types, spaces, manifold_params, num_kernels=64):
        super().__init__()
        self.relation_types = relation_types
        self.spaces = spaces
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_kernels = num_kernels

        # 流形定义
        self.manifolds = nn.ModuleDict({
            'Euclidean': EuclideanManifold(),
            'Hyperbolic': HyperboloidManifold()
        })

        # 每空间-关系 分配kernel
        self.kernels = nn.ModuleDict({
            f'{s}_{r}': SpectroRiemannianKernel(in_dim, num_kernels=num_kernels, sigma=manifold_params[s].get('sigma', 1.0))
            for s in spaces for r in relation_types
        })

        # 每空间-关系 的卷积权重
        self.weight = nn.ParameterDict({
            f'{s}_{r}': nn.Parameter(torch.Tensor(num_kernels, out_dim))
            for s in spaces for r in relation_types
        })

        # 可学习曲率参数
        self.curvature = nn.ParameterDict({
            f'{s}_{r}': nn.Parameter(torch.ones(1) * manifold_params[s].get('init_k', 1.0))
            for s in spaces for r in relation_types
        })

        # 残差/自环
        self.self_loop = nn.ParameterDict({
            s: nn.Parameter(torch.Tensor(in_dim, out_dim))
            for s in spaces
        })
        self.bias = nn.ParameterDict({
            s: nn.Parameter(torch.zeros(out_dim))
            for s in spaces
        })

        # 关系融合权重（softmax）
        self.relation_weight = nn.ParameterDict({
            s: nn.Parameter(torch.zeros(len(relation_types)))
            for s in spaces
        })

        self.reset_parameters()

    def reset_parameters(self):
        for k in self.weight:
            nn.init.xavier_uniform_(self.weight[k])
        for k in self.self_loop:
            nn.init.xavier_uniform_(self.self_loop[k])
        for k in self.bias:
            nn.init.zeros_(self.bias[k])
        # relation_weight 初始化为0

    def forward(self, h, adj_dict):
        out = {s: [] for s in self.spaces}
        for s in self.spaces:
            rel_out = []
            for idx, r in enumerate(self.relation_types):
                edge_index = adj_dict[r]  # [2, num_edges]
                kernel = self.kernels[f'{s}_{r}']
                W = self.weight[f'{s}_{r}']

                h_kernel = kernel(h)  # [N, num_kernels]
                row, col = edge_index
                agg = torch.zeros(h.shape[0], self.out_dim, device=h.device, dtype=h.dtype)  # [N, out_dim]
                agg = agg.index_add(0, row, h_kernel[col] @ W)  # [num_edges, out_dim]
                deg = torch.bincount(row, minlength=h.shape[0]).clamp(min=1).unsqueeze(-1)
                agg = agg / deg  # [N, out_dim]
                rel_out.append(agg)

            # 多关系softmax融合
            alpha = F.softmax(self.relation_weight[s], dim=0)
            rel_stack = torch.stack(rel_out, dim=0)
            fused = (alpha.view(-1,1,1) * rel_stack).sum(dim=0)

            # 残差/自环
            out_h = fused + h @ self.self_loop[s] + self.bias[s]
            out[s] = out_h

        return out  # {space: [N, out_dim]}
