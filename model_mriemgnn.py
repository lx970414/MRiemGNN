import torch
import torch.nn as nn
from layer_mriemgnn import MRiemGNNLayer

class MRiemGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, relation_types, spaces, num_layers=2, manifold_params=None, num_kernels=64, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.spaces = spaces
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = in_dim if i == 0 else hid_dim
            self.layers.append(
                MRiemGNNLayer(
                    in_dim=layer_in,
                    out_dim=hid_dim if i < num_layers - 1 else out_dim,
                    relation_types=relation_types,
                    spaces=spaces,
                    manifold_params=manifold_params,
                    num_kernels=num_kernels
                )
            )

    def forward(self, x, adj_dict):
        h = x
        out_per_space = {s: h for s in self.spaces}
        for layer in self.layers:
            out = layer(h, adj_dict)
            for s in self.spaces:
                out_per_space[s] = out[s]
            h = out_per_space[self.spaces[0]]  # 层间只传主空间特征
        return out_per_space
    # def forward(self, x, adj_dict):
    #     h = x
    #     out_per_space = {s: h for s in self.spaces}
    #     for layer in self.layers:
    #         out = layer(h, adj_dict)
    #         for s in self.spaces:
    #             out_per_space[s] = out[s]
    #         h = out_per_space[self.spaces[0]]  # 层间只传主空间特征
    #     return out_per_space
