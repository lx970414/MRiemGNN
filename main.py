import torch
from model_mriemgnn import MRiemGNN
import torch.optim as optim
from train import train
import argparse
from data import load_data, load_mat_multiplex_auto, build_adj_dict, load_link_pred_edges

def main():
    print("main.py has begun execution")  # ←1

    # 可选命令行参数控制
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='mat', choices=['pt', 'mat'],
                        help='输入数据类型')
    parser.add_argument('--mat_path', type=str, default='data/small_alibaba_1_10/small_alibaba_1_10.mat',
                        help='默认数据集')
    parser.add_argument('--enable_metapath', action='store_true',
                        help='是否自动生成meta-path并参与聚合')
    parser.add_argument('--max_metapath_length', type=int, default=2,
                        help='最大metapath长度')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练epoch数量')
    parser.add_argument('--task', type=str, default='node_cls', choices=['node_cls', 'link_pred'],
                        help='选择任务类型: 节点分类(node_cls) 或 链路预测(link_pred)')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lambda_mutual', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=2.0)
    parser.add_argument('--beta', type=float, default=0.6, help='Fusion coefficient for Euclidean/Hyperbolic')

    args = parser.parse_args()
    print("Parameter parsing complete!")  # ←2

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")

    if args.data_type == 'pt':
        x, y, edge_index, edge_type, train_idx, val_idx, test_idx, num_relations = load_data()
        all_rel_names = [f'relation_{i}' for i in range(num_relations)]
    elif args.data_type == 'mat':
        print("Preparing to read .mat data")  # ←3
        x, y, edge_index, edge_type, train_idx, val_idx, test_idx, num_relations, all_rel_names = \
            load_mat_multiplex_auto(
                args.mat_path,
                enable_metapath=args.enable_metapath,
                max_metapath_length=args.max_metapath_length
            )
        print(".mat data reading complete!")  # ←4

    adj_dict = build_adj_dict(edge_index, edge_type, num_relations)
    print("adj_dict construction complete!")  # ←5
    data = {
        'x': x,
        'y': y,
        'adj_dict': adj_dict,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    print("Data encapsulation complete!")  # ←6

    num_features = x.size(1)
    num_classes = int(y.max().item() + 1)
    relation_types = list(adj_dict.keys())
    spaces = ['Euclidean', 'Hyperbolic']
    manifold_params = {
        'Euclidean': {'init_k': 1.0, 'sigma': 1.0},
        'Hyperbolic': {'init_k': 1.0, 'sigma': 1.0}
    }

    print("Model initialization")  # ←7
    model = MRiemGNN(
        in_dim=num_features,
        hid_dim=64,
        out_dim=num_classes,
        relation_types=relation_types,
        spaces=spaces,
        num_layers=2,
        manifold_params=manifold_params,
        num_kernels=64,
        dropout=0.5
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    print("Model initialization complete, start training!")  # ←8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.task == 'link_pred':
        # 只加载链路预测分好的边
        import scipy.io as sio
        mat = sio.loadmat(args.mat_path)
        fields = list(mat.keys())
        # 检查必须字段
        needed_fields = ['train_edges', 'train_edges_neg', 'valid_edges', 'valid_edges_neg', 'test_edges',
                         'test_edges_neg']
        for field in needed_fields:
            if field not in mat:
                raise ValueError(f"Field {field} missing in your link prediction split mat file!")
        # 组装假的 data 对象，传给后续train()
        data = {
            'train_edges': mat['train_edges'],
            'train_edges_neg': mat['train_edges_neg'],
            'valid_edges': mat['valid_edges'],
            'valid_edges_neg': mat['valid_edges_neg'],
            'test_edges': mat['test_edges'],
            'test_edges_neg': mat['test_edges_neg'],
            # 其他用不到的可置None或不加
        }
        # 下面就不用走load_mat_multiplex_auto了
    else:
        # 节点分类任务走原有流程
        train(
            model, optimizer, data, relation_types, spaces,
            epochs=args.epochs, device=args.device, task=args.task, patience=args.patience,
            lambda_mutual=args.lambda_mutual, tau=args.tau, beta=args.beta
        )
    print("End of train() calls")  # ←9

if __name__ == "__main__":
    print("Coming up on main()")  # ←0
    main()

# import torch
# from model_mriemgnn import MRiemGNN
# import torch.optim as optim
# from train import train
# import argparse
# import numpy as np
# import scipy.io as sio
# from scipy.sparse import issparse
# from data import load_data, load_mat_multiplex_auto, build_adj_dict
#
# def edges_to_array(edges):
#     """兼容各种 mat 的 edges 字段（支持多关系、邻接矩阵、edge list）"""
#     # 多关系 (1,4) 或 (4,) array，内部是稀疏邻接矩阵
#     if isinstance(edges, np.ndarray):
#         if edges.dtype == object:
#             return np.hstack([edges_to_array(e) for e in edges.flatten()])
#         if edges.ndim == 2 and min(edges.shape) == 1 and issparse(edges[0, 0]):
#             return np.hstack([edges_to_array(e) for e in edges.flatten()])
#         if edges.ndim == 1 and all(issparse(e) for e in edges):
#             arrs = []
#             for mat in edges:
#                 coo = mat.tocoo()
#                 arrs.append(np.stack([coo.row, coo.col], axis=0))
#             return np.hstack(arrs)
#         if edges.ndim == 2 and edges.shape[0] == 2:  # [2, N]
#             return edges
#         if edges.ndim == 2 and edges.shape[1] == 2:  # [N, 2]
#             return edges.T
#         if edges.ndim == 2 and edges.shape[0] == edges.shape[1]:
#             coo = edges.tocoo() if issparse(edges) else np.nonzero(edges)
#             return np.stack([coo.row, coo.col], axis=0)
#     # 稀疏矩阵
#     if issparse(edges):
#         coo = edges.tocoo()
#         return np.stack([coo.row, coo.col], axis=0)
#     raise ValueError(f"edges格式未知，type={type(edges)}, shape={getattr(edges, 'shape', None)}")
#
# def sample_edges_by_idx(idx, edges_all):
#     """给定节点子集 idx，从全体 edges_all 中采样 idx 子图内的所有正边，并等数量负边"""
#     edge_array = edges_to_array(edges_all)
#     mask = np.zeros(edge_array.max() + 1, dtype=bool)
#     mask[idx] = True
#     u, v = edge_array
#     submask = mask[u] & mask[v]
#     pos_edges = np.stack([u[submask], v[submask]], axis=1)
#     exist = set(map(tuple, pos_edges))
#     rng = np.random.default_rng(42)
#     neg_edges = []
#     tries = 0
#     while len(neg_edges) < len(pos_edges) and tries < 100 * len(pos_edges):
#         uu = rng.choice(idx)
#         vv = rng.choice(idx)
#         if uu != vv and (uu, vv) not in exist and (vv, uu) not in exist:
#             neg_edges.append((uu, vv))
#         tries += 1
#     neg_edges = np.array(neg_edges)
#     return pos_edges, neg_edges
#
# def main():
#     print("main.py has begun execution")
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_type', type=str, default='mat', choices=['pt', 'mat'])
#     parser.add_argument('--mat_path', type=str)
#     parser.add_argument('--epochs', type=int, default=300)
#     parser.add_argument('--task', type=str, default='node_cls', choices=['node_cls', 'link_pred'])
#     parser.add_argument('--lr', type=float, default=0.01)
#     parser.add_argument('--weight_decay', type=float, default=0.01)
#     parser.add_argument('--patience', type=int, default=100)
#     parser.add_argument('--lambda_mutual', type=float, default=1.0)
#     parser.add_argument('--tau', type=float, default=2.0)
#     parser.add_argument('--hid_dim', type=int, default=64)
#     parser.add_argument('--dropout', type=float, default=0.5)
#     args = parser.parse_args()
#     print("Parameter parsing complete!")
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#
#     if args.task == 'link_pred':
#         mat = sio.loadmat(args.mat_path)
#         print(f"[Field Detection] All field names of the current .mat file： {list(mat.keys())}")
#         link_fields = ['train_edges', 'train_edges_neg', 'valid_edges', 'valid_edges_neg', 'test_edges',
#                        'test_edges_neg']
#         is_pure_linkpred_mat = all(f in mat for f in link_fields)
#
#         if is_pure_linkpred_mat:
#             print("Detected pure link prediction .mat, skipping feature/label detection.")
#             data = {
#                 'train_edges': mat['train_edges'],
#                 'train_edges_neg': mat['train_edges_neg'],
#                 'valid_edges': mat['valid_edges'],
#                 'valid_edges_neg': mat['valid_edges_neg'],
#                 'test_edges': mat['test_edges'],
#                 'test_edges_neg': mat['test_edges_neg'],
#             }
#             all_edges = np.concatenate([
#                 data['train_edges'], data['train_edges_neg'],
#                 data['valid_edges'], data['valid_edges_neg'],
#                 data['test_edges'], data['test_edges_neg'],
#             ])
#             num_nodes = int(all_edges.max()) + 1
#             feat_dim = args.hid_dim
#             x = torch.randn(num_nodes, feat_dim)
#             data['x'] = x
#             relation_types = ['default']
#             data['adj_dict'] = {'default': torch.zeros(num_nodes, num_nodes)}
#             spaces = ['Euclidean', 'Hyperbolic']
#             manifold_params = {
#                 'Euclidean': {'init_k': 1.0, 'sigma': 1.0},
#                 'Hyperbolic': {'init_k': 1.0, 'sigma': 1.0}
#             }
#         else:
#             print("不是纯链路预测mat，自动走节点分类标准读入流程。")
#             x, y, edge_index, edge_type, train_idx, val_idx, test_idx, num_relations, all_rel_names = \
#                 load_mat_multiplex_auto(
#                     args.mat_path,
#                     enable_metapath=False,
#                     max_metapath_length=2
#                 )
#             adj_dict = build_adj_dict(edge_index, edge_type, num_relations)
#             data = {
#                 'x': x,
#                 'adj_dict': adj_dict,
#                 'train_idx': train_idx,
#                 'val_idx': val_idx,
#                 'test_idx': test_idx,
#             }
#             relation_types = list(adj_dict.keys())
#             spaces = ['Euclidean', 'Hyperbolic']
#             manifold_params = {
#                 'Euclidean': {'init_k': 1.0, 'sigma': 1.0},
#                 'Hyperbolic': {'init_k': 1.0, 'sigma': 1.0}
#             }
#             # 采样link prediction所需正负边
#             if 'edges' in mat:
#                 all_edges = mat['edges']
#                 if hasattr(all_edges, 'tocoo'):
#                     all_edges = all_edges.tocoo()
#                 tr_pos, tr_neg = sample_edges_by_idx(train_idx, all_edges)
#                 va_pos, va_neg = sample_edges_by_idx(val_idx, all_edges)
#                 te_pos, te_neg = sample_edges_by_idx(test_idx, all_edges)
#                 data['train_edges'] = tr_pos
#                 data['train_edges_neg'] = tr_neg
#                 data['valid_edges'] = va_pos
#                 data['valid_edges_neg'] = va_neg
#                 data['test_edges'] = te_pos
#                 data['test_edges_neg'] = te_neg
#             else:
#                 raise ValueError("你的mat文件既没有分好train_edges也没有edges原始字段，无法采样！")
#
#         # 初始化模型
#         model = MRiemGNN(
#             in_dim=args.hid_dim if is_pure_linkpred_mat else x.size(1),
#             hid_dim=args.hid_dim,
#             out_dim=args.hid_dim,
#             relation_types=relation_types,
#             spaces=spaces,
#             num_layers=2,
#             manifold_params=manifold_params,
#             num_kernels=64,
#             dropout=args.dropout
#         )
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
#         train_pos = data.get('train_edges', None)
#         train_neg = data.get('train_edges_neg', None)
#         valid_pos = data.get('valid_edges', None)
#         valid_neg = data.get('valid_edges_neg', None)
#         test_pos = data.get('test_edges', None)
#         test_neg = data.get('test_edges_neg', None)
#
#         train(
#             model, optimizer, data, relation_types, spaces,
#             train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg,
#             epochs=args.epochs, device=device, task=args.task, patience=args.patience,
#             lambda_mutual=args.lambda_mutual, tau=args.tau
#         )
#     else:
#         # 节点分类标准流程
#         print("Preparing to read .mat data")
#         x, y, edge_index, edge_type, train_idx, val_idx, test_idx, num_relations, all_rel_names = \
#             load_mat_multiplex_auto(
#                 args.mat_path,
#                 enable_metapath=False,
#                 max_metapath_length=2
#             )
#         print(".mat data reading complete!")
#         adj_dict = build_adj_dict(edge_index, edge_type, num_relations)
#         print("adj_dict construction complete!")
#         data = {
#             'x': x,
#             'y': y,
#             'adj_dict': adj_dict,
#             'train_idx': train_idx,
#             'val_idx': val_idx,
#             'test_idx': test_idx
#         }
#         print("Data encapsulation complete!")
#         num_features = x.size(1)
#         num_classes = int(y.max().item() + 1)
#         relation_types = list(adj_dict.keys())
#         spaces = ['Euclidean', 'Hyperbolic']
#         manifold_params = {
#             'Euclidean': {'init_k': 1.0, 'sigma': 1.0},
#             'Hyperbolic': {'init_k': 1.0, 'sigma': 1.0}
#         }
#
#         model = MRiemGNN(
#             in_dim=num_features,
#             hid_dim=args.hid_dim,
#             out_dim=num_classes,
#             relation_types=relation_types,
#             spaces=spaces,
#             num_layers=2,
#             manifold_params=manifold_params,
#             num_kernels=64,
#             dropout=args.dropout
#         )
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
#         train(
#             model, optimizer, data, relation_types, spaces,
#             epochs=args.epochs, device=device, task=args.task, patience=args.patience,
#             lambda_mutual=args.lambda_mutual, tau=args.tau
#         )
#
#     print("End of train() calls")
#
#
# if __name__ == "__main__":
#     print("Coming up on main()")
#     main()



