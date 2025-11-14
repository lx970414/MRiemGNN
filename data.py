import torch
import numpy as np
import scipy.io
import scipy.sparse
import re

def load_link_pred_edges(mat_path):
    mat = scipy.io.loadmat(mat_path)
    def get(field): return mat[field] if field in mat else None

    pos_neg_pairs = {}
    for split in ["train", "valid", "test"]:
        pos_edges = get(f"{split}_edges")
        neg_edges = get(f"{split}_edges_neg")
        def norm_edges(arr):
            if arr is None:
                return None
            arr = np.array(arr)
            # 常规情况应该是 (N, 2)，否则尝试转置
            if arr.shape[1] == 2:
                return arr
            elif arr.shape[0] == 2:
                return arr.T
            else:
                raise ValueError(f"Edge array shape not recognized: {arr.shape}")
        pos_edges = norm_edges(pos_edges)
        neg_edges = norm_edges(neg_edges)
        pos_neg_pairs[split] = (pos_edges, neg_edges)
    return pos_neg_pairs  # dict: train/valid/test => (pos_edges, neg_edges)


def process_sparse_or_array(field, dtype):
    """兼容稀疏矩阵和普通ndarray/object等"""
    if scipy.sparse.issparse(field):
        arr = field.toarray()
        arr = arr.reshape(-1).astype(dtype)
    else:
        arr = np.array(field)
        if arr.dtype == np.object_:
            if arr.shape == ():
                arr = arr.item()
                arr = np.array(arr)
            elif arr.ndim == 1:
                arr = np.array([np.array(a).flatten() for a in arr])
            else:
                arr = np.array(arr.tolist())
        arr = arr.reshape(-1).astype(dtype)
    return arr

def safe_np_array(mat_field, dtype=np.float32):
    """
    支持matlab导出object嵌套结构，包括0-d array和单独对象。
    """
    arr = np.array(mat_field)
    # 如果是object类型
    while isinstance(arr, np.ndarray) and arr.dtype == np.object_:
        # 如果是0维object，解包后递归
        if arr.shape == ():
            arr = arr.item()
            arr = np.array(arr)
            continue
        # 如果是一维object数组，且每个元素还是array
        elif arr.ndim == 1:
            arr = np.array([np.array(a).flatten() for a in arr])
        # 二维等其他object直接强转
        else:
            arr = np.array(arr.tolist())
        # 若还没转成数值型，继续
        if arr.dtype == np.object_ and arr.shape == ():
            arr = arr.item()
            arr = np.array(arr)
    arr = arr.astype(dtype)
    return arr

def build_adj_dict(edge_index, edge_type, num_relations):
    """
    edge_index: [2, num_edges]
    edge_type: [num_edges]   每条边的类型编号（0 ~ num_relations-1）
    num_relations: 关系类型（或层）总数
    返回 {relation_id: edge_index[2, num_edges_of_r]}
    """
    adj_dict = {}
    for r in range(num_relations):
        mask = (edge_type == r)
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            adj = torch.zeros(2, 0, dtype=torch.long)
        else:
            adj = edge_index[:, idx]
        adj_dict[r] = adj
    return adj_dict

def load_data(
    prefix='data/',
    x_file='node_features.pt',
    y_file='labels.pt',
    edge_index_file='edge_index.pt',
    edge_type_file='edge_type.pt',
    train_file='train_idx.pt',
    val_file='val_idx.pt',
    test_file='test_idx.pt'
):
    x = torch.load(prefix + x_file)        # 节点特征 [N, d]
    y = torch.load(prefix + y_file)        # 节点标签 [N]
    edge_index = torch.load(prefix + edge_index_file)   # [2, num_edges]
    edge_type = torch.load(prefix + edge_type_file)     # [num_edges]
    train_idx = torch.load(prefix + train_file)         # 训练集索引
    val_idx = torch.load(prefix + val_file)             # 验证集索引
    test_idx = torch.load(prefix + test_file)           # 测试集索引
    num_relations = int(edge_type.max().item()) + 1
    return x, y, edge_index, edge_type, train_idx, val_idx, test_idx, num_relations

# 新增：自动识别 .mat 文件中的多邻接和meta-path支持
def is_square_matrix(arr, n):
    try:
        arr = arr if not isinstance(arr, np.ndarray) or arr.ndim != 0 else arr[()]
        if scipy.sparse.issparse(arr):
            shape = arr.shape
        else:
            shape = np.array(arr).shape
        return (len(shape) == 2) and (shape[0] == shape[1] == n)
    except:
        return False

def auto_detect_relation_fields(mat_dict, node_num, skip_fields=None):
    skip = set(skip_fields or []) | {
        'feature', 'full_feature', 'attr', 'attribute', 'x', 'feat',  # 常见特征字段
        'label', 'labels', 'y', 'target',  # 常见标签字段
        'train_idx', 'valid_idx', 'test_idx', 'train', 'val', 'test',  # 常见划分
        'item_num', 'user_num', 'node_num', 'n_nodes',
        'class', '__header__', '__version__', '__globals__'
    }
    adj_fields = []
    for k, v in mat_dict.items():
        if k in skip:
            continue
        # 支持object嵌套
        if isinstance(v, np.ndarray) and v.dtype == np.object_:
            for vi in v.flatten():
                if is_square_matrix(vi, node_num):
                    adj_fields.append((k, vi))
        elif is_square_matrix(v, node_num):
            adj_fields.append((k, v))
    return adj_fields

def auto_generate_metapaths(adj_fields, max_length=2):
    # 简化版，两两乘，生成2-hop meta-path
    meta_adjs = []
    n = len(adj_fields)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            a = adj_fields[i][1]
            b = adj_fields[j][1]
            if not scipy.sparse.issparse(a): a = scipy.sparse.coo_matrix(a)
            if not scipy.sparse.issparse(b): b = scipy.sparse.coo_matrix(b)
            meta_adj = a @ b
            name = f"{adj_fields[i][0]}_{adj_fields[j][0]}"
            meta_adjs.append((name, meta_adj))
    return meta_adjs

def auto_detect_fields(mat):
    # 通用主流特征/标签/划分字段优先自动识别
    possible_features = ['feature', 'features', 'full_feature', 'attr', 'attribute', 'x', 'feat']
    possible_labels   = ['label', 'labels', 'y', 'target', 'gt']
    possible_train    = ['train_idx', 'train', 'train_mask']
    possible_valid    = ['valid_idx', 'val_idx', 'valid', 'val', 'valid_mask', 'val_mask']
    possible_test     = ['test_idx', 'test', 'test_mask']

    feature_field = next((k for k in possible_features if k in mat), None)
    label_field   = next((k for k in possible_labels if k in mat), None)
    train_idx_field = next((k for k in possible_train if k in mat), None)
    val_idx_field   = next((k for k in possible_valid if k in mat), None)
    test_idx_field  = next((k for k in possible_test if k in mat), None)
    # 检查是否有缺失字段
    if not all([feature_field, label_field, train_idx_field, val_idx_field, test_idx_field]):
        raise ValueError(f"Not recognized automatically! Check your .mat file! The fields have：{list(mat.keys())}")
    return feature_field, label_field, train_idx_field, val_idx_field, test_idx_field


def load_mat_multiplex_auto(
    mat_path,
    enable_metapath=False,
    max_metapath_length=2,
    feature_field=None,
    label_field=None,
    train_idx_field=None,
    val_idx_field=None,
    test_idx_field=None,
    node_num_field='item_num'
):
    print("Reading .mat file...", mat_path)
    mat = scipy.io.loadmat(mat_path)
    print(f"[Field Detection] All field names of the current .mat file： {list(mat.keys())}")

    # 判断是否只有链路预测的字段
    if set(['train_edges', 'train_edges_neg', 'valid_edges', 'valid_edges_neg', 'test_edges',
            'test_edges_neg']).issubset(mat.keys()):
        print("Detected pure link prediction .mat, skipping feature/label detection.")
        return {
            'train_edges': mat['train_edges'],
            'train_edges_neg': mat['train_edges_neg'],
            'valid_edges': mat['valid_edges'],
            'valid_edges_neg': mat['valid_edges_neg'],
            'test_edges': mat['test_edges'],
            'test_edges_neg': mat['test_edges_neg'],
        }

    # 自动识别主字段
    if not feature_field or not label_field or not train_idx_field or not val_idx_field or not test_idx_field:
        feature_field, label_field, train_idx_field, val_idx_field, test_idx_field = auto_detect_fields(mat)

    node_num = int(mat[node_num_field][0][0]) if node_num_field in mat else \
        mat[feature_field].shape[0] if hasattr(mat[feature_field],'shape') else mat[feature_field].shape[0]
    adj_fields = auto_detect_relation_fields(mat, node_num)
    edge_indices, edge_types, all_rel_names = [], [], []
    for rel_id, (rel_name, adj) in enumerate(adj_fields):
        if scipy.sparse.issparse(adj): coo = adj.tocoo()
        else: coo = scipy.sparse.coo_matrix(adj)
        edges = np.vstack((coo.row, coo.col))
        edge_indices.append(edges)
        edge_types.append(np.full(edges.shape[1], rel_id, dtype=np.int64))
        all_rel_names.append(rel_name)
    # meta-path支持
    if enable_metapath:
        meta_adjs = auto_generate_metapaths(adj_fields, max_length=max_metapath_length)
        for rel_name, adj in meta_adjs:
            rel_id = len(all_rel_names)
            coo = adj.tocoo()
            edges = np.vstack((coo.row, coo.col))
            edge_indices.append(edges)
            edge_types.append(np.full(edges.shape[1], rel_id, dtype=np.int64))
            all_rel_names.append(rel_name)
    edge_index = np.hstack(edge_indices)
    edge_type = np.concatenate(edge_types)

    # 特征
    feat = mat[feature_field]
    if scipy.sparse.issparse(feat):
        x = feat.toarray().astype(np.float32)
    else:
        x = np.array(feat).astype(np.float32)
    # 标签与index
    y = process_sparse_or_array(mat[label_field], dtype=np.int64)
    train_idx = process_sparse_or_array(mat[train_idx_field], dtype=np.int64)
    val_idx = process_sparse_or_array(mat[val_idx_field], dtype=np.int64)
    test_idx = process_sparse_or_array(mat[test_idx_field], dtype=np.int64)

    return (
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long(),
        torch.from_numpy(edge_index).long(),
        torch.from_numpy(edge_type).long(),
        torch.from_numpy(train_idx).long(),
        torch.from_numpy(val_idx).long(),
        torch.from_numpy(test_idx).long(),
        len(all_rel_names),
        all_rel_names
    )


