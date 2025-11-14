import numpy as np
import torch
import torch.nn.functional as F
import collections
import time
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from data import load_link_pred_edges
from main import argparse

def node_cls_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    acc = (y_true == y_pred).sum() / len(y_true)
    return acc, macro_f1, micro_f1

# def train(model, optimizer, data, relation_types, spaces, epochs=100, lambda_mutual=1.0, tau=2.0, device='cpu', early_stop=True, patience=100):
#     model = model.to(device)
#     x = data['x'].to(device)
#     y = data['y'].to(device)
#     train_idx = data['train_idx'].to(device)
#     valid_idx = data['valid_idx'].to(device)
#     adj_dict = {r: data['adj_dict'][r].to(device) for r in relation_types}

def train_node_classification(model, optimizer, data, relation_types, spaces, epochs=100, lambda_mutual=1.0,
                             tau=2.0, device='cpu', patience=100, beta=0.6):
    import collections
    import time
    import torch
    import torch.nn.functional as F
    model = model.to(device)
    x = data['x'].to(device)
    y = data['y'].to(device)
    train_idx = data['train_idx'].to(device)
    valid_idx = data['val_idx'].to(device)
    test_idx = data['test_idx'].to(device)
    adj_dict = {r: data['adj_dict'][r].to(device) for r in relation_types}

    best_val_f1 = 0
    best_metrics = {}
    # metric_window = collections.deque(maxlen=10)
    best_val_acc = 0
    best_macro_f1 = 0
    best_micro_f1 = 0
    patience_counter = 0
    best_epoch = 0
    start_time = time.time()
    metric_window = collections.deque(maxlen=5)  # For tracking early stopping

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        out_dict = model(x, adj_dict)

        logits_euc = out_dict['Euclidean']      # [N, C]
        logits_hyp = out_dict['Hyperbolic']     # [N, C]
        # beta = args.beta, 自定义
        # beta = args.beta

        # === 先得到表征再融合 ===
        # 若 out_dict 中有 'emb_euc', 'emb_hyp' 可直接用
        if 'emb_euc' in out_dict and 'emb_hyp' in out_dict:
            z_euc = out_dict['emb_euc']
            z_hyp = out_dict['emb_hyp']
        else:
            # 如果只返回logits，用最后一层hidden
            z_euc = logits_euc  
            z_hyp = logits_hyp

        # 融合表征
        z_final = beta * z_euc + (1 - beta) * z_hyp

        # 用融合表征进分类头（如MLP），这里用model.classifier
        logits = model.classifier(z_final)

        # 主损失（只用融合logits）
        loss_task = F.cross_entropy(logits[train_idx], y[train_idx])

        # KL-divergence loss for mutual learning
        p_euc = F.log_softmax(logits_euc[train_idx] / tau, dim=1)
        p_hyp = F.softmax(logits_hyp[train_idx] / tau, dim=1)
        p_euc_s = F.softmax(logits_euc[train_idx] / tau, dim=1)
        p_hyp_l = F.log_softmax(logits_hyp[train_idx] / tau, dim=1)
        loss_mutual = F.kl_div(p_euc, p_hyp, reduction='batchmean') + \
                      F.kl_div(p_hyp_l, p_euc_s, reduction='batchmean')

        # 总损失
        loss = loss_task + lambda_mutual * loss_mutual
        loss.backward()
        optimizer.step()

        # EVAL on train/val/test
        model.eval()
        with torch.no_grad():
            out = model(x, adj_dict)
            logits_euc_eval = out['Euclidean']
            logits_hyp_eval = out['Hyperbolic']

            # 推理时也用融合表征预测
            if 'emb_euc' in out and 'emb_hyp' in out:
                z_euc_eval = out['emb_euc']
                z_hyp_eval = out['emb_hyp']
            else:
                z_euc_eval = logits_euc_eval
                z_hyp_eval = logits_hyp_eval

            z_final_eval = beta * z_euc_eval + (1 - beta) * z_hyp_eval
            logits_eval = model.classifier(z_final_eval)
            pred_eval = logits_eval.argmax(dim=1).cpu().numpy()

            for split, idx in [('Train', train_idx), ('Val', valid_idx), ('Test', test_idx)]:
                idx_np = idx.cpu().numpy()
                acc, macro_f1, micro_f1 = node_cls_metrics(y[idx].cpu().numpy(), pred_eval[idx_np])
                if split == 'Val' and macro_f1 > best_val_f1:
                    best_val_f1 = macro_f1
                    best_metrics = dict(epoch=epoch, acc=acc, macro_f1=macro_f1, micro_f1=micro_f1)
                if split == 'Val' and epoch % 1 == 0:  # Output every epoch
                    epoch_time = time.time() - epoch_start
                    print(
                        f"Epoch {epoch} | {split} Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Micro-F1: {micro_f1:.4f} | Loss: {loss.item():.4f} | Time: {epoch_time:.2f}s")

        metric_window.append(macro_f1)
        if len(metric_window) == metric_window.maxlen and all(x <= metric_window[0] for x in metric_window):
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop at epoch {epoch} (patience={patience})")
                break
        else:
            patience_counter = 0

    total_time = time.time() - start_time
    print(f"\n==== Training Finished ====")
    print(f"Best Val Macro-F1: {best_val_f1:.4f} at epoch {best_metrics.get('epoch', '-')}")
    print(
        f"Val: Acc {best_metrics.get('acc', '-'):.4f} | Macro-F1 {best_metrics.get('macro_f1', '-'):.4f} | Micro-F1 {best_metrics.get('micro_f1', '-'):.4f}")
    print(f"Total Training Time: {total_time:.2f}s")

    # Test result at best Val (for fair reporting, reload best if needed)
    acc, macro_f1, micro_f1 = node_cls_metrics(y[test_idx].cpu().numpy(), pred_eval[test_idx.cpu().numpy()])
    print(f"Test: Acc {acc:.4f} | Macro-F1 {macro_f1:.4f} | Micro-F1 {micro_f1:.4f}\n")

# --------------------- 链路预测 ---------------------
def merge_all_adj_dict(adj_dict, num_nodes):
    """
    合并多关系邻接字典为一个总邻接矩阵（取并集）。返回稠密邻接矩阵 adj_all [num_nodes, num_nodes]，0-1阵。
    """
    adj_all = torch.zeros(num_nodes, num_nodes)
    for r, edge_index in adj_dict.items():
        if edge_index.numel() > 0:
            adj_all[edge_index[0], edge_index[1]] = 1
    return adj_all

def get_pos_neg_edges(adj, num_sample=None):
    """
    获取正负边对（不含自环，负样本随机采样）。adj是0-1稠密邻接阵
    """
    adj = adj.cpu().numpy() if hasattr(adj, 'cpu') else adj
    num_nodes = adj.shape[0]
    pos = np.transpose(np.where(np.triu(adj, 1) > 0))   # 只取上三角（无向图无重复）
    all_pairs = set((i, j) for i in range(num_nodes) for j in range(i+1, num_nodes))
    pos_set = set(map(tuple, pos))
    neg = np.array(list(all_pairs - pos_set))
    np.random.shuffle(neg)
    if num_sample is not None:
        if len(pos) > num_sample:
            pos = pos[np.random.choice(len(pos), num_sample, replace=False)]
        if len(neg) > num_sample:
            neg = neg[:num_sample]
    else:
        num_sample = min(len(pos), len(neg))
        pos = pos[:num_sample]
        neg = neg[:num_sample]
    return pos, neg


def get_link_pred_edges(data):
    # 优先加载分割好的正负样本
    if hasattr(data, 'mat_path'):
        edge_splits = load_link_pred_edges(data.mat_path)
        train_pos, train_neg = edge_splits["train"]
        valid_pos, valid_neg = edge_splits["valid"]
        test_pos,  test_neg  = edge_splits["test"]
        # 如果有 None，回退全图采样
        if train_pos is None or train_neg is None:
            print('[Warning] train_edges 未发现，将自动从全图采样。')
            train_pos, train_neg = get_pos_neg_edges(data, split='train')
        if valid_pos is None or valid_neg is None:
            print('[Warning] valid_edges 未发现，将自动从全图采样。')
            valid_pos, valid_neg = get_pos_neg_edges(data, split='valid')
        if test_pos is None or test_neg is None:
            print('[Warning] test_edges 未发现，将自动从全图采样。')
            test_pos, test_neg = get_pos_neg_edges(data, split='test')
    else:
        # 全图采样（视你原有函数命名）
        train_pos, train_neg = get_pos_neg_edges(data, split='train')
        valid_pos, valid_neg = get_pos_neg_edges(data, split='valid')
        test_pos, test_neg  = get_pos_neg_edges(data, split='test')
    return train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg

# def link_prediction_train(model, optimizer, data, relation_types, spaces,
#     train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg,
#     epochs, device, patience):
#     model = model.to(device)
#     x = data['x'].to(device)
#     adj_dict = {r: data['adj_dict'][r].to(device) for r in relation_types}
#     num_nodes = x.size(0)
#     adj_all = merge_all_adj_dict(adj_dict, num_nodes)
#
#     # 优先加载预分割的边对
#     def load_edge(data, field):
#         e = data.get(field, None)
#         if e is None:
#             return None
#         e = np.array(e)
#         if e.shape[0] == 0:
#             return None
#         return e
#
#     pos_train = load_edge(data, 'train_edges')
#     neg_train = load_edge(data, 'train_edges_neg')
#     pos_val   = load_edge(data, 'valid_edges')
#     neg_val   = load_edge(data, 'valid_edges_neg')
#     pos_test  = load_edge(data, 'test_edges')
#     neg_test  = load_edge(data, 'test_edges_neg')
#
#     # 如无则自动采样（只对train/val/test自己的idx子图采样！）
#     def sample_edges_by_idx(idx):
#         mask = np.zeros(num_nodes, dtype=bool)
#         mask[idx] = True
#         sub_adj = adj_all[mask][:, mask]
#         pos, neg = get_pos_neg_edges(sub_adj)
#         pos = np.array([[idx[i], idx[j]] for i, j in pos])
#         neg = np.array([[idx[i], idx[j]] for i, j in neg])
#         return pos, neg
#
#     if pos_train is None or neg_train is None:
#         print("[Warning] train_edges 未发现，将自动从全图采样。")
#         train_idx = data['train_idx'].cpu().numpy()
#         pos_train, neg_train = sample_edges_by_idx(train_idx)
#
#     if pos_val is None or neg_val is None:
#         print("[Warning] valid_edges 未发现，将自动从全图采样。")
#         valid_idx = data['valid_idx'].cpu().numpy()
#         pos_val, neg_val = sample_edges_by_idx(valid_idx)
#
#     if pos_test is None or neg_test is None:
#         print("[Warning] test_edges 未发现，将自动从全图采样。")
#         test_idx = data['test_idx'].cpu().numpy()
#         pos_test, neg_test = sample_edges_by_idx(test_idx)
#
#     # ====== 训练 ======
#     best_val_auc = 0
#     best_epoch = 0
#     best_result = {}
#     patience_counter = 0
#     start_time = time.time()
#
#     for epoch in range(epochs):
#         epoch_start = time.time()
#         model.train()
#         optimizer.zero_grad()
#         out = model(x, adj_dict)['Euclidean']  # [N, D]
#
#         # train
#         train_pos_score = (out[pos_train[:, 0]] * out[pos_train[:, 1]]).sum(dim=1)
#         train_neg_score = (out[neg_train[:, 0]] * out[neg_train[:, 1]]).sum(dim=1)
#         train_scores = torch.cat([train_pos_score, train_neg_score])
#         train_labels = torch.cat([torch.ones(len(train_pos_score)), torch.zeros(len(train_neg_score))]).to(device)
#         loss = F.binary_cross_entropy_with_logits(train_scores, train_labels)
#         loss.backward()
#         optimizer.step()
#
#         # 验证/测试
#         model.eval()
#         with torch.no_grad():
#             out = model(x, adj_dict)['Euclidean']
#             # 验证集
#             val_pos_score = (out[pos_val[:, 0]] * out[pos_val[:, 1]]).sum(dim=1)
#             val_neg_score = (out[neg_val[:, 0]] * out[neg_val[:, 1]]).sum(dim=1)
#             val_scores = torch.cat([val_pos_score, val_neg_score]).cpu().numpy()
#             val_labels = np.concatenate([np.ones(len(val_pos_score)), np.zeros(len(val_neg_score))])
#             val_auc = roc_auc_score(val_labels, val_scores)
#             val_ap = average_precision_score(val_labels, val_scores)
#             # 测试集
#             test_pos_score = (out[pos_test[:, 0]] * out[pos_test[:, 1]]).sum(dim=1)
#             test_neg_score = (out[neg_test[:, 0]] * out[neg_test[:, 1]]).sum(dim=1)
#             test_scores = torch.cat([test_pos_score, test_neg_score]).cpu().numpy()
#             test_labels = np.concatenate([np.ones(len(test_pos_score)), np.zeros(len(test_neg_score))])
#             test_auc = roc_auc_score(test_labels, test_scores)
#             test_ap = average_precision_score(test_labels, test_scores)
#
#         epoch_time = time.time() - epoch_start
#         print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f} | "
#               f"Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f} | Time: {epoch_time:.2f}s")
#
#         if val_auc > best_val_auc:
#             best_val_auc = val_auc
#             best_result = dict(epoch=epoch, val_auc=val_auc, val_ap=val_ap,
#                                test_auc=test_auc, test_ap=test_ap)
#             patience_counter = 0
#         else:
#             patience_counter += 1
#         if patience_counter >= patience:
#             print(f"Early stop at epoch {epoch} (patience={patience})")
#             break
#
#     total_time = time.time() - start_time
#     print(f"\n==== Link Prediction Training Finished ====")
#     print(f"Best Val AUC: {best_result.get('val_auc', 0):.4f} | Best Val AP: {best_result.get('val_ap', 0):.4f} "
#           f"@ epoch {best_result.get('epoch', '-')}")
#     print(f"Test AUC at best Val: {best_result.get('test_auc', 0):.4f} | Test AP: {best_result.get('test_ap', 0):.4f}")
#     print(f"Total Training Time: {total_time:.2f}s\n")

def link_prediction_train(model, optimizer, data, relation_types, spaces,
                         train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg,
                         epochs, device, patience):
    """
    链路预测主训练流程。train_pos/neg等为外部传入的numpy数组，若为None则自动采样（按idx子集采样）。
    """
    adj_dict, adj_all = None, None
    model = model.to(device)
    # x = data['x'].to(device)
    # adj_dict = {r: data['adj_dict'][r].to(device) for r in relation_types}
    # num_nodes = x.size(0)
    # # 构建全图邻接矩阵
    # adj_all = merge_all_adj_dict(adj_dict, num_nodes)
    # 检查是不是只带有分割边（没有adj_dict）
    # ==== 1. 只要是 link_pred，不要任何邻接矩阵相关内容 ====
    if 'x' in data:
        x = data['x'].to(device)
        num_nodes = x.size(0)
    else:
        # 自动生成 one-hot 特征
        all_nodes = set()
        for split in ['train', 'valid', 'test']:
            for arr in [data.get(f"{split}_edges"), data.get(f"{split}_edges_neg")]:
                if arr is not None:
                    all_nodes |= set(np.unique(arr))
        num_nodes = max(all_nodes) + 1 if all_nodes else 0
        x = torch.eye(num_nodes)

    # ==== 2. 不用adj_dict, adj_all, merge_all_adj_dict！====

    # 优先用外部传入正负边，没有就从mat自动加载
    def to_numpy(x):
        if x is None: return None
        return np.array(x)

    # 需要采样时，仅在该子集上采样
    def sample_edges_by_idx(idx):
        mask = np.zeros(num_nodes, dtype=bool)
        mask[idx] = True
        sub_adj = adj_all[mask][:, mask]
        pos, neg = get_pos_neg_edges(sub_adj)
        # 注意返回原图节点编号
        pos = np.array([[idx[i], idx[j]] for i, j in pos])
        neg = np.array([[idx[i], idx[j]] for i, j in neg])
        return pos, neg

    # Train split
    if train_pos is None or train_neg is None:
        print("[Warning] train_edges 未发现，将自动从train_idx子集采样。")
        train_idx = data['train_idx'].cpu().numpy() if hasattr(data['train_idx'], 'cpu') else data['train_idx']
        train_pos, train_neg = sample_edges_by_idx(train_idx)
    else:
        train_pos, train_neg = to_numpy(train_pos), to_numpy(train_neg)

    # Val split
    if valid_pos is None or valid_neg is None:
        print("[Warning] valid_edges 未发现，将自动从valid_idx子集采样。")
        valid_idx = data['val_idx'].cpu().numpy() if hasattr(data['val_idx'], 'cpu') else data['val_idx']
        valid_pos, valid_neg = sample_edges_by_idx(valid_idx)
    else:
        valid_pos, valid_neg = to_numpy(valid_pos), to_numpy(valid_neg)

    # Test split
    if test_pos is None or test_neg is None:
        print("[Warning] test_edges 未发现，将自动从test_idx子集采样。")
        test_idx = data['test_idx'].cpu().numpy() if hasattr(data['test_idx'], 'cpu') else data['test_idx']
        test_pos, test_neg = sample_edges_by_idx(test_idx)
    else:
        test_pos, test_neg = to_numpy(test_pos), to_numpy(test_neg)

    # 确保shape安全（无论采样几条，始终二维）
    def ensure_2d(arr):
        arr = np.array(arr)
        if arr.ndim == 1:
            if arr.size == 0:
                return arr.reshape(0, 2)
            if arr.size == 2:
                return arr.reshape(1, 2)
            raise ValueError(f"非法shape: {arr.shape}")
        return arr

    train_pos = ensure_2d(train_pos)
    train_neg = ensure_2d(train_neg)
    valid_pos = ensure_2d(valid_pos)
    valid_neg = ensure_2d(valid_neg)
    test_pos = ensure_2d(test_pos)
    test_neg = ensure_2d(test_neg)

    # ====== 训练 ======
    best_val_auc = 0
    best_epoch = 0
    best_result = {}
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        out = model(x, adj_dict)['Euclidean']  # [N, D]


        # train
        train_pos_score = (out[train_pos[:, 0]] * out[train_pos[:, 1]]).sum(dim=1)
        train_neg_score = (out[train_neg[:, 0]] * out[train_neg[:, 1]]).sum(dim=1)
        train_scores = torch.cat([train_pos_score, train_neg_score])
        train_labels = torch.cat([torch.ones(len(train_pos_score)), torch.zeros(len(train_neg_score))]).to(device)
        loss = F.binary_cross_entropy_with_logits(train_scores, train_labels)
        loss.backward()
        optimizer.step()

        # 验证/测试
        model.eval()
        with torch.no_grad():
            out = model(x, adj_dict)['Euclidean']

            # 验证集
            val_pos_score = (out[valid_pos[:, 0]] * out[valid_pos[:, 1]]).sum(dim=1)
            val_neg_score = (out[valid_neg[:, 0]] * out[valid_neg[:, 1]]).sum(dim=1)
            val_scores = torch.cat([val_pos_score, val_neg_score]).cpu().numpy()
            val_labels = np.concatenate([np.ones(len(val_pos_score)), np.zeros(len(val_neg_score))])
            val_auc = roc_auc_score(val_labels, val_scores)
            val_ap = average_precision_score(val_labels, val_scores)
            # 测试集
            test_pos_score = (out[test_pos[:, 0]] * out[test_pos[:, 1]]).sum(dim=1)
            test_neg_score = (out[test_neg[:, 0]] * out[test_neg[:, 1]]).sum(dim=1)
            test_scores = torch.cat([test_pos_score, test_neg_score]).cpu().numpy()
            test_labels = np.concatenate([np.ones(len(test_pos_score)), np.zeros(len(test_neg_score))])
            test_auc = roc_auc_score(test_labels, test_scores)
            test_ap = average_precision_score(test_labels, test_scores)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f} | "
              f"Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f} | Time: {epoch_time:.2f}s")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_result = dict(epoch=epoch, val_auc=val_auc, val_ap=val_ap,
                               test_auc=test_auc, test_ap=test_ap)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch} (patience={patience})")
            break

    total_time = time.time() - start_time
    print(f"\n==== Link Prediction Training Finished ====")
    print(f"Best Val AUC: {best_result.get('val_auc', 0):.4f} | Best Val AP: {best_result.get('val_ap', 0):.4f} "
          f"@ epoch {best_result.get('epoch', '-')}")
    print(f"Test AUC at best Val: {best_result.get('test_auc', 0):.4f} | Test AP: {best_result.get('test_ap', 0):.4f}")
    print(f"Total Training Time: {total_time:.2f}s\n")


def train(model, optimizer, data, relation_types, spaces,
    train_pos=None, train_neg=None, valid_pos=None, valid_neg=None, test_pos=None, test_neg=None,
    epochs=100, device='cpu', task='node_cls', patience=100,
          lambda_mutual=1.0, tau=2.0,
          ):
    if task == 'node_cls':
        train_node_classification(model, optimizer, data, relation_types, spaces,
            epochs, device, patience, lambda_mutual, tau)
    elif task == 'link_pred':
        link_prediction_train(model, optimizer, data, relation_types, spaces,
            train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg,
            epochs, device, patience)
    else:
        raise ValueError("Unknown task. Supported: node_cls, link_pred")
