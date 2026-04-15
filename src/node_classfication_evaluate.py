# 文件: node_classfication_evaluate.py
import numpy as np
import scipy.io as sio
import pickle as pkl
import torch.nn as nn
from sklearn.metrics import f1_score
import time

import torch
import torch.nn.functional as F
from src.logreg import LogReg

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def load_data(dataset, datasetfile_type):
    """"Get the label of node classification, training set, verification machine and test set"""
    if datasetfile_type == 'mat':
        data = sio.loadmat('data/{}.mat'.format(dataset))
    else:
        data = pkl.load(open('data/{}.pkl'.format(dataset), "rb"))
    try:
        labels = data['label']
    except:
        labels = data['labelmat']

    idx_train = data['train_idx'].ravel()
    try:
        idx_val = data['valid_idx'].ravel()
    except:
        idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    return labels, idx_train.astype(np.int32) - 1, idx_val.astype(np.int32) - 1, idx_test.astype(np.int32) - 1

def loss_specific_divergence(specific_list):
    """
    强化不同路径特有表示之间的区分度
    使用路径间 pairwise cosine similarity
    """
    loss = 0
    num_paths = len(specific_list)
    for i in range(num_paths):
        for j in range(i+1, num_paths):
            sim = F.cosine_similarity(specific_list[i], specific_list[j], dim=1)  # [N]
            loss += abs(torch.mean(sim))
    # 越小越好，取负值，作为 loss 加入
    return loss / (num_paths * (num_paths - 1) / 2)


def loss_shared_variance(shared_list):
    """
    共有特征应当彼此接近，我们以 variance 来约束
    """
    mean_shared = torch.mean(torch.stack(shared_list), dim=0)  # [N, d]
    loss = 0
    for feat in shared_list:
        loss += F.mse_loss(feat, mean_shared)
    return loss / len(shared_list)

def loss_collab_contrastive(collab_feat, raw_feat, temperature=0.3):
    """
    协作特征与原始特征间的互信息最大化
    """
    collab_feat = F.normalize(collab_feat, dim=1)
    raw_feat = F.normalize(raw_feat, dim=1)

    pos_score = torch.sum(collab_feat * raw_feat, dim=1) / temperature  # [N]

    # 负样本：打乱 raw_feat
    raw_feat_perm = raw_feat[torch.randperm(raw_feat.size(0))]
    neg_score = torch.sum(collab_feat * raw_feat_perm, dim=1) / temperature

    logits = torch.cat([pos_score.unsqueeze(1), neg_score.unsqueeze(1)], dim=1)  # [N, 2]
    labels = torch.zeros(collab_feat.size(0), dtype=torch.long).to(collab_feat.device)

    return F.cross_entropy(logits, labels)

def total_decoupling_loss(specific_list, shared_list, collab_feat, raw_feat,
                          alpha=2.0, beta=0.8, gamma=0.8):
    loss_spec = loss_specific_divergence(specific_list)
    loss_shared = loss_shared_variance(shared_list)
    loss_collab = loss_collab_contrastive(collab_feat, raw_feat)

    return alpha * loss_spec+ gamma * loss_collab
 #+ beta * loss_shared 

def node_classification_evaluate(model, feature, A, file_name, file_type, device, isTest=True):
    """Node classification training process"""

    embeds, specific_list, shared_feat, collab_feat,raw_feat = model(feature, A)
    embeds = embeds.unsqueeze(0)
    labels, idx_train, idx_val, idx_test = load_data(file_name, file_type)

    try:
        labels = labels.todense()
    except:
        pass
    labels = labels.astype(np.int16)
    
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]
    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []

    for _ in range(1):
        log = LogReg(hid_units, nb_classes)
        # 目前最好结果的学习率是 0.05
        opt = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.05}, {'params': log.parameters()}], lr=0.005, weight_decay=0.0005)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        starttime = time.time()
        for iter_ in range(200):
            # train
            model.train()#新加的可以删
            log.train()
            embeds, specific_list, shared_feat, collab_feat,raw_feat = model(feature, A)
            # print(embeds)
            embeds = embeds.unsqueeze(0) 
            train_embs = embeds[0, idx_train]
            val_embs = embeds[0, idx_val]
            test_embs = embeds[0, idx_test]

            
            opt.zero_grad()

            logits = log(train_embs)
            loss_cls = xent(logits, train_lbls)

            loss_decouple = total_decoupling_loss(
                specific_list=[feat[idx_train] for feat in specific_list],
                shared_list=[feat[idx_train] for feat in shared_feat],         
                collab_feat=collab_feat[idx_train],
                raw_feat=raw_feat[idx_train]                           
            )


            
            # 总损失
            loss = loss_cls +0.01* loss_decouple
            loss.backward()
            opt.step()

            #要删
            model.eval()                  # 关闭 Dropout，固定 BatchNorm 的 running stats
            log.eval()

            logits_val = log(val_embs)
            preds = torch.argmax(logits_val, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')

            print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(iter_ + 1, loss.item(), val_acc, val_f1_macro,
                                                              val_f1_micro))
            print("weight_b:{}".format(model.weight_b))

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits_test = log(test_embs)
            preds = torch.argmax(logits_test, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')
            print("test_f1-ma: {:.4f}\ttest_f1-mi: {:.4f}".format(test_f1_macro, test_f1_micro))

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)

        endtime = time.time()

        print('time: {:.10f}'.format(endtime - starttime))

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])
        model.eval()  # 确保模型处于评估模式
        with torch.no_grad():
            final_embeds, _, _, _, _ = model(feature, A)  # 获取最终嵌入（不包括unsqueeze）

        # 保存节点嵌入到文件（使用torch.save，便于后续加载）
        #torch.save(final_embeds, 'data/DBLP_embedding/dblp_final_embeddings.pt')

        # 保存节点标签（类别索引形式，便于可视化）
        all_labels = torch.argmax(labels[0], dim=1)  # [N] 的类别索引
        #torch.save(all_labels, 'data/DBLP_embedding/dblp_labels.pt')

        #print("最终节点嵌入已保存到 'data/DBLP_embedding/dblp_final_embeddings.pt'")
        #print("节点标签已保存到 'data/DBLP_embedding/dblp_labels.pt'")

        # 新增: 在循环后进行可视化
        model.eval()
        with torch.no_grad():
            final_embeds, specific_list_final, shared_list, H_col, H_raw = model(feature, A)

        # 提取三通道特征 (假设 out_dim = d)
        # Specific: 使用 H_sp_fused (从代码中复制逻辑)
        W_tilde_prob = F.softmax(model.weight_b, dim=0).view(1, -1, 1)  # 注意: 使用 model.weight_b 或 W_tilde
        stacked_specific = torch.stack(specific_list_final).transpose(0, 1)
        specific_feat = torch.sum(stacked_specific * W_tilde_prob, dim=1).cpu().numpy()  # [N, d]

        # Shared: 使用 H_sh (平均 shared_list)
        shared_feat = torch.mean(torch.stack(shared_list), dim=0).cpu().numpy()  # [N, d]

        # Collaborative: H_col
        collab_feat = H_col.cpu().numpy()  # [N, d]

        # 所有节点标签 (numpy)
        all_labels_np = torch.argmax(labels[0], dim=1).cpu().numpy()  # [N]

        # 降维函数 (极限优化: 先 PCA 50D, 再 t-SNE)
        def reduce_dim(feat, method='tsne', n_components=2):
            if feat.shape[0] > 5000:  # 大数据集预降维
                pca = PCA(n_components=50)
                feat = pca.fit_transform(feat)
            if method == 'tsne':
                tsne = TSNE(n_components=n_components, perplexity=30, learning_rate=200, n_iter=1000, random_state=42, n_jobs=-1)  # 全核并行
                return tsne.fit_transform(feat)
            elif method == 'pca':
                pca = PCA(n_components=n_components)
                return pca.fit_transform(feat)

        # 降维三通道 (2D)
        specific_2d = reduce_dim(specific_feat)
        shared_2d = reduce_dim(shared_feat)
        collab_2d = reduce_dim(collab_feat)

        # 计算 Silhouette 分数 (量化聚类质量)
        sil_specific = silhouette_score(specific_2d, all_labels_np)
        sil_shared = silhouette_score(shared_2d, all_labels_np)
        sil_collab = silhouette_score(collab_2d, all_labels_np)
        print(f"Silhouette Scores: Specific={sil_specific:.4f}, Shared={sil_shared:.4f}, Collab={sil_collab:.4f}")

        # 可视化函数
        def plot_channel(embed_2d, labels_np, title, save_path):
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(embed_2d[:, 0], embed_2d[:, 1], c=labels_np, cmap='tab10', s=10, alpha=0.7)
            plt.colorbar(scatter)
            plt.title(title)
            plt.xlabel('Dim 1')
            plt.ylabel('Dim 2')
            plt.savefig(save_path)
            plt.close()

        # 绘制独立图
        plot_channel(specific_2d, all_labels_np, 'Specific Channel Visualization', 'data/DBLP_visualization/specific.png')
        plot_channel(shared_2d, all_labels_np, 'Shared Channel Visualization', 'data/DBLP_visualization/shared.png')
        plot_channel(collab_2d, all_labels_np, 'Collaborative Channel Visualization', 'data/DBLP_visualization/collab.png')

        # 叠加图 (所有通道在同一图中，用不同标记)
        plt.figure(figsize=(10, 8))
        plt.scatter(specific_2d[:, 0], specific_2d[:, 1], c=all_labels_np, marker='o', s=10, alpha=0.5, label='Specific')
        plt.scatter(shared_2d[:, 0], shared_2d[:, 1], c=all_labels_np, marker='x', s=10, alpha=0.5, label='Shared')
        plt.scatter(collab_2d[:, 0], collab_2d[:, 1], c=all_labels_np, marker='^', s=10, alpha=0.5, label='Collab')
        plt.colorbar()
        plt.legend()
        plt.title('Overlay of Three Channels')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.savefig('data/DBLP_visualization/overlay.png')
        plt.close()

        print("可视化已保存到 'data/DBLP_visualization/' 目录下。")


    if isTest:
        print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
                                                                                                np.std(macro_f1s),
                                                                                                np.mean(micro_f1s),
                                                                                                np.std(micro_f1s)))
    else:
        return np.mean(macro_f1s), np.mean(micro_f1s)

    return np.mean(macro_f1s), np.mean(micro_f1s)