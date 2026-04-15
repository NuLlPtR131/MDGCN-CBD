import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, auc
import torch.nn.functional as F

'''
def load_training_data(f_name):
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            # print(words)
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_edges.append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('total training nodes: ' + str(len(all_nodes)))
    # print('Finish loading training data')
    return edge_data_by_type


def load_testing_data(f_name):
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    # print('Finish loading testing data')
    return true_edge_data_by_type, false_edge_data_by_type


def get_score(local_model, node1, node2):
    """
    Calculate embedding similarity
    """
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        if type(vector1) != np.ndarray:
            vector1 = vector1.toarray()[0]
            vector2 = vector2.toarray()[0]

        return np.dot(vector1, vector2)
        # return np.dot(vector1, vector2) / ((np.linalg.norm(vector1) * np.linalg.norm(vector2) + 0.00000000000000001))
    except Exception as e:
        pass


def link_prediction_evaluate(model, true_edges, false_edges):
    """
    Link prediction process
    """

    true_list = list()
    prediction_list = list()
    true_num = 0

    # Calculate the similarity score of positive sample embedding
    for edge in true_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    # Calculate the the similarity score of negative sample embedding
    for edge in false_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    # Determine the positive and negative sample threshold
    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    # Compare the similarity score with the threshold to predict whether the connection exists
    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)


def predict_model(model, file_name, feature, A, eval_type, node_matching):
    """
    Link prediction training proces
    """

    training_data_by_type = load_training_data(file_name + '/train.txt')
    # train_true_data_by_edge, train_false_data_by_edge = load_testing_data(file_name + '/train.txt')
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test.txt')

    network_data = training_data_by_type
    edge_types = list(network_data.keys())  # ['1', '2', '3', '4', 'Base']
    edge_type_count = len(edge_types) - 1
    # edge_type_count = len(eval_type) - 1s

    device = torch.device('cpu')

    aucs, f1s, prs = [], [], []

    for _ in range(1):
        for iter_ in range(500):
            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
            emb = model(feature, A)  # 输入 A 必须是邻接矩阵列表

            # ==== 关键修改1：确保嵌入维度与 reshape 一致 ====
            # 检查嵌入维度是否为200（需与模型初始化参数 out=200 对应）
            assert emb.shape[1] == 200, f"模型输出维度应为200，当前为{emb.shape[1]}"

            # ==== 关键修改2：适配嵌入收集逻辑 ====
            emb_true_first = []
            emb_true_second = []
            emb_false_first = []
            emb_false_second = []

            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    true_edges = valid_true_data_by_edge[edge_types[i]]
                    false_edges = valid_false_data_by_edge[edge_types[i]]

                # 确保节点索引在嵌入矩阵范围内
                for edge in true_edges:
                    node1 = int(edge[0])
                    node2 = int(edge[1])
                    if node1 < emb.shape[0] and node2 < emb.shape[0]:  # 防止索引越界
                        emb_true_first.append(emb[node1])
                        emb_true_second.append(emb[node2])

                for edge in false_edges:
                    node1 = int(edge[0])
                    node2 = int(edge[1])
                    if node1 < emb.shape[0] and node2 < emb.shape[0]:
                        emb_false_first.append(emb[node1])
                        emb_false_second.append(emb[node2])

            # ==== 关键修改3：消除潜在的维度错误 ====
            # 使用 torch.stack 代替 torch.cat 避免维度不匹配
            if len(emb_true_first) > 0:
                emb_true_first = torch.stack(emb_true_first).reshape(-1, 200)
                emb_true_second = torch.stack(emb_true_second).reshape(-1, 200)
            else:
                emb_true_first = torch.zeros(0, 200)
                emb_true_second = torch.zeros(0, 200)

            if len(emb_false_first) > 0:
                emb_false_first = torch.stack(emb_false_first).reshape(-1, 200)
                emb_false_second = torch.stack(emb_false_second).reshape(-1, 200)
            else:
                emb_false_first = torch.zeros(0, 200)
                emb_false_second = torch.zeros(0, 200)

            T1 = emb_true_first @ emb_true_second.T
            T2 = -(emb_false_first @ emb_false_second.T)

            pos_out = torch.diag(T1)
            neg_out = torch.diag(T2)

            loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
            loss = loss.requires_grad_()

            opt.zero_grad()
            loss.backward()
            opt.step()

            td = model(feature, A).detach().numpy()
            final_model = {}
            try:
                if node_matching:
                    # 假设 td 是密集矩阵，且第一列为节点ID
                    for i in range(td.shape[0]):
                        node_id = str(int(td[i, 0]))  # 第一列为节点ID
                        final_model[node_id] = td[i, 1:]  # 后续列为嵌入向量
                else:
                    # 直接按顺序映射节点索引
                    for i in range(td.shape[0]):
                        final_model[str(i)] = td[i]
            except:
                # 处理稀疏矩阵情况（根据实际需求调整）
                td = td.tocsr()
                if node_matching:
                    for i in range(td.shape[0]):
                        node_id = str(int(td[i, 0]))
                        final_model[node_id] = td[i, 1:].toarray().flatten()
                else:
                    for i in range(td.shape[0]):
                        final_model[str(i)] = td[i].toarray().flatten()
                        
            train_aucs, train_f1s, train_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    train_auc, triain_f1, train_pr = link_prediction_evaluate(final_model,
                                                                              valid_true_data_by_edge[edge_types[i]],
                                                                              valid_false_data_by_edge[edge_types[i]])
                    train_aucs.append(train_auc)
                    train_f1s.append(triain_f1)
                    train_prs.append(train_pr)

                    test_auc, test_f1, test_pr = link_prediction_evaluate(final_model,
                                                                          testing_true_data_by_edge[edge_types[i]],
                                                                          testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(test_auc)
                    test_f1s.append(test_f1)
                    test_prs.append(test_pr)

            print("{}\t{:.4f}\tweight_b:{}".format(iter_ + 1, loss.item(), model.weight_b))
            print("train_auc:{:.4f}\ttrain_f1:{:.4f}\ttrain_pr:{:.4f}".format(np.mean(train_aucs), np.mean(train_f1s),
                                                                              np.mean(train_prs)))
            print("test_auc:{:.4f}\ttest_f1:{:.4f}\ttest_pr:{:.4f}".format(np.mean(test_aucs), np.mean(test_f1s),
                                                                           np.mean(test_prs)))
            aucs.append(np.mean(test_aucs))
            f1s.append(np.mean(test_f1s))
            prs.append(np.mean(test_prs))

    max_iter = aucs.index(max(aucs))

    return aucs[max_iter], f1s[max_iter], prs[max_iter]





'''
# V2.0 效果已优于原代码

from collections import defaultdict


def load_training_data(f_name, device):
    # 生成节点映射字典
    node_mapping = dict()
    current_idx = 0
    
    edge_data_by_type = defaultdict(list)
    all_edges = []
    
    with open(f_name, 'r') as f:
        for line in f:
            words = line.strip().split()
            if len(words) < 3:
                continue
                
            edge_type, x, y = words[0], words[1], words[2]
            
            # 构建节点映射
            if x not in node_mapping:
                node_mapping[x] = current_idx
                current_idx += 1
            if y not in node_mapping:
                node_mapping[y] = current_idx 
                current_idx += 1
                
            # 转换为数字索引
            x_idx = node_mapping[x]
            y_idx = node_mapping[y]
            
            edge_data_by_type[edge_type].append((x_idx, y_idx))
            all_edges.append((x_idx, y_idx))
    
    # 转换为Tensor并发送到指定设备
    edge_pairs = torch.LongTensor(all_edges).to(device)
    
    print(f'Total training nodes: {len(node_mapping)}')
    print(f'Total training edges: {len(all_edges)}')
    
    return edge_data_by_type, edge_pairs



def load_testing_data(f_name):
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    # print('Finish loading testing data')
    return true_edge_data_by_type, false_edge_data_by_type


def get_score(local_model, node1, node2):
    """
    Calculate embedding similarity
    """
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        if type(vector1) != np.ndarray:
            vector1 = vector1.toarray()[0]
            vector2 = vector2.toarray()[0]

        # return np.dot(vector1, vector2)
        return np.dot(vector1, vector2) / ((np.linalg.norm(vector1) * np.linalg.norm(vector2) + 0.00000000000000001))
    except Exception as e:
        pass


def link_prediction_evaluate(model, true_edges, false_edges):
    """
    Link prediction process
    """

    true_list = list()
    prediction_list = list()
    true_num = 0

    # Calculate the similarity score of positive sample embedding
    for edge in true_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    # Calculate the the similarity score of negative sample embedding
    for edge in false_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0])), str(int(edge[1])))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    # Determine the positive and negative sample threshold
    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    # Compare the similarity score with the threshold to predict whether the connection exists
    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)

'''
def predict_model(model, file_name, feature, A, eval_type, node_matching):
    """
    Link prediction training proces
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_data_by_type, edge_pairs = load_training_data(
        file_name + '/train.txt',device=device)
    # train_true_data_by_edge, train_false_data_by_edge = load_testing_data(file_name + '/train.txt')
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test.txt')

    network_data = training_data_by_type
    edge_types = list(network_data.keys())  # ['1', '2', '3', '4', 'Base']
    edge_type_count = len(edge_types) - 1
    # edge_type_count = len(eval_type) - 1s

    

    aucs, f1s, prs = [], [], []


    alpha = 0.1  # 特有特征损失权重
    beta = 0.1   # 共有特征损失权重
    gamma = 0.1  # 协作损失权重
    lambda1 = 0.1  # 正交约束
    lambda2 = 0.1  # 低秩约束

    for _ in range(1):
        for iter_ in range(500):
            
            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
            emb, specifics, commons, collab = model(feature, A)
            # ==== 关键修改1：确保嵌入维度与 reshape 一致 ====
            # 检查嵌入维度是否为200（需与模型初始化参数 out=200 对应）
            assert emb.shape[1] == 200, f"模型输出维度应为200，当前为{emb.shape[1]}"

            # ==== 关键修改2：适配嵌入收集逻辑 ====
            emb_true_first = []
            emb_true_second = []
            emb_false_first = []
            emb_false_second = []

            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    true_edges = valid_true_data_by_edge[edge_types[i]]
                    false_edges = valid_false_data_by_edge[edge_types[i]]

                # 确保节点索引在嵌入矩阵范围内
                for edge in true_edges:
                    node1 = int(edge[0])
                    node2 = int(edge[1])
                    if node1 < emb.shape[0] and node2 < emb.shape[0]:  # 防止索引越界
                        emb_true_first.append(emb[node1])
                        emb_true_second.append(emb[node2])

                for edge in false_edges:
                    node1 = int(edge[0])
                    node2 = int(edge[1])
                    if node1 < emb.shape[0] and node2 < emb.shape[0]:
                        emb_false_first.append(emb[node1])
                        emb_false_second.append(emb[node2])

            # ==== 关键修改3：消除潜在的维度错误 ====
            # 使用 torch.stack 代替 torch.cat 避免维度不匹配

            
            #if len(emb_true_first) > 0:
            #    emb_true_first = torch.stack(emb_true_first).reshape(-1, 200)
            #    emb_true_second = torch.stack(emb_true_second).reshape(-1, 200)
            #else:
            #    emb_true_first = torch.zeros(0, 200)
            #    emb_true_second = torch.zeros(0, 200)

            #if len(emb_false_first) > 0:
            #    emb_false_first = torch.stack(emb_false_first).reshape(-1, 200)
            #    emb_false_second = torch.stack(emb_false_second).reshape(-1, 200)
            #else:
            #    emb_false_first = torch.zeros(0, 200)
            #    emb_false_second = torch.zeros(0, 200)
            
            emb_true_first = torch.cat(emb_true_first).reshape(-1, 200)
            emb_true_second = torch.cat(emb_true_second).reshape(-1, 200)
            emb_false_first = torch.cat(emb_false_first).reshape(-1, 200)
            emb_false_second = torch.cat(emb_false_second ).reshape(-1, 200)


            T1 = emb_true_first @ emb_true_second.T
            T2 = -(emb_false_first @ emb_false_second.T)

            pos_out = torch.diag(T1)
            neg_out = torch.diag(T2)

            main_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
            #==== 新增损失项计算 ====#
            # 1. 特有特征损失
            spec_loss = 0
            for sp in specifics:
                # 类内相似性
                intra_loss = torch.norm(sp[edge_pairs[:,0]] - sp[edge_pairs[:,1]], p=2)
                spec_loss += intra_loss
                
            # 正交约束
            ortho_loss = 0
            for i in range(len(specifics)):
                for j in range(i+1, len(specifics)):
                    ortho_loss += torch.norm(torch.mm(specifics[i], specifics[j].t()), p='fro')
            spec_loss += lambda1 * ortho_loss
            
            # 2. 共有特征损失
            comm_loss = 0
            for i in range(len(commons)):
                for j in range(i+1, len(commons)):
                    comm_loss += torch.norm(commons[i] - commons[j], p='fro')
                    
            # 低秩约束
            rank_loss = sum([torch.norm(c, p='nuc') for c in commons])
            comm_loss += lambda2 * rank_loss
            
            # 3. 协作特征损失
            # 互信息估计（简化实现）
            mi_loss = 0
            for sp in specifics:
                mi_loss -= torch.mean(torch.log_softmax(collab, dim=1) * torch.softmax(sp, dim=1))
                
            # 对抗损失
            real_labels = torch.ones(collab.size(0), 1, device=device)
            fake_labels = torch.zeros(commons[0].size(0), 1, device=device)
            adv_loss = F.binary_cross_entropy(model.discriminator(collab), real_labels) + \
                      F.binary_cross_entropy(model.discriminator(commons[0].detach()), fake_labels)
            
            collab_loss = mi_loss + gamma * adv_loss
            
            #==== 总损失 ====#
            total_loss = main_loss  + alpha*spec_loss + beta*comm_loss + gamma*collab_loss

            # 新加的
            total_loss.requires_grad()

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            emb, _, _, _ = model(feature, A)  # 仅提取嵌入部分
            td = emb.detach().cpu().numpy()    # 移动到 CPU 并转为 numpy
            final_model = {}
            try:
                if node_matching:
                    # 假设 td 是密集矩阵，且第一列为节点ID
                    for i in range(td.shape[0]):
                        node_id = str(int(td[i, 0]))  # 第一列为节点ID
                        final_model[node_id] = td[i, 1:]  # 后续列为嵌入向量
                else:
                    # 直接按顺序映射节点索引
                    for i in range(td.shape[0]):
                        final_model[str(i)] = td[i]
            except:
                # 处理稀疏矩阵情况（根据实际需求调整）
                td = td.tocsr()
                if node_matching:
                    for i in range(td.shape[0]):
                        node_id = str(int(td[i, 0]))
                        final_model[node_id] = td[i, 1:].toarray().flatten()
                else:
                    for i in range(td.shape[0]):
                        final_model[str(i)] = td[i].toarray().flatten()
                        
            train_aucs, train_f1s, train_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    train_auc, triain_f1, train_pr = link_prediction_evaluate(final_model,
                                                                              valid_true_data_by_edge[edge_types[i]],
                                                                              valid_false_data_by_edge[edge_types[i]])
                    train_aucs.append(train_auc)
                    train_f1s.append(triain_f1)
                    train_prs.append(train_pr)

                    test_auc, test_f1, test_pr = link_prediction_evaluate(final_model,
                                                                          testing_true_data_by_edge[edge_types[i]],
                                                                          testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(test_auc)
                    test_f1s.append(test_f1)
                    test_prs.append(test_pr)

            print("{}\t{:.4f}\tweight_b:{}".format(iter_ + 1, total_loss.item(), model.weight_b))
            print("train_auc:{:.4f}\ttrain_f1:{:.4f}\ttrain_pr:{:.4f}".format(np.mean(train_aucs), np.mean(train_f1s),
                                                                              np.mean(train_prs)))
            print("test_auc:{:.4f}\ttest_f1:{:.4f}\ttest_pr:{:.4f}".format(np.mean(test_aucs), np.mean(test_f1s),
                                                                           np.mean(test_prs)))
            print("loss:{:.4f}".format(total_loss.item()))
            aucs.append(np.mean(test_aucs))
            f1s.append(np.mean(test_f1s))
            prs.append(np.mean(test_prs))

    max_iter = aucs.index(max(aucs))

    return aucs[max_iter], f1s[max_iter], prs[max_iter]

'''





def predict_model(model, file_name, feature, A, eval_type, node_matching):
    """
    Link prediction training proces
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_data_by_type, edge_pairs = load_training_data(
        file_name + '/train.txt',device=device)
    #train_true_data_by_edge, train_false_data_by_edge = load_testing_data(file_name + '/train.txt')
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test.txt')

    network_data = training_data_by_type
    edge_types = list(network_data.keys())  # ['1', '2', '3', '4', 'Base']
    edge_type_count = len(edge_types) - 1
    # edge_type_count = len(eval_type) - 1s

    

    aucs, f1s, prs = [], [], []


    alpha = 0.1  # 特有特征损失权重
    beta = 0.1   # 共有特征损失权重
    gamma = 0.1  # 协作损失权重
    lambda1 = 0.05  # 正交约束
    lambda2 = 0.01  # 低秩约束

    for _ in range(1):
        for iter_ in range(500):
            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
            emb, specifics, commons, collab = model(feature, A)
            # ==== 关键修改1：确保嵌入维度与 reshape 一致 ====
            # 检查嵌入维度是否为200（需与模型初始化参数 out=200 对应）
            assert emb.shape[1] == 200, f"模型输出维度应为200，当前为{emb.shape[1]}"

            # ==== 关键修改2：适配嵌入收集逻辑 ====
            emb_true_first = []
            emb_true_second = []
            emb_false_first = []
            emb_false_second = []

            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    true_edges = valid_true_data_by_edge[edge_types[i]]
                    false_edges = valid_false_data_by_edge[edge_types[i]]

                # 确保节点索引在嵌入矩阵范围内
                for edge in true_edges:
                    node1 = int(edge[0])
                    node2 = int(edge[1])
                    if node1 < emb.shape[0] and node2 < emb.shape[0]:  # 防止索引越界
                        emb_true_first.append(emb[node1])
                        emb_true_second.append(emb[node2])

                for edge in false_edges:
                    node1 = int(edge[0])
                    node2 = int(edge[1])
                    if node1 < emb.shape[0] and node2 < emb.shape[0]:
                        emb_false_first.append(emb[node1])
                        emb_false_second.append(emb[node2])

            # ==== 关键修改3：消除潜在的维度错误 ====
            # 使用 torch.stack 代替 torch.cat 避免维度不匹配
            if len(emb_true_first) > 0:
                emb_true_first = torch.stack(emb_true_first).reshape(-1, 200)
                emb_true_second = torch.stack(emb_true_second).reshape(-1, 200)
            else:
                emb_true_first = torch.zeros(0, 200)
                emb_true_second = torch.zeros(0, 200)

            if len(emb_false_first) > 0:
                emb_false_first = torch.stack(emb_false_first).reshape(-1, 200)
                emb_false_second = torch.stack(emb_false_second).reshape(-1, 200)
            else:
                emb_false_first = torch.zeros(0, 200)
                emb_false_second = torch.zeros(0, 200)

            T1 = emb_true_first @ emb_true_second.T
            T2 = -(emb_false_first @ emb_false_second.T)

            pos_out = torch.diag(T1)
            neg_out = torch.diag(T2)

            main_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
            #==== 新增损失项计算 ====#
            # 1. 特有特征损失
            spec_loss = 0
            for sp in specifics:
                # 类内相似性
                intra_loss = torch.norm(sp[edge_pairs[:,0]] - sp[edge_pairs[:,1]], p=2)
                spec_loss += intra_loss
                
            # 正交约束
            ortho_loss = 0
            for i in range(len(specifics)):
                for j in range(i+1, len(specifics)):
                    ortho_loss += torch.norm(torch.mm(specifics[i], specifics[j].t()), p='fro')
            spec_loss += lambda1 * ortho_loss
            
            # 2. 共有特征损失
            comm_loss = 0
            for i in range(len(commons)):
                for j in range(i+1, len(commons)):
                    comm_loss += torch.norm(commons[i] - commons[j], p='fro')
                    
            # 低秩约束
            rank_loss = sum([torch.norm(c, p='nuc') for c in commons])
            comm_loss += lambda2 * rank_loss
            
            # 3. 协作特征损失
            # 互信息估计（简化实现）
            mi_loss = 0
            for sp in specifics:
                mi_loss -= torch.mean(torch.log_softmax(collab, dim=1) * torch.softmax(sp, dim=1))
                
            # 对抗损失
            real_labels = torch.ones(collab.size(0), 1, device=device)
            fake_labels = torch.zeros(commons[0].size(0), 1, device=device)
            adv_loss = F.binary_cross_entropy(model.discriminator(collab), real_labels) + \
                      F.binary_cross_entropy(model.discriminator(commons[0].detach()), fake_labels)
            
            collab_loss = mi_loss + gamma * adv_loss
            
            #==== 总损失 ====#
            total_loss = main_loss + alpha*spec_loss + beta*comm_loss + gamma*collab_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()
            emb, _, _, _ = model(feature, A)  # 仅提取嵌入部分
            td = emb.detach().cpu().numpy()    # 移动到 CPU 并转为 numpy
            final_model = {}
            try:
                if node_matching:
                    # 假设 td 是密集矩阵，且第一列为节点ID
                    for i in range(td.shape[0]):
                        node_id = str(int(td[i, 0]))  # 第一列为节点ID
                        final_model[node_id] = td[i, 1:]  # 后续列为嵌入向量
                else:
                    # 直接按顺序映射节点索引
                    for i in range(td.shape[0]):
                        final_model[str(i)] = td[i]
            except:
                # 处理稀疏矩阵情况（根据实际需求调整）
                td = td.tocsr()
                if node_matching:
                    for i in range(td.shape[0]):
                        node_id = str(int(td[i, 0]))
                        final_model[node_id] = td[i, 1:].toarray().flatten()
                else:
                    for i in range(td.shape[0]):
                        final_model[str(i)] = td[i].toarray().flatten()
                        
            train_aucs, train_f1s, train_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if eval_type == 'all' or edge_types[i] in eval_type.split(','):
                    train_auc, triain_f1, train_pr = link_prediction_evaluate(final_model,
                                                                              valid_true_data_by_edge[edge_types[i]],
                                                                              valid_false_data_by_edge[edge_types[i]])
                    train_aucs.append(train_auc)
                    train_f1s.append(triain_f1)
                    train_prs.append(train_pr)

                    test_auc, test_f1, test_pr = link_prediction_evaluate(final_model,
                                                                          testing_true_data_by_edge[edge_types[i]],
                                                                          testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(test_auc)
                    test_f1s.append(test_f1)
                    test_prs.append(test_pr)

            print("{}\t{:.4f}\tweight_b:{}".format(iter_ + 1, total_loss.item(), model.weight_b))
            print("train_auc:{:.4f}\ttrain_f1:{:.4f}\ttrain_pr:{:.4f}".format(np.mean(train_aucs), np.mean(train_f1s),
                                                                              np.mean(train_prs)))
            print("test_auc:{:.4f}\ttest_f1:{:.4f}\ttest_pr:{:.4f}".format(np.mean(test_aucs), np.mean(test_f1s),
                                                                           np.mean(test_prs)))
            aucs.append(np.mean(test_aucs))
            f1s.append(np.mean(test_f1s))
            prs.append(np.mean(test_prs))

    max_iter = aucs.index(max(aucs))

    return aucs[max_iter], f1s[max_iter], prs[max_iter]
