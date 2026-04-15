import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

from src.Decoupling_matrix_aggregation import adj_matrix_weight_merge
import scipy.sparse as sp
import numpy as np


class GraphConvolution(nn.Module):
    """
    Standard Graph Convolutional layer, consistent with the original code.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Ensure input is float, as in original code
        input = input.float()
        support = torch.mm(input, self.weight)
        # Use torch.spmm for sparse matrix multiplication
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MHGCN(nn.Module):
    """
    Meta-Path Disentangled Graph Convolutional Network (MPDGCN).
    This implementation is based on the provided research paper.
    """
    def __init__(self, nfeat, nhid, out, dropout, num_paths=2):
        super(MHGCN, self).__init__()

        self.dropout = dropout
        self.num_paths = 3

        # --- Hyperparameters from the paper and original code ---
        #self.alpha =0.05 # Blending factor for path weights
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.walk_steps = 13 # L, number of steps in the random walk
        self.lambda_rw = 0.2  # λ, restart probability in the random walk 
        self.beta_drop = 0.1  # Dropout probability for path weights (p_beta) 
        self.tau = nn.Parameter(torch.tensor(0.5))  # Learnable temperature for similarity 

        # --- I. Meta-Path Specific and Shared Graph Encoding ---
        # 1) Meta-Path Specific GCNs (2-layer) 
        self.spec_gc1s = nn.ModuleList([GraphConvolution(nfeat, nhid) for _ in range(self.num_paths)])
        self.spec_gc2s = nn.ModuleList([GraphConvolution(nhid, out) for _ in range(self.num_paths)])

        # 2) Meta-Path Shared GCNs (2-layer) 
        self.shared_gc1 = GraphConvolution(nfeat, nhid)
        self.shared_gc2 = GraphConvolution(nhid, out)

        # --- II. Markov-Enhanced Path Attention Modeling ---
        # Learnable score vector 'b' for the random walk process 
        self.weight_b = Parameter(torch.FloatTensor(self.num_paths, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight_b)
        
        # --- III. Collaborative Feature Fusion ---
        # MLP to capture nonlinear cross-path interactions 
        self.collab_mlp = nn.Sequential(
            nn.Linear(out * self.num_paths, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, out)
        )
        
        # --- Raw Feature Stream (H_raw) ---
        # GCN stream for raw features, structure based on original code's U_origin 
        self.raw_gc1 = GraphConvolution(nfeat, out)
        self.raw_gc2 = GraphConvolution(out, out)

        # --- Final Projection Layer ---
        # Concatenates the four feature channels: H_sp_fused, H_sh, H_col, H_raw 
        self.proj = nn.Linear(out * 4, out)


    def _markov_enhanced_path_attention(self, specific_list_final):
        """
        Calculates dynamic path weights using a Markov random walk process.
        Corresponds to Section IV-B in the paper. 
        """
        R, d_prime = self.num_paths, specific_list_final[0].size(1)

        # Eq (8): Create path summary vectors using MeanPool and MaxPool 
        '''
        path_summaries = torch.stack([
            torch.cat([s.mean(dim=0), s.max(dim=0).values], dim=-1)
            for s in specific_list_final
        ], dim=0)
        '''
        path_summaries = []
        for s in specific_list_final:
            mean_pool = s.mean(dim=0)
            max_pool = s.max(dim=0).values
            # 计算熵：先对每个维度（列）softmax转为概率分布
            prob_dist = F.softmax(s, dim=0)  # (N x D) -> 每列归一化为概率
            entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-6), dim=0)  # Shannon熵向量，大小D
            # 直接拼接融合：捕捉均值、极值和信息不确定性
            entropy_summary = torch.cat([mean_pool, max_pool, entropy], dim=-1)
            path_summaries.append(entropy_summary)
        path_summaries = torch.stack(path_summaries, dim=0)

        # Eq (9): Calculate the path similarity matrix 
        sim_matrix = torch.matmul(path_summaries, path_summaries.T) / (math.sqrt(3 * d_prime) * self.tau)

        # Eq (10): Form the row-stochastic transition matrix T 
        T = F.softmax(sim_matrix, dim=1)

        # Eq (11-12): Perform Random Walk with Restart 
        pi_0 = F.softmax(self.weight_b.squeeze(), dim=0)
        pi = pi_0
        for _ in range(self.walk_steps):
            pi = self.lambda_rw * pi_0 + (1 - self.lambda_rw) * torch.matmul(pi, T)

        # Eq (14): Apply dropout for stochasticity 
        beta = F.dropout(pi.unsqueeze(1), p=self.beta_drop, training=self.training)
        alpha = torch.sigmoid(self.alpha_logit)
        # Eq (15): Blend refined weights with original scores to get W_tilde 
        #W_tilde =  0.001 * self.weight_b + 1-0.001 * beta
        W_tilde = beta
        return W_tilde

    def TeLU(self,x):
        return x * torch.tanh(torch.exp(x))
    

    def forward(self, feature, A):
        """
        Defines the forward pass of the MPDGCN model.
        """
        # --- Input Handling (Preserved from original code) ---
        try:
            feature = torch.from_numpy(feature.astype(np.float32).toarray())
        except AttributeError:
            # Already a tensor or a dense numpy array
            if not torch.is_tensor(feature):
                 feature = torch.from_numpy(feature.astype(np.float32))

        feature = feature.float()

        # Pre-process adjacency matrices to sparse tensors
        adj_tensors = []
        for i in range(self.num_paths):
            A_i = A[0][i]
            if not torch.is_tensor(A_i) or not A_i.is_sparse:
                 A_i_coo = sp.coo_matrix(A_i)
                 indices = torch.from_numpy(np.vstack((A_i_coo.row, A_i_coo.col))).long()
                 values = torch.from_numpy(A_i_coo.data).float()
                 adj_tensors.append(torch.sparse.FloatTensor(indices, values, torch.Size(A_i_coo.shape)))
            else:
                 adj_tensors.append(A_i)
        
        # --- A. Meta-Path Specific and Shared Graph Encoding ---
        # 1) Generate Meta-Path-Specific embeddings (H_sp) 
        specific_list_final = []
        for i in range(self.num_paths):
            #h_sp1 = F.relu(self.spec_gc1s[i](feature, adj_tensors[i]))
            h_sp1 = self.TeLU(self.spec_gc1s[i](feature, adj_tensors[i]))
            h_sp1 = F.dropout(h_sp1, self.dropout, training=self.training)
            h_sp2 = self.spec_gc2s[i](h_sp1, adj_tensors[i])
            specific_list_final.append(h_sp2)

        # 2) Generate and average Meta-Path-Shared embeddings (H_sh) 
        shared_list = []
        for i in range(self.num_paths):
            #h_sh1 = F.relu(self.shared_gc1(feature, adj_tensors[i]))
            h_sh1 = self.TeLU(self.shared_gc1(feature, adj_tensors[i]))
            h_sh1 = F.dropout(h_sh1, self.dropout, training=self.training)
            h_sh2 = self.shared_gc2(h_sh1, adj_tensors[i])
            shared_list.append(h_sh2)
        H_sh = torch.mean(torch.stack(shared_list, dim=0), dim=0)

        # --- B. Get dynamic path weights ---
        W_tilde = self._markov_enhanced_path_attention(specific_list_final)

        # --- C. Fuse features from different channels ---
        # Create the fused adjacency matrix using the dynamic weights
        final_A = adj_matrix_weight_merge(A, W_tilde)

        # Generate Collaborative features (H_col) 
        concat_sp_feats = torch.cat(specific_list_final, dim=1)
        H_col = self.collab_mlp(concat_sp_feats)
        
        # Generate Fused Specific features (H_sp_fused) by soft-fusing with path weights 
        W_tilde_prob = F.softmax(W_tilde, dim=0).view(1, -1, 1)
        stacked_specific = torch.stack(specific_list_final).transpose(0, 1)
        H_sp_fused = torch.sum(stacked_specific * W_tilde_prob, dim=1)

        # Generate Raw features (H_raw), replicating the logic for U_origin 
        U1 = self.raw_gc1(feature, final_A)
        U2 = self.raw_gc2(U1, final_A) # Note: No activation on U1, as per original code
        H_raw = (U1 + U2) / 2

        # --- Final Assembly and Projection ---
        # Eq (21): Concatenate all four feature types 
        all_feat = torch.cat([H_sp_fused, H_sh, H_col, H_raw], dim=1)
        out = self.proj(all_feat)
        
        # Return final output and intermediate features for consistency
        return out, specific_list_final,shared_list, H_col, H_raw