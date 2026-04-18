import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, Linear
import torch
from torch_geometric.data import Data
import warnings
from torch_geometric.nn import MessagePassing, LayerNorm, Linear, global_add_pool, SAGPooling, global_mean_pool
from torch_scatter import scatter_add
from einops import rearrange
import copy


warnings.filterwarnings("ignore")



class DMPNNLayer(MessagePassing):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, T, dropedge_rate):
        super().__init__(aggr='add')
        self.T = T
        self.hidden_dim = hidden_dim
        self.dropedge_rate = dropedge_rate
        self.edge_init = nn.Linear(node_feat_dim + edge_feat_dim, hidden_dim)
        self.msg_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.node_update = nn.Linear(node_feat_dim + hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        num_edges = edge_index.size(1)
        num_nodes = x.size(0)
        if self.training and self.dropedge_rate > 0:
            mask = torch.rand(num_edges, device=edge_index.device) >= self.dropedge_rate
            src = src[mask]
            dst = dst[mask]
            edge_attr = edge_attr[mask]
            edge_index = torch.stack([src, dst], dim=0)
        edge_input = torch.cat([x[src], edge_attr], dim=1)
        edge_hidden = F.relu(self.edge_init(edge_input))
        edge_hidden_init = edge_hidden.clone()
        
        
        same_dst = dst.view(-1, 1) == src.view(1, -1)
        not_back = src.view(-1, 1) != dst.view(1, -1)
        valid = same_dst & not_back
        edge_sources, edge_targets = valid.nonzero(as_tuple=True)
        edge_index_edge = torch.stack([edge_sources, edge_targets], dim=0)
        
        for _ in range(self.T):
            m = scatter_add(edge_hidden[edge_index_edge[0]], edge_index_edge[1], dim=0, dim_size=src.size(0))
            edge_hidden = F.relu(edge_hidden_init + self.msg_mlp(m))

        node_msg = scatter_add(edge_hidden, dst, dim=0, dim_size=num_nodes)
        node_input = torch.cat([x, node_msg], dim=1)
        node_hidden = F.relu(self.node_update(node_input))
        return node_hidden


class Encode_Block(nn.Module):
    def __init__(self, in_features, n_heads, head_out_feats, edge_feature, dp, T, dropedge_rate):
        super().__init__()
        self.feature_conv = DMPNNLayer(in_features, edge_feature, in_features, T, dropedge_rate)
        self.feature_conv2 = DMPNNLayer(in_features, edge_feature, head_out_feats, T, dropedge_rate)
        self.norm = LayerNorm(in_features)
        self.norm2 = LayerNorm(head_out_feats)
        self.lin_up = Linear(edge_feature, edge_feature, bias=True, weight_initializer='glorot')
        self.lin_up2 = Linear(edge_feature, edge_feature, bias=True, weight_initializer='glorot')
        self.dropout_e = nn.Dropout(dp)
        self.dropout_node = nn.Dropout(dp)
        self.dropout_e2 = nn.Dropout(dp)
        self.dropout_node2 = nn.Dropout(dp)

    def forward(self, drug_data):
        drug_data.edge_attr = F.elu(self.dropout_e(self.lin_up(drug_data.edge_attr)))
        drug_data.x = F.elu(self.norm(self.feature_conv(drug_data.x, drug_data.edge_index, drug_data.edge_attr), drug_data.batch))
        drug_data.x = self.dropout_node(drug_data.x)

        drug_data.edge_attr = F.elu(self.dropout_e2(self.lin_up2(drug_data.edge_attr)))
        drug_data.x = F.elu(self.norm2(self.feature_conv2(drug_data.x, drug_data.edge_index, drug_data.edge_attr), drug_data.batch))
        drug_data.x = self.dropout_node2(drug_data.x)

        return drug_data

class CoAttentionLayer2(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
    def forward(self, query, keyvalue):
        # LayerNorm
        query = self.norm(query)
        keyvalue = self.norm(keyvalue)
        q = self.to_q(query)  # [B, N, heads * dim_head]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)  # [B, H, N, D]
        # Linear projection to K, V
        kv = self.to_kv(keyvalue).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)  # [B, H, N, D]
        # Attention score: [B, H, N, N]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, N, heads * dim_head]
        out = self.to_out(out)  # [B, N, dim]
        return out     
class SD(nn.Module):
    def __init__(self, nclass, in_node_features, in_edge_features, hidd_dim, n_out_feats, n_heads, edge_feature, dp,T,dropedge_rate):
        super(SD, self).__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.n_out_feats = n_out_feats
        self.hidd_dim = hidd_dim
        #self.dropout_readout1 = nn.Dropout(dropedge_rate)
        self.edge_feature = edge_feature
        self.n_blocks = len(n_heads)
        assert len(self.n_out_feats) == len(n_heads), 'head_out_feats长度与blocks_params必须相等'
        self.initial_node_feature = Linear(self.in_node_features, self.in_node_features ,bias=True, weight_initializer='glorot')
        self.initial_edge_feature = Linear(self.in_edge_features, edge_feature ,bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(self.in_node_features)
        self.n_out_feats = copy.deepcopy(n_out_feats)
        self.n_out_feats.insert(0, self.in_node_features)
        self.encode_blocks = []
        self.readouts = []
        self.to_trans = []
        for i in range(self.n_blocks):
            self.encode_blocks.append(Encode_Block(self.n_out_feats[i], n_heads[i], self.n_out_feats[i+1], edge_feature, dp,T,dropedge_rate=dropedge_rate ))
            self.readouts.append(SAGPooling(self.n_out_feats[i+1], min_score=-1)) 
            self.to_trans.append(Linear(self.n_out_feats[i+1], hidd_dim, bias=False))
        self.n_out_feats = self.n_out_feats [1:]
        self.encode_blocks = nn.ModuleList(self.encode_blocks)
        self.readouts = nn.ModuleList(self.readouts)
        self.to_trans = nn.ModuleList(self.to_trans)
        self.inter_mol_attention = CoAttentionLayer2(hidd_dim, self.n_blocks, hidd_dim) 
        dim_in = len(self.n_out_feats) ** 2
        self.dropout_readout = nn.Dropout(dp)
        self.merge_all = nn.Sequential(
            nn.Dropout(dp),
            nn.Linear(dim_in, dim_in // 2),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(dim_in // 2, 1)
        )

        self.re_shape_e = Linear(edge_feature, hidd_dim, bias=True, weight_initializer='glorot')
    def forward(self, data1, data2, edge_node1=None, edge_node2=None):
        data1.x = F.elu(self.initial_node_norm(self.initial_node_feature(data1.x), data1.batch))
        data2.x = F.elu(self.initial_node_norm(self.initial_node_feature(data2.x), data2.batch))
        
        if edge_node1 is not None:
            data1.edge_attr = F.elu(self.initial_edge_feature(edge_node1.x))
            data2.edge_attr = F.elu(self.initial_edge_feature(edge_node2.x))
        else:
            data1.edge_attr = F.elu(self.initial_edge_feature(data1.edge_attr))
            data2.edge_attr = F.elu(self.initial_edge_feature(data2.edge_attr))
    
        repr_h = []
        repr_t = []
        for i in range(self.n_blocks):
            data1 = self.encode_blocks[i](data1)
            data2 = self.encode_blocks[i](data2)
            h_global_graph_emb, t_global_graph_emb = self.GlosbalPool(data1, data2, i)
            repr_h.append(h_global_graph_emb)
            repr_t.append(t_global_graph_emb)
    
        repr_h = torch.stack(repr_h, dim=1)
        repr_t = torch.stack(repr_t, dim=1)
    
        head_attentions = self.inter_mol_attention(repr_h, repr_t)
        tail_attentions = self.inter_mol_attention(repr_t, repr_h)
        similarity_matrix = torch.matmul(head_attentions, tail_attentions.transpose(-2, -1))  # [B, n_blocks, n_blocks]
    
        out = self.merge_all(similarity_matrix.flatten(1))  # [B, n_blocks^2] → [B, 1]
        final_output=out.squeeze(-1)
        return out.squeeze(-1)

    def GlosbalPool(self, h_data, t_data, i):
        h_att_x, _, _, h_att_batch, _, _ = self.readouts[i](h_data.x, h_data.edge_index, edge_attr=h_data.edge_attr, batch=h_data.batch)
        t_att_x, _, _, t_att_batch, _, _ = self.readouts[i](t_data.x, t_data.edge_index, edge_attr=t_data.edge_attr, batch=t_data.batch)
        
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)
        
        h_edge_batch = h_data.batch[h_data.edge_index[0]]
        t_edge_batch = t_data.batch[t_data.edge_index[0]]
        h_global_graph_emb_edge = global_add_pool(h_data.edge_attr, h_edge_batch)
        t_global_graph_emb_edge = global_add_pool(t_data.edge_attr, t_edge_batch)
        
        h_global_graph_emb_edge = F.elu(self.re_shape_e(h_global_graph_emb_edge))
        t_global_graph_emb_edge = F.elu(self.re_shape_e(t_global_graph_emb_edge))
        
        h_global_graph_emb = F.normalize(self.to_trans[i](h_global_graph_emb))
        t_global_graph_emb = F.normalize(self.to_trans[i](t_global_graph_emb))
        
        h_global_graph_emb = self.dropout_readout(h_global_graph_emb)
        t_global_graph_emb = self.dropout_readout(t_global_graph_emb)
        
        return h_global_graph_emb, t_global_graph_emb

