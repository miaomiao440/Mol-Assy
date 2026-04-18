import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from preprocess_new import create_pytorch_geometric_graph_data_list_from_smiles_and_labels2
def calculate_complete_weight(df, val_c, y, feature_name, K=2, alpha=1):
    tmp_lab = df[df[feature_name] == val_c]['classes']
    if tmp_lab.empty:
        return 1.0, {"reason": "no data"}
    Nd = len(tmp_lab)               
    Ndy = (tmp_lab == y).sum()      
    w_raw = np.log10(1 + np.log(1 + (Nd + K * alpha) / (Ndy + alpha)))
    C1 = tmp_lab.eq(1).sum()
    C0 = tmp_lab.eq(0).sum()
    minority_class_in_cond = 1 if C1 < C0 else 0
    is_minority_in_condition = (C1 != C0) and (y == minority_class_in_cond)
    is_extreme_imbalance = (C1 == 1 or C0 == 1)

    m_cond = 1.5 if (is_minority_in_condition or is_extreme_imbalance) else 1.0
    global_counts = df['classes'].value_counts()
    G1 = global_counts.get(1, 0)
    G0 = global_counts.get(0, 0)
    G_total = G1 + G0

    p_global = G1 / G_total if G_total > 0 else 0.5

    is_balanced_global = (abs(p_global - 0.5) < 0.1)

    global_minority_class = 1 if G1 < G0 else 0
    is_minority_globally = (y == global_minority_class)

    if is_balanced_global or G_total == 0:
        m_global = 1.0
    else:
        s = abs(2 * p_global - 1)

        if is_minority_globally:
            m_global = 0.7 + 0.3 * (1 - s)
        else:
            m_global = 0.9 + 0.1 * (1 - s)
        m_global = min(1.0, max(0.0, m_global))

    m_global = 1.0
    w_final = w_raw * m_cond * m_global


    reason = {
        "w_raw": w_raw,
        "m_cond": m_cond,
        "m_global": m_global,
        "conditional_reason": (
            "Minority categories within the conditions" if is_minority_in_condition else
            ("Extreme imbalance within the conditions" if is_extreme_imbalance else "Within the conditions, it is flat衡")
        ),
        "global_reason": (
            "Global balance" if is_balanced_global else
            ("Global minority class" if is_minority_globally else "Global majority class")
        )
    }

    return w_final, reason

def load_feature_smiles(dataset_path, label_flag='classification', vis=False, out_weight=False):
    dataset_path = str(dataset_path)

    if dataset_path[-3:] == 'tsv':
        data_file = pd.read_csv(dataset_path, sep='\t')
    elif dataset_path[-3:] == 'csv':
        data_file = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.xlsx'):
        data_file = pd.read_excel(dataset_path)
        df = pd.read_excel(dataset_path)
    else:
        assert False, f'{dataset_path}:未知的后缀'
    ##############################################################################
    drug_ls = pd.read_excel("./data/all_smiles.xlsx")#######################################
    son_ls = pd.read_excel("./data/all_smiles.xlsx")#######################################
    '''
    for i in range(len(drug_ls)):
        name = drug_ls.loc[i, 'NAME']
        smiles = drug_ls.loc[i, 'SMILES']
    '''
    row, _ = data_file.shape
    A_ls = []
    B_ls = []
    label_ls = []
    S2D_weights = []
    D2S_weights = []
    weight_details = []
    
    for i in range(row):
        A_ls.append(data_file.iloc[i,0])
        B_ls.append(data_file.iloc[i,1])
        if label_flag == 'regression':
            label_ls.append(data_file.iloc[i,0])
        elif label_flag == 'classification':
            val_c1 = data_file.iloc[i]['C1']
            val_c2 = data_file.iloc[i]['C2']
            y = data_file.iloc[i]['classes'] 
            
            w1, reason1 = calculate_complete_weight(df, val_c1, y, 'C1')
            w2, reason2 = calculate_complete_weight(df, val_c2, y, 'C2')
            
            weight_details.append({
                'A': data_file.iloc[i,0],
                'B': data_file.iloc[i,1],
                'C1': val_c1,
                'C2': val_c2,
                'label': y,
                'D2S_weight': w1,
                'S2D_weight': w2,
                'D2S_reason': reason1,
                'S2D_reason': reason2
            })
            
            D2S_weights.append(w1)
            S2D_weights.append(w2)
            label_ls.append(y)
        else:
            assert False, '未实现'

    drug_name_ls = list(drug_ls.iloc[:,0])
    drug_smiles_ls = list(drug_ls.iloc[:,1])
    if 'size' in data_file.columns:
        x_sizes = data_file['size'].tolist()
    else:
        x_sizes = None

    drug_node_features, drug_edge_index, drug_edge_feature, drug_adj_mat = create_pytorch_geometric_graph_data_list_from_smiles_and_labels2(drug_smiles_ls,x_sizes = None)
    drug_dict = dict(zip(drug_name_ls,list(zip(drug_node_features, drug_edge_index, drug_adj_mat, drug_edge_feature, drug_smiles_ls))))

    son_name_ls = list(son_ls.iloc[:,0])
    son_smiles_ls = list(son_ls.iloc[:,1])
    son_node_features, son_edge_index, son_edge_feature, son_adj_mat = create_pytorch_geometric_graph_data_list_from_smiles_and_labels2(son_smiles_ls,x_sizes = None)
    son_dict = dict(zip(son_name_ls,list(zip(son_node_features, son_edge_index, son_adj_mat, son_edge_feature, son_smiles_ls))))
    
    drug_features_ls = []
    drug_edge_index_ls = []
    drug_adj_ls = []
    drug_edge_features_ls = []

    son_features_ls = []
    son_edge_index_ls = []
    son_adj_ls = []
    son_edge_features_ls = []

    drug_info = []
    son_info = []

    for son_name, drug_name in zip(A_ls, B_ls):
        drug_features_ls.append(drug_dict[drug_name][0])
        drug_edge_index_ls.append(drug_dict[drug_name][1])
        drug_adj_ls.append(drug_dict[drug_name][2])
        drug_edge_features_ls.append(drug_dict[drug_name][3])

        son_features_ls.append(son_dict[son_name][0])
        son_edge_index_ls.append(son_dict[son_name][1])
        son_adj_ls.append(son_dict[son_name][2])
        son_edge_features_ls.append(son_dict[son_name][3])

        drug_info.append({'drug-name':drug_name,'drug-smile':drug_dict[drug_name][4]})
        son_info.append({'son-name':son_name, 'son-smile':son_dict[son_name][4]})

    label_onehot = np.zeros((len(label_ls), 2))
    
    for i, val in enumerate(label_ls):
        label_onehot[i, val] = 1


    if not vis and not out_weight:
        return drug_features_ls, drug_edge_index_ls, drug_adj_ls,  drug_edge_features_ls, son_features_ls, son_edge_index_ls, son_adj_ls, son_edge_features_ls, label_onehot
    elif vis and not out_weight:
        return drug_features_ls, drug_edge_index_ls, drug_adj_ls,  drug_edge_features_ls, son_features_ls, son_edge_index_ls, son_adj_ls, son_edge_features_ls, label_onehot, drug_info, son_info
    elif not vis and out_weight:
        return drug_features_ls, drug_edge_index_ls, drug_adj_ls,  drug_edge_features_ls, son_features_ls, son_edge_index_ls, son_adj_ls, son_edge_features_ls, label_onehot, D2S_weights, S2D_weights
    else:
        return drug_features_ls, drug_edge_index_ls, drug_adj_ls,  drug_edge_features_ls, son_features_ls, son_edge_index_ls, son_adj_ls, son_edge_features_ls, label_onehot, D2S_weights, S2D_weights,  drug_info, son_info
        







class GraphDataset4(Dataset):
    def __init__(self, dataset_path, label_flag='classification', adj=False, bidrection=False):
        super(GraphDataset4, self).__init__()
        
        node_attr1, edge_index1, edge_adj1, edge_attr1, node_attr2, edge_index2, edge_adj2, edge_attr2, label, weights_D2S, weights_S2D, drug_info, son_info = load_feature_smiles(
            dataset_path, label_flag='classification', out_weight=True, vis=True)
        
        self.data_list1 = []
        self.data_list2 = []
        self.weights_D2S = []  
        self.weights_S2D = []
        self.drug_info = []
        self.sample_ids = []

        reversed_count = 0  
        
        print("...")
        
        for i in range(len(node_attr1)):
            if adj:
                tmp_data1 = Data(x=torch.tensor(node_attr1[i], dtype=torch.float32),
                                edge_index=torch.tensor(edge_adj1[i], dtype=torch.int64), 
                                y=torch.tensor([label[i,1]], dtype=torch.float32))
                tmp_data2 = Data(x=torch.tensor(node_attr2[i], dtype=torch.float32),
                                edge_index=torch.tensor(edge_adj2[i], dtype=torch.int64),
                                y=torch.tensor([label[i,1]], dtype=torch.float32))
            else:
                tmp_data1 = Data(x=node_attr1[i].float(),
                                edge_index=edge_index1[i].long(), 
                                y=torch.tensor([label[i,1]], dtype=torch.float32))
                tmp_data2 = Data(x=node_attr2[i].float(),
                                edge_index=edge_index2[i].long(),
                                y=torch.tensor([label[i,1]], dtype=torch.float32))
            
            if edge_attr1 is not None and edge_attr2 is not None:
                tmp_data1.edge_attr = edge_attr1[i]
                tmp_data2.edge_attr = edge_attr2[i]
            
            current_son_name = son_info[i]['son-name']
            current_drug_name = drug_info[i]['drug-name']
            
            is_same_drug = (current_son_name == current_drug_name)
            
            
            self.data_list1.append(tmp_data1)
            self.data_list2.append(tmp_data2)
            self.weights_D2S.append(weights_D2S[i])
            self.weights_S2D.append(weights_S2D[i])
            self.drug_info.append({
                'drug1-name': current_son_name, 
                'drug1-smile': son_info[i]['son-smile'],
                'drug2-name': current_drug_name, 
                'drug2-smile': drug_info[i]['drug-smile'],
                'is_reversed': False  #
            })
            self.sample_ids.append(i)
            
            if bidrection and not is_same_drug:
                self.data_list1.append(tmp_data2)
                self.data_list2.append(tmp_data1)
                self.weights_D2S.append(weights_S2D[i])  
                self.weights_S2D.append(weights_D2S[i]) 
                self.drug_info.append({
                    'drug1-name': current_drug_name,  
                    'drug1-smile': drug_info[i]['drug-smile'],
                    'drug2-name': current_son_name, 
                    'drug2-smile': son_info[i]['son-smile'],
                    'is_reversed': True  
                })
                self.sample_ids.append(i)
                reversed_count += 1

    def __len__(self):
        return len(self.data_list1)
    
    def __getitem__(self, idx):
        data1 = self.data_list1[idx]
        data2 = self.data_list2[idx]
        edge_node1 = Data(x=data1.edge_attr, edge_index=torch.LongTensor())
        edge_node2 = Data(x=data2.edge_attr, edge_index=torch.LongTensor())
        weight_D2S = torch.tensor(self.weights_D2S[idx], dtype=torch.float32)
        weight_S2D = torch.tensor(self.weights_S2D[idx], dtype=torch.float32)
        
        drug_info = self.drug_info[idx]
        sample_id = self.sample_ids[idx]
        

        return data1, data2, edge_node1, edge_node2, weight_D2S, weight_S2D , sample_id


