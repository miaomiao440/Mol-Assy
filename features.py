import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdchem import HybridizationType
import math
import pandas as pd
from rdkit.Chem import Draw
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from rdkit.Chem import Crippen  
import argparse
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
import os
from rdkit import RDLogger


RDLogger.DisableLog('rdApp.*')

def edge_index_to_adjacency(edge_index, num_nodes=None):

    edge_index = np.array(edge_index)
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    adjacency[edge_index[0, :], edge_index[1, :]] = 1
    # adjacency += adjacency.T  
    return adjacency
 
def adjacency_to_edge_index(adjacency):

    adjacency = np.array(adjacency)
    edge_index = np.array(np.nonzero(adjacency))
    return edge_index

def one_hot_encoding(value, choices):
    """Basic one-hot encoding with fallback for unknowns."""
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    elif "MoreThanFour" in choices and isinstance(value, int) and value > 4:
        encoding[choices.index("MoreThanFour")] = 1
    elif "Extreme" in choices:
        encoding[choices.index("Extreme")] = 1
    return encoding

def get_atom_features(atom,
                      mol=None,
                      atom_index=None,
                      use_chirality=True,
                      hydrogens_implicit=True,
                      conf=None):

    permitted_list_of_atoms = ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl','Yb',
                               'Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt',
                               'Hg','Pb','Unknown']
    if not hydrogens_implicit:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    radical_electrons_enc = [atom.GetNumRadicalElectrons()]
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "OTHER", "SP3D2"])

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, "MoreThanFour"])
    else:
        n_hydrogens_enc = []
    val = atom.GetExplicitValence()
    explicit_valence_enc = one_hot_encoding(val if val <= 8 else "MoreThanFour",
                                            [0, 1, 2, 3, 4, 5, 6, 7, 8, "MoreThanFour"])
    degree = atom.GetDegree()
    n_heavy_neighbors_enc = one_hot_encoding(degree if degree <= 4 else "MoreThanFour", [0, 1, 2, 3, 4, "MoreThanFour"])
    is_in_a_ring_enc = [int(atom.IsInRing())]

    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass_enc = [atom.GetMass()]
    vdw_radius_enc = [Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())]
    covalent_radius_enc = [Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())]
    chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                          ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]) \
                          if use_chirality else []
    logP_contribution = [rdMolDescriptors._CalcCrippenContribs(mol)[atom_index][0]]
    if conf and atom_index is not None:
        pos_i = np.array(conf.GetAtomPosition(atom_index))
        x_y_z = list(pos_i)
        neighbor_positions = []  
        direction_vectors = []   
        distances = []
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            pos_j = np.array(conf.GetAtomPosition(j))
            vec_ij = pos_j - pos_i
            dist_ij = np.linalg.norm(vec_ij)
            if dist_ij > 0:
                direction_vectors.append((vec_ij / dist_ij).tolist())
                distances.append(dist_ij)
                neighbor_positions.append(pos_j)
        avg_distance_to_neighbors = [sum(distances) / len(distances)] if distances else [0.0]
        topological_distance = [0]
        if direction_vectors:
            direction_sum = np.sum(np.array(direction_vectors), axis=0)
            norm = np.linalg.norm(direction_sum)
            relative_direction = (direction_sum / norm).tolist() if norm > 0 else [0.0, 0.0, 0.0]
        else:
            relative_direction = [0.0, 0.0, 0.0]
        bond_angles = []
        torsion_angles = []
        if len(neighbor_positions) >= 2:
            for i in range(len(neighbor_positions)):
                for j in range(i + 1, len(neighbor_positions)):
                    angle = Chem.rdMolTransforms.GetAngleDeg(conf, atom.GetNeighbors()[i].GetIdx(), atom_index,
                                                             atom.GetNeighbors()[j].GetIdx())
                    bond_angles.append(angle)
        if len(atom.GetNeighbors()) >= 3:
            for i in range(len(neighbor_positions)):
                for j in range(i + 1, len(neighbor_positions)):
                    torsion = Chem.rdMolTransforms.GetDihedralDeg(conf,
                                                                  atom.GetNeighbors()[i].GetIdx(),
                                                                  atom_index,
                                                                  atom.GetNeighbors()[j].GetIdx(),
                                                                  atom.GetNeighbors()[(j + 1) % len(neighbor_positions)].GetIdx())
                    torsion_angles.append(torsion)
        bond_angle_feat = [sum(bond_angles) / len(bond_angles)] if bond_angles else [0.0]
        torsion_angle_feat = [sum(torsion_angles) / len(torsion_angles)] if torsion_angles else [0.0]
    else:
        x_y_z = [0.0, 0.0, 0.0]
        avg_distance_to_neighbors = [0.0]
        topological_distance = [0]
        relative_direction = [0.0, 0.0, 0.0]
        bond_angle_feat = [0.0]
        torsion_angle_feat = [0.0]
    atom_feature_vector = (
    list(atom_type_enc) +
    list(formal_charge_enc) +
    list(radical_electrons_enc) +
    list(hybridisation_type_enc) +
    list(n_hydrogens_enc) +
    list(explicit_valence_enc) +
    list(n_heavy_neighbors_enc) +
    list(is_in_a_ring_enc) +
    list(is_aromatic_enc) +
    list(atomic_mass_enc) +
    list(vdw_radius_enc) +
    list(covalent_radius_enc) +
    list(chirality_type_enc) +
    logP_contribution +
    list(x_y_z) +
    list(avg_distance_to_neighbors) +
    list(topological_distance) +
    list(relative_direction) +
    list(bond_angle_feat) +
    list(torsion_angle_feat))

    return np.array(atom_feature_vector)

def get_bond_features(bond, use_stereochemistry=True):

    permitted_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_bond_types)
    is_aromatic_enc = [int(bond.GetIsAromatic())]
    is_conj_enc = [int(bond.GetIsConjugated())]
    is_in_ring_enc = [int(bond.IsInRing())]
    stereo_enc = one_hot_encoding(str(bond.GetStereo()),
                                  ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]) \
                 if use_stereochemistry else []
    bond_dir_enc = one_hot_encoding(str(bond.GetBondDir()),
                                    ["NONE", "BEGINWEDGE", "BEGINDASH", "EITHERDOUBLE"])
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    begin_atomic_num = [begin_atom.GetAtomicNum()]
    end_atomic_num = [end_atom.GetAtomicNum()]
    bond_feature_vector = (
        bond_type_enc + is_aromatic_enc + is_conj_enc + is_in_ring_enc +
        stereo_enc + bond_dir_enc + begin_atomic_num + end_atomic_num
    )
    return np.array(bond_feature_vector)


def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles):

    node_ls = []
    edge_index_ls = []
    edge_attr_ls = []
    edge_adj_ls = []

    for smiles in x_smiles:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res != 0:
            print(f"[Warning] Failed to embed molecule: {smiles}")
            continue  
        conf = mol.GetConformer()
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        ref_mol = Chem.MolFromSmiles("O=O")
        ref_mol = Chem.AddHs(ref_mol)
        AllChem.EmbedMolecule(ref_mol)
        ref_conf = ref_mol.GetConformer()
        n_node_features = len(get_atom_features(ref_mol.GetAtomWithIdx(0), ref_mol, 0, conf=ref_conf))
        print('n_node_features',n_node_features)
        n_edge_features = len(get_bond_features(ref_mol.GetBondBetweenAtoms(0, 1)))
        print('n_edge_features',n_edge_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            X[atom_idx] = get_atom_features(atom, mol=mol, atom_index=atom_idx, conf=conf)

        X = torch.tensor(X, dtype=torch.float)
        rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)
        EF = np.zeros((len(rows), n_edge_features))
        for k, (i, j) in enumerate(zip(rows, cols)):
            bond = mol.GetBondBetweenAtoms(int(i), int(j))
            if bond is not None:
                EF[k] = get_bond_features(bond)
            else:
                EF[k] = np.zeros(n_edge_features) 

        EF = torch.tensor(EF, dtype=torch.float)
        edge_adj = edge_index_to_adjacency(E, X.shape[0])

        node_ls.append(X)
        edge_index_ls.append(E)
        edge_attr_ls.append(EF)
        edge_adj_ls.append(edge_adj)

    data = {
        'node_features': node_ls,
        'edge_index_list': edge_index_ls,
        'edge_attr_list': edge_attr_ls,
        'adj_list': edge_adj_ls,
        'smiles_list': x_smiles
    }

    torch.save(data, "./data/weight/preprocessed_graph_data.pt")

    return node_ls, edge_index_ls, edge_attr_ls, edge_adj_ls


def load_feature_smiles(dataset_path, label_flag='classification', vis=False, out_weight=False):

    if dataset_path.endswith('.tsv'):
        data_file = pd.read_csv(dataset_path, sep='\t')
    elif dataset_path.endswith('.csv'):
        data_file = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.xlsx'):
        data_file = pd.read_excel(dataset_path)
    else:
        raise ValueError(f'{dataset_path}: unhave')
    dataset_path1="./data/weight/train_val_weight.xlsx"
    weight=pd.read_excel(dataset_path1)

    mol_df = pd.read_excel("./data/all_smiles.xlsx")
    mol_name_list = mol_df['NAME'].tolist()
    mol_smiles_list = mol_df['SMILES'].tolist()

    cache_path = "./data/weight/preprocessed_graph_data.pt"
    if os.path.exists(cache_path):
        print("waiting...")
        data = torch.load(cache_path, weights_only=False)
        node_features = data['node_features']
        edge_index_list = data['edge_index_list']
        edge_attr_list = data['edge_attr_list']
        adj_list = data['adj_list']
        mol_smiles_list = data['smiles_list'] 

        sample_node_features = node_features[0]
        sample_edge_features = edge_attr_list[0]
        
        print("node:", sample_node_features.shape)   # (num_atoms, feature_dim)
        print("edge:", sample_edge_features.shape)   # (num_edges, feature_dim)
        node_dims = set([nf.shape[1] for nf in node_features])
        edge_dims = set([ef.shape[1] for ef in edge_attr_list])
        
        print("All types of node feature dimensions:", node_dims)
        print("All types of edge feature dimensions:", edge_dims)
    else:
        print("RDKit....")
        node_features, edge_index_list, edge_attr_list, adj_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(mol_smiles_list)

    mol_dict = dict(zip(mol_name_list, zip(node_features, edge_index_list, adj_list, edge_attr_list, mol_smiles_list)))

    C1_list = data_file['C1'].tolist()
    C2_list = data_file['C2'].tolist()

    if label_flag == 'classification':
        label_list = data_file['classes'].tolist()
    else:
        raise NotImplementedError('Only supports classification')

    S2D_weights = data_file['C1weight'].tolist()
    D2S_weights = data_file['C2weight'].tolist()

    C1_features_ls, C1_edge_index_ls, C1_adj_ls, C1_edge_attr_ls = [], [], [], []
    C2_features_ls, C2_edge_index_ls, C2_adj_ls, C2_edge_attr_ls = [], [], [], []
    C1_info, C2_info = [], []

    for c1, c2 in zip(C1_list, C2_list):
        C1_features_ls.append(mol_dict[c1][0])
        C1_edge_index_ls.append(mol_dict[c1][1])
        C1_adj_ls.append(mol_dict[c1][2])
        C1_edge_attr_ls.append(mol_dict[c1][3])

        C2_features_ls.append(mol_dict[c2][0])
        C2_edge_index_ls.append(mol_dict[c2][1])
        C2_adj_ls.append(mol_dict[c2][2])
        C2_edge_attr_ls.append(mol_dict[c2][3])

        C1_info.append({'C1-name': c1, 'C1-smile': mol_dict[c1][4]})
        C2_info.append({'C2-name': c2, 'C2-smile': mol_dict[c2][4]})

    label_onehot = np.zeros((len(label_list), 2))
    for i, val in enumerate(label_list):
        label_onehot[i, val] = 1


    if not vis and not out_weight:
        return C1_features_ls, C1_edge_index_ls, C1_adj_ls, C1_edge_attr_ls, \
               C2_features_ls, C2_edge_index_ls, C2_adj_ls, C2_edge_attr_ls, label_onehot
    elif vis and not out_weight:
        return C1_features_ls, C1_edge_index_ls, C1_adj_ls, C1_edge_attr_ls, \
               C2_features_ls, C2_edge_index_ls, C2_adj_ls, C2_edge_attr_ls, label_onehot, C1_info, C2_info
    elif not vis and out_weight:
        return C1_features_ls, C1_edge_index_ls, C1_adj_ls, C1_edge_attr_ls, \
               C2_features_ls, C2_edge_index_ls, C2_adj_ls, C2_edge_attr_ls, label_onehot, D2S_weights, S2D_weights
    else:
        return C1_features_ls, C1_edge_index_ls, C1_adj_ls, C1_edge_attr_ls, \
               C2_features_ls, C2_edge_index_ls, C2_adj_ls, C2_edge_attr_ls, label_onehot, D2S_weights, S2D_weights, C1_info, C2_info

def build_edge_to_edge_index(edge_index):
    src, dst = edge_index  # [E]
    same_dst = dst.view(-1, 1) == src.view(1, -1)
    not_back = src.view(-1, 1) != dst.view(1, -1)
    valid = same_dst & not_back
    senders, receivers = valid.nonzero(as_tuple=True)
    return torch.stack([senders, receivers], dim=0)


class GraphDataset(Dataset):
    def __init__(self, dataset_path, label_flag='classification', adj=False, bidrection=False):
        super(GraphDataset, self).__init__()

        node_attr1, edge_index1, edge_adj1, edge_attr1, \
        node_attr2, edge_index2, edge_adj2, edge_attr2, \
        label, weights_D2S, weights_S2D, C1_info, C2_info = load_feature_smiles(
            dataset_path, label_flag='classification', out_weight=True, vis=True)

        self.data_list1 = []     # C1 
        self.data_list2 = []     # C2 
        self.weights = []        
        self.mol_info = []       

        for i in range(len(node_attr1)):
            if adj:
                tmp_data1 = Data(
                    x=torch.tensor(node_attr1[i], dtype=torch.float32),
                    edge_index=torch.tensor(edge_adj1[i], dtype=torch.int64),
                    y=torch.tensor(label[i, :])
                )
                tmp_data2 = Data(
                    x=torch.tensor(node_attr2[i], dtype=torch.float32),
                    edge_index=torch.tensor(edge_adj2[i], dtype=torch.int64),
                    y=torch.tensor(label[i, :])
                )
            else:
                tmp_data1 = Data(
                    x=node_attr1[i].float(),
                    edge_index=edge_index1[i].long(),
                    y=torch.tensor(label[i, :])
                )
                tmp_data2 = Data(
                    x=node_attr2[i].float(),
                    edge_index=edge_index2[i].long(),
                    y=torch.tensor(label[i, :])
                )

            if edge_attr1 is not None and edge_attr2 is not None:
                tmp_data1.edge_attr = edge_attr1[i]
                tmp_data2.edge_attr = edge_attr2[i]

            self.data_list1.append(tmp_data1)
            self.data_list2.append(tmp_data2)
            self.weights.append(weights_S2D[i])
            self.mol_info.append({
                'C1-name': C1_info[i]['C1-name'], 'C1-smile': C1_info[i]['C1-smile'],
                'C2-name': C2_info[i]['C2-name'], 'C2-smile': C2_info[i]['C2-smile'],
                'label': int(label[i, 1])  
            })

            if bidrection:
                self.data_list1.append(tmp_data2)
                self.data_list2.append(tmp_data1)
                self.weights.append(weights_D2S[i])
                self.mol_info.append({
                    'C1-name': C2_info[i]['C2-name'], 'C1-smile': C2_info[i]['C2-smile'],
                    'C2-name': C1_info[i]['C1-name'], 'C2-smile': C1_info[i]['C1-smile'],
                    'label': int(label[i, 1])  
                })

        print('Data loading finished. Total samples:', len(self.data_list1))

    def __len__(self):
        return len(self.data_list1)

    def __getitem__(self, idx):
        data1 = self.data_list1[idx]
        data2 = self.data_list2[idx]
        weight = self.weights[idx]
        mol_info = self.mol_info[idx]
        edge_index_edge1 = build_edge_to_edge_index(data1.edge_index.to(data1.x.device))
        edge_index_edge2 = build_edge_to_edge_index(data2.edge_index.to(data2.x.device))
    
        edge_node1 = Data(
            x=data1.edge_attr,
            edge_index=edge_index_edge1
        )
        edge_node2 = Data(
            x=data2.edge_attr,
            edge_index=edge_index_edge2
        )



        return data1, data2, edge_node1, edge_node2, weight, mol_info

