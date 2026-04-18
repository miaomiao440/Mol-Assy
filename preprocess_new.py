import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import DataLoader

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

def one_hot_encoding(x, permitted_list):

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = []
    for s in permitted_list:
        boolean_value = (x == s) 
        integer_value = int(boolean_value)  
        binary_encoding.append(integer_value)
    return binary_encoding


def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):

    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms) 
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])  
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"]) 
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]) 
    is_in_a_ring_enc = [int(atom.IsInRing())]  
    is_aromatic_enc = [int(atom.GetIsAromatic())]  
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)] 
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)] 
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)] 
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled                                
    if use_chirality == True: 
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    if hydrogens_implicit == True: 
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)

def get_bond_features(bond, use_stereochemistry = True):

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types) 
    bond_is_conj_enc = [int(bond.GetIsConjugated())] 
    bond_is_in_ring_enc = [int(bond.IsInRing())] 
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]) 
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)
    
    
def create_pytorch_geometric_graph_data_list_from_smiles_and_labels2(x_smiles, x_sizes=None):

    node_ls = []
    edge_index_ls = []
    edge_attr_ls = []
    edge_adj_ls = []
    for i, smiles in enumerate(x_smiles):
        
        mol = Chem.MolFromSmiles(smiles) 
        #tiqutezheng
        n_nodes = mol.GetNumAtoms() 
        n_edges = 2*mol.GetNumBonds() 
        unrelated_smiles = "O=O" 
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
        X = np.zeros((n_nodes, n_node_features))
        
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom) 
        if x_sizes is not None:
            x_size = x_sizes[i]
            n_node_features += 1
            tmp_X = np.zeros((n_nodes, n_node_features))
            tmp_X[:,:-1] = X
            tmp_X[:,-1] = x_size 
            X = tmp_X
        X = torch.tensor(X, dtype = torch.float)
        
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol)) 
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0) 
        
        EF = np.zeros((n_edges, n_edge_features)) 
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        EF = torch.tensor(EF, dtype = torch.float) 
        node_ls.append(X)
        edge_index_ls.append(E)
        edge_attr_ls.append(EF)
        edge_adj_ls.append(edge_index_to_adjacency(E, X.shape[0]))
    return  node_ls, edge_index_ls, edge_attr_ls, edge_adj_ls

