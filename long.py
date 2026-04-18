import pandas as pd
from rdkit import Chem
from collections import defaultdict

def build_graph(mol):
    graph = defaultdict(list)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        graph[a1].append(a2)
        graph[a2].append(a1)
    return graph

def dfs(graph, node, visited):
    visited.add(node)
    max_len = 0
    for neighbor in graph[node]:
        if neighbor not in visited:
            max_len = max(max_len, dfs(graph, neighbor, visited.copy()) + 1)
    return max_len

def get_longest_chain_length(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        graph = build_graph(mol)
        return max(dfs(graph, node, set()) for node in graph)
    except Exception as e:
        return None

file_path = "./data/right.xlsx"
df = pd.read_excel(file_path)

df["LongestChainLength"] = df["SMILES"].apply(get_longest_chain_length)

output_path = file_path.replace(".xlsx", "_with_chain_length.xlsx")
df.to_excel(output_path, index=False)

print(f"ok：{output_path}")
