"""
Combining 2D and 3D sampling
"""
import pandas as pd
import numpy as np
import os


output_dir = "./datasets/1/"
step7_dir = os.path.join(output_dir, "step7_pref_c1_neq_c2")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(step7_dir, exist_ok=True)

target_total = 587
path_2d = "./datasets/butina_cluster_results.xlsx"
path_3d = "./datasets/3d_cluster_results.xlsx"
path_all = "./datasets/all_dataset_lower.xlsx"

df_2d = pd.read_excel(path_2d)
df_3d = pd.read_excel(path_3d)
df_all = pd.read_excel(path_all)

#Small cluster filtering
small_2d_ids = df_2d['cluster_id'].value_counts()
small_2d_ids = small_2d_ids[small_2d_ids <= 10].index
df_small_2d = df_2d[df_2d['cluster_id'].isin(small_2d_ids)].sort_values(by='cluster_id')
df_small_2d.to_excel(os.path.join(output_dir, "step1_small_2d_full.xlsx"), index=False)

small_3d_ids = df_3d['cluster_id'].value_counts()
small_3d_ids = small_3d_ids[small_3d_ids <= 80].index
df_small_3d = df_3d[df_3d['cluster_id'].isin(small_3d_ids)].sort_values(by='cluster_id')
df_small_3d.to_excel(os.path.join(output_dir, "step2_small_3d_full.xlsx"), index=False)

#Merge small clusters to remove duplicates
df_small_combined = pd.concat([df_small_2d, df_small_3d]).drop_duplicates(subset="nub")
df_small_combined.to_excel(os.path.join(output_dir, "step3_small_combined_full.xlsx"), index=False)
nub_small = set(df_small_combined["nub"])
small_total = len(nub_small)

#Calculate the remaining quantity
rest = target_total - small_total

#Large cluster sampling function
def sample_large(df, nub_exclude, cluster_threshold, rest_budget, label):
    large_ids = df['cluster_id'].value_counts()
    large_ids = large_ids[large_ids > cluster_threshold].index
    df_large = df[df['cluster_id'].isin(large_ids) & ~df['nub'].isin(nub_exclude)]
    total_large = len(df_large)

    samples = []
    for clus in large_ids:
        group = df_large[df_large['cluster_id'] == clus]
        prop = len(group) / total_large
        n = int(np.round(prop * rest_budget))
        if n > 0:
            samples.append(group.sample(n=min(n, len(group)), random_state=42))

    if samples:
        result = pd.concat(samples).drop_duplicates(subset="nub")
    else:
        result = pd.DataFrame(columns=df.columns)

    result.to_excel(os.path.join(output_dir, f"step5_large_{label}_sampled_full.xlsx"), index=False)
    return result

df_large_2d = sample_large(df_2d, nub_small, 10, rest, "2d")
df_large_3d = sample_large(df_3d, nub_small, 80, rest, "3d")

#Merge all and deduplicate again
df_large_combined = pd.concat([df_large_2d, df_large_3d]).drop_duplicates(subset="nub")
df_all_combined = pd.concat([df_small_combined, df_large_combined]).drop_duplicates(subset="nub")
df_all_combined.to_excel(os.path.join(output_dir, "step6_combined_total_full.xlsx"), index=False)

#Prioritize choosing combinations with different binary values
df_c1_not_eq_c2 = df_all_combined[df_all_combined["C1"] != df_all_combined["C2"]]
df_c1_eq_c2 = df_all_combined[df_all_combined["C1"] == df_all_combined["C2"]]
df_c1_not_eq_c2.to_excel(os.path.join(step7_dir, "step7a_c1_not_eq_c2.xlsx"), index=False)

#select 295 combinations that meet different binary requirements
needed_large = 295
if len(df_c1_not_eq_c2) >= needed_large:
    df_selected_large = df_c1_not_eq_c2.sample(n=needed_large, random_state=42)
    df_selected_large.to_excel(os.path.join(step7_dir, "step7a_selected_large_295.xlsx"), index=False)
else:
    df_selected_large = df_c1_not_eq_c2.copy()
    df_selected_large.to_excel(os.path.join(step7_dir, "step7a_selected_large_less_than_295.xlsx"), index=False)

#Merge small clusters and optimize results
df_all_step7 = pd.concat([df_small_combined, df_selected_large]).drop_duplicates(subset="nub")
df_all_step7.to_excel(os.path.join(step7_dir, "step7b_combined_small_large.xlsx"), index=False)

#If it is still less than 587, add unused binary combinations from the entire collection
n_selected = len(df_all_step7)
if n_selected < target_total:
    rest_needed = target_total - n_selected
    used_nub = set(df_all_step7["nub"])

    df_all_candidates = df_all[
        (df_all["classes"] == 0) &
        (df_all["C1"] != df_all["C2"]) &
        (~df_all["nub"].isin(used_nub))
    ]

    df_extra = df_all_candidates.sample(n=min(rest_needed, len(df_all_candidates)), random_state=42)
    df_extra["cluster_id"] = -1  #Mark as by election

    df_final = pd.concat([df_all_step7, df_extra], ignore_index=True)
else:
    df_final = df_all_step7.copy()

#Final save result
df_final.to_excel(os.path.join(step7_dir, "final_result_587_pref_c1_neq_c2_from_all.xlsx"), index=False)
print("Completed, total sample size:", len(df_final))



"""
3D-Sampling based on 3D descriptors
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.read_excel("./datasets/all_dataset_lower.xlsx")
df0 = df[df["classes"] == 0].copy()

#Fingerprints
fps_data = np.load("./datasets/Fingerprints.npy", allow_pickle=True)
name_to_smiles = {entry["name"]: entry["smiles"] for entry in fps_data}

#3D
descriptor_functions = {
    "RadiusOfGyration": rdMolDescriptors.CalcRadiusOfGyration,
    "InertialShapeFactor": rdMolDescriptors.CalcInertialShapeFactor,
    "Asphericity": rdMolDescriptors.CalcAsphericity,
    "SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex,
    "Eccentricity": rdMolDescriptors.CalcEccentricity,
    "PMI1": rdMolDescriptors.CalcPMI1,
    "PMI2": rdMolDescriptors.CalcPMI2,
    "PMI3": rdMolDescriptors.CalcPMI3,
    "NPR1": rdMolDescriptors.CalcNPR1,
    "PBF": rdMolDescriptors.CalcPBF,
    "LabuteASA": rdMolDescriptors.CalcLabuteASA,
}


#Generate 3D descriptors
descriptor_cache = {}

def compute_descriptors(smiles):
    if smiles in descriptor_cache:
        return descriptor_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return None
    desc_values = []
    for name, func in descriptor_functions.items():
        try:
            value = func(mol)
        except:
            value = np.nan
        desc_values.append(value)
    descriptor_cache[smiles] = desc_values
    return desc_values

#Build drug pair vectors (use mean)
pair_vectors = []
meta_data = []

for _, row in df0.iterrows():
    c1, c2 = row["C1"], row["C2"]
    smiles1 = name_to_smiles.get(c1)
    smiles2 = name_to_smiles.get(c2)
    if smiles1 is None or smiles2 is None:
        continue
    desc1 = compute_descriptors(smiles1)
    desc2 = compute_descriptors(smiles2)
    if desc1 is None or desc2 is None:
        continue
    vec = list((np.array(desc1) + np.array(desc2)) / 2)
    pair_vectors.append(vec)
    meta_data.append({
        "C1": c1,
        "C2": c2,
        "classes": row["classes"],
        "nub": row.get("nub", None)  # if nub exists
    })

X = np.array(pair_vectors)
meta_df = pd.DataFrame(meta_data)

# cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=20, random_state=42)
labels = kmeans.fit_predict(X_scaled)
meta_df["cluster_id"] = labels

meta_df.to_excel("./datasets/3d_cluster_results.xlsx", index=False)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab20', s=20)
plt.title("3D Descriptor-based Drug Pair Clustering (PCA View)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar(scatter, label="Cluster ID")
plt.tight_layout()
plt.savefig("./datasets/3d_cluster_scatter.png")
plt.close()





"""
2D-Sampling based on 2D descriptors
"""

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


df = pd.read_excel("./datasets/all_dataset_lower.xlsx")  
df0 = df[df["classes"] == 0].copy()


fps_data = np.load("./datasets/Fingerprints.npy", allow_pickle=True)  
fp_dict = {item["name"]: item["fingerprint"] for item in fps_data}


pair_names = []
pair_fps = []

for _, row in df0.iterrows():
    c1, c2 = row["C1"], row["C2"]
    if c1 in fp_dict and c2 in fp_dict:
        fp1 = fp_dict[c1]
        fp2 = fp_dict[c2]
        combined = np.concatenate([fp1, fp2])  
        pair_names.append((c1, c2))
        pair_fps.append(combined)


def array_to_bv(arr):
    bv = DataStructs.ExplicitBitVect(len(arr))
    for i, v in enumerate(arr):
        if v:
            bv.SetBit(i)
    return bv

bv_fps = [array_to_bv(fp) for fp in pair_fps]
n = len(bv_fps)


sim_matrix = np.zeros((n, n))
for i in range(n):
    sims = DataStructs.BulkTanimotoSimilarity(bv_fps[i], bv_fps)
    sim_matrix[i] = sims

plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Tanimoto Similarity Heatmap of Drug Pairs")
plt.tight_layout()
plt.savefig("./datasets/tanimoto_similarity_matrix.png")
plt.close()

pair_labels = [f"{c1}_{c2}" for c1, c2 in pair_names]
sim_df = pd.DataFrame(sim_matrix, index=pair_labels, columns=pair_labels)
sim_df.to_excel("./datasets/tanimoto_similarity_matrix.xlsx")
print("tanimoto_similarity_matrix.xlsx")


dists = []
for i in range(1, n):
    sims = DataStructs.BulkTanimotoSimilarity(bv_fps[i], bv_fps[:i])
    dists.extend([1 - x for x in sims])

cutoff = 0.9  ###################
clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)


X = np.array(pair_fps)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)


labels = np.zeros(n, dtype=int) - 1  
for idx, cluster in enumerate(clusters):
    for i in cluster:
        labels[i] = idx

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab20', s=10)
plt.title(f"Butina Clustering (cutoff={cutoff}) — PCA Projection")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(scatter, label='Cluster ID')
plt.tight_layout()
plt.savefig("./datasets/butina_cluster_scatter.png")
plt.close()

#print(f"总药物对数: {n}")
#print(f"聚类数量: {len(clusters)}")
#print(f"最大簇大小: {max([len(c) for c in clusters]) if clusters else 0}")


pair_to_meta = {}
for _, row in df0.iterrows():
    pair = (row["C1"], row["C2"])
    if pair in pair_names:
        idx = pair_names.index(pair)
        pair_to_meta[pair] = {
            "nub": row["nub"],
            "classes": row["classes"]
        }
cluster_data = []
for cluster_id, cluster in enumerate(clusters):
    for idx in cluster:
        c1, c2 = pair_names[idx]
        meta = pair_to_meta.get((c1, c2), {})
        cluster_data.append({
            "C1": c1,
            "C2": c2,
            "nub": meta.get("nub", None),
            "classes": meta.get("classes", None),
            "cluster_id": cluster_id
        })


cluster_df = pd.DataFrame(cluster_data)
output_path = "./datasets/butina_cluster_results.xlsx"
cluster_df.to_excel(output_path, index=False)

print(f"\nThe clustering results have been saved to：{output_path}")


"""
Generate Morgan fingerprints
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs


input_excel_path = "./datasets/all_smiles.xlsx"
output_npy_path = "./datasets/Fingerprints.npy"
df = pd.read_excel(input_excel_path)

# name-SMILES
names = df.iloc[:, 0].astype(str).tolist()
smiles_list = df.iloc[:, 1].astype(str).tolist()

# fingerprints
generator = GetMorganGenerator(radius=2, fpSize=2048)
records = []
for name, smi in zip(names, smiles_list):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
    else:
        arr = np.zeros((2048,), dtype=np.int8)  
    records.append({
        "name": name,
        "smiles": smi,
        "fingerprint": arr
    })

np.save(output_npy_path, records, allow_pickle=True)



