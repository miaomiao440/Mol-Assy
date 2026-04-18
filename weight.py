import numpy as np
import pandas as pd



def calculate_complete_weight(df, val_c, y, feature_name, K=2, alpha=1):

    tmp_lab = df[df[feature_name] == val_c]['classes']
    if tmp_lab.empty:
        return 1.0

    Nd = len(tmp_lab)
    Ndy = (tmp_lab == y).sum()

    # 1. w_raw
    w_raw = np.log10(1 + np.log(1 + (Nd + K * alpha) / (Ndy + alpha)))

    # 2. 条件内调整 m_cond
    C1 = tmp_lab.eq(1).sum()
    C0 = tmp_lab.eq(0).sum()

    minority_class_in_cond = 1 if C1 < C0 else 0
    is_minority_in_condition = (C1 != C0) and (y == minority_class_in_cond)

    is_extreme_imbalance = (C1 == 1 or C0 == 1)

    if is_minority_in_condition or is_extreme_imbalance:
        m_cond = 1.5
    else:
        m_cond = 1.0

    # 3. 全局调整 m_global
    global_counts = df["classes"].value_counts()
    G1 = global_counts.get(1, 0)
    G0 = global_counts.get(0, 0)
    G_total = G1 + G0

    p_global = G1 / G_total if G_total > 0 else 0.5
    is_balanced_global = (abs(p_global - 0.5) < 0.1)

    global_minority_class = 1 if G1 < G0 else 0
    is_minority_globally = (y == global_minority_class)

    if min(G1, G0) > 0:
        r_global = max(G1, G0) / min(G1, G0)
    else:
        r_global = 1  # 避免除0

    if (not is_balanced_global) and is_minority_globally:
        m_global = np.log(1 + r_global)
    else:
        m_global = 1.0

    # 4. final weight
    return w_raw * m_cond * m_global


input_path = "./data/train_val4.xlsx"
output_path = "./data/train_val4_with_weights.xlsx"

df = pd.read_excel(input_path)


weights_C1 = []
for idx, row in df.iterrows():
    w = calculate_complete_weight(df, row["C1"], row["classes"], "C1")
    weights_C1.append(w)


weights_C2 = []
for idx, row in df.iterrows():
    w = calculate_complete_weight(df, row["C2"], row["classes"], "C2")
    weights_C2.append(w)


df["weight_C1"] = weights_C1
df["weight_C2"] = weights_C2


df.to_excel(output_path, index=False)

print("计算完成，文件已输出到：", output_path)
