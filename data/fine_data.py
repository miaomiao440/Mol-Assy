import pandas as pd

# 输入文件
file_path = "/home/Zmm/ZMM/7.5/mm-data-weight11.18-41/data/train_val4.xlsx"

# 读取文件
df = pd.read_excel(file_path)

# 打乱顺序
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# 输出文件（避免覆盖）
output_path = "/home/Zmm/ZMM/7.5/mm-data-weight11.18-41/data/train_val4_s.xlsx"
df_shuffled.to_excel(output_path, index=False)

print(f"打乱完成，共 {len(df_shuffled)} 行")
print("保存文件:", output_path)

'''
import pandas as pd

file_path = "/home/zmm/7.17/pk/SDI-data-ssh/data_zmm/train_val4.xlsx"
df = pd.read_excel(file_path)

# 检查必要列
required_cols = ["C1", "C2", "classes"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"列 {col} 不存在，请检查 Excel 列名。")

# 找到 C1 != C2 的行
mask = df["C1"] != df["C2"]

# 取出这些行，复制一份用于新增
df_new = df.loc[mask].copy()

# 在新增行里交换 C1 和 C2
df_new[["C1", "C2"]] = df_new[["C2", "C1"]].values

# 原始数据 + 新增数据 纵向拼接
df_out = pd.concat([df, df_new], ignore_index=True)

# 保存到新文件（避免覆盖原文件）
output_path = "/home/zmm/7.17/pk/SDI-data-ssh/data_zmm/train_val4_added_swapped.xlsx"
df_out.to_excel(output_path, index=False)

print(f"完成：原始 {len(df)} 行，新增 {len(df_new)} 行，总共 {len(df_out)} 行")
print("保存路径：", output_path)
'''