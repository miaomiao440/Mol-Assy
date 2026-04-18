# -*- coding: utf-8 -*-
"""
#欠采样
#对数据进行欠采样，随机的减少负样本数量，使他与正样本数量相同
import pandas as pd

# 文件路径
input_path = "./datasets/all_dataset.xlsx"
output_path = "./datasets/undersampling/balanced_dataset.xlsx"

df = pd.read_excel(input_path)
assert set(['sonosensitizer', 'drug', 'classes']).issubset(df.columns), "列名不匹配"

#分离正负样本
pos_df = df[df['classes'] == 1]
neg_df = df[df['classes'] == 0]
neg_sampled = neg_df.sample(n=len(pos_df), random_state=42)

balanced_df = pd.concat([pos_df, neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

balanced_df.to_excel(output_path, index=False)

print(f"欠采样完成：正样本 {len(pos_df)}，负样本 {len(neg_sampled)}，共 {len(balanced_df)} 条")
print(f"已保存到：{output_path}")



#对数据进行5折划分
import pandas as pd
from sklearn.model_selection import StratifiedKFold

input_path = "./datasets/undersampling/balanced_dataset.xlsx"
output_dir = "./datasets/undersampling/"

# 读取上述欠采样后的数据
df = pd.read_excel(input_path)

#5折
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['classes']), 1):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_path = f"{output_dir}fold{fold}_train.xlsx"
    val_path = f"{output_dir}fold{fold}_val.xlsx"
    
    train_df.to_excel(train_path, index=False)
    val_df.to_excel(val_path, index=False)

    print(f"Fold {fold} 完成：Train={len(train_df)}，Val={len(val_df)}")
    print(f"保存路径：{train_path}, {val_path}")


#划分出20%的测试集+剩余的进行5折划分
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

input_path = "./datasets/undersampling/balanced_dataset1.xlsx"
output_dir = "./datasets/undersampling/"

df = pd.read_excel(input_path)


train_val_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['classes'], 
    random_state=42
)

# 保存测试集
test_path = f"{output_dir}test.xlsx"
test_df.to_excel(test_path, index=False)
print(f"测试集划分完成：Test={len(test_df)}，保存路径：{test_path}")

#对剩下的训练验证数据进行StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df['classes']), 1):
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
    
    train_path = f"{output_dir}sfold{fold}_train.xlsx"
    val_path = f"{output_dir}sfold{fold}_val.xlsx"
    
    train_df.to_excel(train_path, index=False)
    val_df.to_excel(val_path, index=False)

    print(f"Fold {fold} 完成：Train={len(train_df)}，Val={len(val_df)}")
    print(f"保存路径：{train_path}, {val_path}")

"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split

input_file = "./datasets/weight.xlsx"
output_dir = "./datasets/undersampling/julie/"

df = pd.read_excel(input_file)

train_ratio = 0.7
val_ratio = 0.15  

# 划分训练集和临时集
train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42, shuffle=True)

# 再从临时集中划分验证集和测试集
val_size = val_ratio / (1 - train_ratio)
val_df, test_df = train_test_split(temp_df, test_size=(1 - val_size), random_state=42, shuffle=True)

os.makedirs(output_dir, exist_ok=True)
train_df.to_excel(os.path.join(output_dir, "all_train.xlsx"), index=False)
val_df.to_excel(os.path.join(output_dir, "all_val.xlsx"), index=False)
test_df.to_excel(os.path.join(output_dir, "all_test.xlsx"), index=False)

print("数据集划分完成并已保存：")
print(f"训练集: {len(train_df)} 条")
print(f"验证集: {len(val_df)} 条")
print(f"测试集: {len(test_df)} 条")



'''
"""
讲字符串转化为小写，因为
"""
import pandas as pd

# 文件路径
input_path = "./datasets/undersampling/balanced_dataset.xlsx"
output_path = "./datasets/undersampling/balanced_dataset1.xlsx"  

# 读取 Excel 文件
df = pd.read_excel(input_path)

# 将所有列名转为小写
df.columns = df.columns.str.lower()

# 将所有字符串类型的单元格转为小写
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.lower()

# 保存结果
df.to_excel(output_path, index=False)
print(f"已将所有内容转换为小写并保存到：{output_path}")
'''