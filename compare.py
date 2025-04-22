import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer
import time
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeRegressor


# 加载本地数据集（包含数值和字符属性）
data_path = "table6_with_missing.csv"
data = pd.read_csv(data_path)
#manual_cat_cols = ['resting ecg','exercise angina','oldpeak','ST slope','target'] 
#manual_cat_cols = ['sex','chest pain type','fasting blood sugar'] 
#manual_cat_cols = [] 
manual_cat_cols = ['FCVC','NCP','CH2O','FAF','TUE']  # 根据实际列名调整
# ------------------------------
# 方法1：迭代填补
def impute_iterative(data):
    start_time = time.time()
    imputed_data = data.copy()
    
    # 新增：手动指定分类列
    
    auto_cat_cols = imputed_data.select_dtypes(include='object').columns.tolist()  # 自动识别的字符型分类列
    cat_cols = manual_cat_cols + auto_cat_cols  # 合并分类列名单
    
    # 编码字符列
    encoder = OrdinalEncoder()
    encoded_data = imputed_data.copy()
    encoded_data[cat_cols] = encoder.fit_transform(imputed_data[cat_cols])
    
    # 填补
    imputer = IterativeImputer(random_state=0)
    imputed_encoded = imputer.fit_transform(encoded_data)
    
    # 逆转换
    imputed_data = pd.DataFrame(imputed_encoded, columns=data.columns)
    imputed_data[cat_cols] = encoder.inverse_transform(imputed_data[cat_cols].astype(int))
    end_time = time.time()
    elapsed_time = end_time - start_time
    return imputed_data, elapsed_time

# 方法2：改进版MICE（区分数值和字符的估计器）
def impute_mice_advanced(data):
    start_time = time.time()
    imputed_data = data.copy()
    
    # 新增：手动指定分类列
    auto_cat_cols = imputed_data.select_dtypes(include='object').columns.tolist()  # 自动识别的字符型分类列
    cat_cols = manual_cat_cols + auto_cat_cols  # 合并分类列名单
    num_cols = [col for col in imputed_data.columns if col not in cat_cols]  # 剩余列为数值列
    
    # 预处理管道：字符列使用Ordinal Encoding避免高维度
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OrdinalEncoder(), cat_cols)  # 使用OrdinalEncoder处理分类变量
        ])
    
    # 定义MICE估算器：数值用回归，字符用分类
    imputer = IterativeImputer(
        estimator=DecisionTreeRegressor(max_depth=5),  # 数值使用回归模型
        imputation_order='ascending',
        initial_strategy='most_frequent',  # 分类列用众数初始化
        max_iter=20,
        verbose=0
    )
    
    # 构建完整管道
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('imputer', imputer)
    ])
    
    # 执行填补
    imputed_encoded = pipeline.fit_transform(imputed_data)
    
    # 获取预处理中的OrdinalEncoder，用于逆转换
    ordinal_enc = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    
    # 重建DataFrame，确保列顺序与原数据一致
    imputed_df = pd.DataFrame(imputed_encoded, columns=num_cols + cat_cols)
    
    # 逆转换分类列（将数值转回为原始类别）
    imputed_df[cat_cols] = ordinal_enc.inverse_transform(imputed_df[cat_cols].astype(int))
    
    # 恢复列顺序，确保按原始数据的列顺序返回
    imputed_df = imputed_df[data.columns.tolist()]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    return imputed_df, elapsed_time

# 方法2：混合KNN填补
def impute_knn_mixed_weighted(data):
    start_time = time.time()
    imputed_data = data.copy()
    
    auto_cat_cols = imputed_data.select_dtypes(include='object').columns.tolist()  # 自动识别的字符型分类列
    cat_cols = manual_cat_cols + auto_cat_cols  # 合并分类列名单
    num_cols = [col for col in imputed_data.columns if col not in cat_cols]  # 剩余列为数值列
    
    # 填补字符列的缺失值为众数
    imputer_cat = SimpleImputer(strategy='most_frequent')
    data_filled = imputed_data.copy()
    data_filled[cat_cols] = imputer_cat.fit_transform(imputed_data[cat_cols])
    
    # One-Hot编码字符列
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_encoded = ohe.fit_transform(data_filled[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    
    # 合并数值和编码后的字符列
    combined_data = np.hstack([imputed_data[num_cols].values, cat_encoded])
    
    # 定义列权重：数值列权重=1，每个字符列的每个One-Hot特征权重=1/类别数
    col_weights = []
    for col in num_cols:
        col_weights.append(1.0)
    for col in cat_cols:
        # 使用填补后的数据计算类别数
        n_categories = len(data_filled[col].unique())
        col_weights.extend([1.0/n_categories] * n_categories)
    
    # 验证权重长度与合并后的列数一致
    if len(col_weights) != combined_data.shape[1]:
        print(f"实际列数: {combined_data.shape[1]}, 计算权重数: {len(col_weights)}")
        print("字符列编码后特征数详情:")
        for col in cat_cols:
            n_cat = len(data_filled[col].unique())
            print(f"- {col}: {n_cat} categories")
        raise ValueError("列权重长度与合并数据列数不一致，请检查字符列的唯一值数量")
    
    # 加权KNN填补
    imputer = KNNImputer(n_neighbors=3, weights='distance')
    imputed_combined = imputer.fit_transform(combined_data * np.array(col_weights))
    
    # 分离并逆转换字符列
    imputed_num = imputed_combined[:, :len(num_cols)]
    imputed_cat_encoded = imputed_combined[:, len(num_cols):]
    imputed_cat = ohe.inverse_transform(imputed_cat_encoded)
    
    # 重建DataFrame
    imputed_df = pd.DataFrame(
        np.hstack([imputed_num, imputed_cat]),
        columns=np.concatenate([num_cols, cat_cols])
    )
    # 恢复数据类型
    imputed_df[num_cols] = imputed_df[num_cols].apply(pd.to_numeric)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return imputed_df, elapsed_time

# 方法4：分类型众数填补
def impute_mode(data):
    start_time = time.time()
    imputed_data = data.copy()
    
    auto_cat_cols = imputed_data.select_dtypes(include='object').columns.tolist()  # 自动识别的字符型分类列
    cat_cols = manual_cat_cols + auto_cat_cols  # 合并分类列名单
    all_cols = cat_cols
    
    # 众数填补
    imputer = SimpleImputer(strategy='most_frequent')
    imputed_data[all_cols] = imputer.fit_transform(imputed_data[all_cols])
    end_time = time.time()
    elapsed_time = end_time - start_time
    return imputed_data, elapsed_time

# 方法5：MissForest
def impute_missforest(data):
    start_time = time.time()
    imputed_data = data.copy()
    
    # 分离数值列和分类列
    auto_cat_cols = imputed_data.select_dtypes(include='object').columns.tolist()
    cat_cols = manual_cat_cols + auto_cat_cols
    num_cols = [col for col in imputed_data.columns if col not in cat_cols]
    
    # 预处理器：数值列保持原样，分类列编码为Ordinal
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OrdinalEncoder(), cat_cols)
        ])
    
    # 使用随机森林的迭代填补
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, random_state=0),
        initial_strategy='mean',
        max_iter=10,
        verbose=0
    )
    
    # 构建管道
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('imputer', imputer)
    ])
    
    # 执行填补
    imputed_encoded = pipeline.fit_transform(imputed_data)
    
    # 逆转换分类列
    imputed_df = pd.DataFrame(imputed_encoded, columns=num_cols + cat_cols)
    imputed_df[cat_cols] = imputed_df[cat_cols].round().astype(int)  # 分类列取整
    imputed_df[cat_cols] = pipeline.named_steps['preprocessor'].named_transformers_['cat'].inverse_transform(imputed_df[cat_cols])
    
    # 恢复列顺序
    imputed_df = imputed_df[data.columns.tolist()]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    return imputed_df, elapsed_time

# ------------------------------
# 执行所有方法并保存结果
# ------------------------------
methods = {
    #"Mode": impute_mode(data),
    #"Iterative": impute_iterative(data),
    #"MICE_Advanced": impute_mice_advanced(data),
    #"KNN_Weighted": impute_knn_mixed_weighted(data)
    #"MissForest": impute_missforest(data)
}

for name, result in methods.items():
    imputed_data, elapsed_time = result
    imputed_data.to_csv(f"table6_imputed_{name}.csv", index=False)
    print(f"填补结果已保存到 table6_imputed_{name}.csv")
    print(f"填补用时: {elapsed_time:.4f} 秒")

print("\n字符列填补验证（示例）：")
# 自动获取第一个分类列名
cat_cols = data.select_dtypes(include='object').columns.tolist()

if len(cat_cols) > 0:
    target_cat_col = cat_cols[0]  # 使用第一个分类列
    original_cat = data[target_cat_col].head(5)
    
    for name, result in methods.items():
        imputed_data, elapsed_time = result
        imputed_cat = imputed_data[target_cat_col].head(5)
        print(f"\nMethod: {name}")
        print(pd.concat([original_cat, imputed_cat], axis=1, 
                        keys=['Original', 'Imputed']))
else:
    print("警告：数据集中未找到分类列，跳过验证")