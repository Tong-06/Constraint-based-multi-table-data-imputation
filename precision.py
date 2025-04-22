import pandas as pd
import numpy as np

def calculate_imputation_metrics(original_data_path, missing_data_path, imputed_data_path):
    # 读取原始数据、缺失数据和填补后的数据
    original_data = pd.read_csv(original_data_path)
    missing_data = pd.read_csv(missing_data_path)
    imputed_data = pd.read_csv(imputed_data_path)
    
    # 确保数据的列名和索引一致
    original_data.columns = imputed_data.columns
    original_data.index = imputed_data.index
    
    # 找到在缺失数据集中被标记为缺失的位置（即NaN的位置）
    missing_mask = missing_data.isnull()
    
    # 统计原始缺失值的总数
    total_missing = missing_mask.sum().sum()
    
    # 统计填补后的数据中实际填补的值的数量（即填补的总数量）
    imputed_values = imputed_data.where(missing_mask)
    total_imputed = imputed_values.notnull().sum().sum()
    
    # 统计正确填补的值的数量
    correct_imputed = (imputed_data.where(missing_mask) == original_data.where(missing_mask)).sum().sum()
    
    # 计算准确率
    accuracy = correct_imputed / total_imputed if total_imputed != 0 else 0
    
    # 计算填补率（填补的值的数量 / 原始缺失值的总数）
    imputation_rate = total_imputed / total_missing if total_missing != 0 else 0
    
    # 计算召回率（正确填补的值的数量 / 原始缺失值的总数）
    recall = correct_imputed / total_missing if total_missing != 0 else 0
    
    # 计算 F 分数（准确率和召回率的调和平均）
    if accuracy + recall != 0:
        f_score = 2 * (accuracy * recall) / (accuracy + recall)
    else:
        f_score = 0
    
    # 计算 RMSE（根均方误差）
    # 确保只计算数值型列的 RMSE
    numeric_columns = original_data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        original_numeric = original_data[numeric_columns].where(missing_mask)
        imputed_numeric = imputed_data[numeric_columns].where(missing_mask)
        
        # 确保 imputed_numeric 也是数值型
        imputed_numeric = imputed_numeric.apply(pd.to_numeric, errors='coerce')
        
        # 计算 RMSE
        squared_errors = (original_numeric - imputed_numeric) ** 2
        mean_squared_error = squared_errors.mean().mean()
        rmse = np.sqrt(mean_squared_error)
    else:
        rmse = 0  # 如果没有数值型列，RMSE设为0
    
    return accuracy, imputation_rate, recall, f_score, rmse, total_missing, total_imputed, correct_imputed

# 计算table1的填补指标
accuracy_table1, imputation_rate_table1, recall_table1, f_score_table1, rmse_table1, total_missing_table1, total_imputed_table1, correct_table1 = calculate_imputation_metrics(
    'table1.csv', 'table1_with_missing.csv', 'table1_imputed.csv'
)
print(f"Table1填补准确率: {accuracy_table1:.4f}")
print(f"Table1填补率: {imputation_rate_table1:.4f}")
print(f"Table1召回率: {recall_table1:.4f}")
print(f"Table1 F 分数: {f_score_table1:.4f}")
print(f"Table1 RMSE: {rmse_table1:.4f}")
print(f"Table1原始缺失值总数: {total_missing_table1}, 填补的总数量: {total_imputed_table1}, 正确填补的数量: {correct_table1}\n")

# 计算table2的填补指标
accuracy_table2, imputation_rate_table2, recall_table2, f_score_table2, rmse_table2, total_missing_table2, total_imputed_table2, correct_table2 = calculate_imputation_metrics(
    'table2.csv', 'table2_with_missing.csv', 'table2_imputed.csv'
)
print(f"Table2填补准确率: {accuracy_table2:.4f}")
print(f"Table2填补率: {imputation_rate_table2:.4f}")
print(f"Table2召回率: {recall_table2:.4f}")
print(f"Table2 F 分数: {f_score_table2:.4f}")
print(f"Table2 RMSE: {rmse_table2:.4f}")
print(f"Table2原始缺失值总数: {total_missing_table2}, 填补的总数量: {total_imputed_table2}, 正确填补的数量: {correct_table2}")

# 计算table3的填补指标
accuracy_table3, imputation_rate_table3, recall_table3, f_score_table3, rmse_table3, total_missing_table3, total_imputed_table3, correct_table3 = calculate_imputation_metrics(
    'table3.csv', 'table3_with_missing.csv', 'table3_imputed.csv'
)
print(f"Table3填补准确率: {accuracy_table3:.4f}")
print(f"Table3填补率: {imputation_rate_table3:.4f}")
print(f"Table3召回率: {recall_table3:.4f}")
print(f"Table3 F 分数: {f_score_table3:.4f}")
print(f"Table3 RMSE: {rmse_table3:.4f}")
print(f"Table3原始缺失值总数: {total_missing_table3}, 填补的总数量: {total_imputed_table3}, 正确填补的数量: {correct_table3}\n")

# 计算table4的填补指标
accuracy_table4, imputation_rate_table4, recall_table4, f_score_table4, rmse_table4, total_missing_table4, total_imputed_table4, correct_table4 = calculate_imputation_metrics(
    'table4.csv', 'table4_with_missing.csv', 'table4_imputed.csv'
)
print(f"Table4填补准确率: {accuracy_table4:.4f}")
print(f"Table4填补率: {imputation_rate_table4:.4f}")
print(f"Table4召回率: {recall_table4:.4f}")
print(f"Table4 F 分数: {f_score_table4:.4f}")
print(f"Table4 RMSE: {rmse_table4:.4f}")
print(f"Table4原始缺失值总数: {total_missing_table4}, 填补的总数量: {total_imputed_table4}, 正确填补的数量: {correct_table4}")

# 计算table5的填补指标
accuracy_table5, imputation_rate_table5, recall_table5, f_score_table5, rmse_table5, total_missing_table5, total_imputed_table5, correct_table5 = calculate_imputation_metrics(
    'table5.csv', 'table5_with_missing.csv', 'table5_imputed.csv'
)
print(f"Table5填补准确率: {accuracy_table5:.4f}")
print(f"Table5填补率: {imputation_rate_table5:.4f}")
print(f"Table5召回率: {recall_table5:.4f}")
print(f"Table5 F 分数: {f_score_table5:.4f}")
print(f"Table5 RMSE: {rmse_table5:.4f}")
print(f"Table5原始缺失值总数: {total_missing_table5}, 填补的总数量: {total_imputed_table5}, 正确填补的数量: {correct_table5}\n")

# 计算table6的填补指标
accuracy_table6, imputation_rate_table6, recall_table6, f_score_table6, rmse_table6, total_missing_table6, total_imputed_table6, correct_table6 = calculate_imputation_metrics(
    'table6.csv', 'table6_with_missing.csv', 'table6_imputed.csv'
)
print(f"Table6填补准确率: {accuracy_table6:.4f}")
print(f"Table6填补率: {imputation_rate_table6:.4f}")
print(f"Table6召回率: {recall_table6:.4f}")
print(f"Table6 F 分数: {f_score_table6:.4f}")
print(f"Table6 RMSE: {rmse_table6:.4f}")
print(f"Table6原始缺失值总数: {total_missing_table6}, 填补的总数量: {total_imputed_table6}, 正确填补的数量: {correct_table6}")

