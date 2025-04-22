import pandas as pd
import numpy as np

# 读取已有的CSV数据集
data = pd.read_csv('obesity.csv')

# 将非数值型数据转换为数值型（类别编码）
for column in data.columns:
    if data[column].dtype == 'object' or data[column].dtype.name == 'category':
        # 对类别型数据进行编码
        data[column] = pd.factorize(data[column])[0]

# 计算相关系数矩阵
corr_matrix = data.corr(method='pearson')  # 可以选择其他方法如'spearman'或'kendall'

# 提取上三角矩阵（不包括对角线）
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 计算平均相关性（忽略NaN值）
average_corr = upper_triangle.stack().mean()

# 计算最大相关性
max_corr = upper_triangle.stack().max()

# 计算相关性分布的其他统计量
median_corr = upper_triangle.stack().median()
std_corr = upper_triangle.stack().std()

# 打印结果
print(f"平均相关性: {average_corr:.4f}")
print(f"最大相关性: {max_corr:.4f}")
print(f"中位数相关性: {median_corr:.4f}")
print(f"标准差: {std_corr:.4f}")