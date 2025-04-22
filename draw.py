import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 从文档中提取数据，按表格整理为字典
data = {
    "Table1": {
        "OURS": {"Accuracy": 0.8750, "Recall": 0.8642, "F1": 0.8696, "RMSE": 43.0000},
        "MODE": {"Accuracy": 0.8519, "Recall": 0.8519, "F1": 0.8519, "RMSE": 21.0000},
        "Iterative": {"Accuracy": 0.8519, "Recall": 0.8519, "F1": 0.8519, "RMSE": 174.8466},
        "KNN": {"Accuracy": 0.8519, "Recall": 0.8519, "F1": 0.8519, "RMSE": 121.6667},
        "MICE": {"Accuracy": 0.8889, "Recall": 0.8889, "F1": 0.8889, "RMSE": 168.7701},
        "MissForest": {"Accuracy": 0.9012, "Recall": 0.9012, "F1": 0.9012, "RMSE": None}
    },
    "Table2": {
        "OURS": {"Accuracy": 0.7958, "Recall": 0.7925, "F1": 0.7942, "RMSE": 18.0000},
        "MODE": {"Accuracy": 0.6598, "Recall": 0.6598, "F1": 0.6598, "RMSE": 82.0000},
        "Iterative": {"Accuracy": 0.5311, "Recall": 0.5311, "F1": 0.5311, "RMSE": 8.2328},
        "KNN": {"Accuracy": 0.6598, "Recall": 0.6598, "F1": 0.6598, "RMSE": 17.4254},
        "MICE": {"Accuracy": 0.6141, "Recall": 0.6141, "F1": 0.6141, "RMSE": 14.2353},
        "MissForest": {"Accuracy": 0.8174, "Recall": 0.8174, "F1": 0.8174, "RMSE": None}
    },
    "Table3": {
        "OURS": {"Accuracy": 0.7158, "Recall": 0.7158, "F1": 0.7158, "RMSE": 312.2028},
        "MODE": {"Accuracy": 0.6448, "Recall": 0.6448, "F1": 0.6448, "RMSE": 0.7923},
        "Iterative": {"Accuracy": 0.3934, "Recall": 0.3934, "F1": 0.3934, "RMSE": 168.9044},
        "KNN": {"Accuracy": 0.0874, "Recall": 0.0874, "F1": 0.0874, "RMSE": 148.9470},
        "MICE": {"Accuracy": 0.4153, "Recall": 0.4153, "F1": 0.4153, "RMSE": 133.3740},
        "MissForest": {"Accuracy": 0.6448, "Recall": 0.6448, "F1": 0.6448, "RMSE": None}
    },
    "Table4": {
        "OURS": {"Accuracy": 0.6977, "Recall": 0.6954, "F1": 0.6965, "RMSE": 35.1546},
        "MODE": {"Accuracy": 0.5099, "Recall": 0.5099, "F1": 0.5099, "RMSE": 0.9750},
        "Iterative": {"Accuracy": 0.4603, "Recall": 0.4603, "F1": 0.4603, "RMSE": 171.7312},
        "KNN": {"Accuracy": 0.3377, "Recall": 0.3377, "F1": 0.3377, "RMSE": 179.2215},
        "MICE": {"Accuracy": 0.4901, "Recall": 0.4901, "F1": 0.4901, "RMSE": 119.4963},
        "MissForest": {"Accuracy": 0.5927, "Recall": 0.5927, "F1": 0.5927, "RMSE": None}
    },
    "Table5": {
        "OURS": {"Accuracy": 0.8111, "Recall": 0.8111, "F1": 0.8111, "RMSE": 1066.7334},
        "MODE": {"Accuracy": 0.6774, "Recall": 0.6774, "F1": 0.6774, "RMSE": 1311.0648},
        "Iterative": {"Accuracy": 0.4747, "Recall": 0.4747, "F1": 0.4747, "RMSE": 278.7597},
        "KNN": {"Accuracy": 0.4055, "Recall": 0.4055, "F1": 0.4055, "RMSE": 365.2016},
        "MICE": {"Accuracy": 0.5714, "Recall": 0.5714, "F1": 0.5714, "RMSE": 209.5743},
        "MissForest": {"Accuracy": 0.8986, "Recall": 0.8986, "F1": 0.8986, "RMSE": None}
    },
    "Table6": {
        "OURS": {"Accuracy": 0.6460, "Recall": 0.6455, "F1": 0.6457, "RMSE": 71.0994},
        "MODE": {"Accuracy": 0.5668, "Recall": 0.5655, "F1": 0.5662, "RMSE": 0.9256},
        "Iterative": {"Accuracy": 0.3051, "Recall": 0.3051, "F1": 0.3051, "RMSE": 147.5026},
        "KNN": {"Accuracy": 0.1529, "Recall": 0.1529, "F1": 0.1529, "RMSE": 64.5245},
        "MICE": {"Accuracy": 0.3969, "Recall": 0.3969, "F1": 0.3969, "RMSE": 65.2416},
        "MissForest": {"Accuracy": 0.5129, "Recall": 0.5129, "F1": 0.5129, "RMSE": None}
    }
}

# 转换为DataFrame
df = pd.DataFrame.from_dict({(i,j): data[i][j] 
                           for i in data.keys() 
                           for j in data[i].keys()}, 
                          orient='index')
df.index = pd.MultiIndex.from_tuples(df.index, names=['Table', 'Method'])

# ------------ 可视化 ------------
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 学术论文常用字体
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14
})

# 绘制RMSE柱状图
fig, ax = plt.subplots(figsize=(12, 8))
df['RMSE'].unstack().plot(kind='bar', ax=ax, width=0.8, 
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
ax.set_ylabel('RMSE', fontweight='bold')
ax.set_xlabel('Table', fontweight='bold')
ax.set_ylim(0, 1500)
ax.set_title('RMSE Comparison Across Tables', fontweight='bold')
ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('rmse_comparison.png', dpi=300, bbox_inches='tight')

plt.show()

# 绘制准确率柱状图
fig, ax = plt.subplots(figsize=(12, 8))
df['Accuracy'].unstack().plot(kind='bar', ax=ax, width=0.8, 
                            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_xlabel('Table', fontweight='bold')
ax.set_ylim(0, 1.0)
ax.set_title('Accuracy Comparison Across Tables', fontweight='bold')
ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')

# 绘制F1分数折线图
fig, ax = plt.subplots(figsize=(12, 8))
methods = ['OURS', 'MODE', 'Iterative', 'KNN', 'MICE', 'MissForest']
markers = ['o', 's', 'D', '^', 'v', 'p']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, method in enumerate(methods):
    df_method = df.xs(method, level='Method')['F1']
    ax.plot(df_method.index, df_method.values, 
            marker=markers[idx], linestyle='--', 
            color=colors[idx], label=method, markersize=8)

ax.set_ylabel('F1 Score', fontweight='bold')
ax.set_xlabel('Table', fontweight='bold')
ax.set_ylim(0, 1.0)
ax.set_title('F1 Score Trend Across Tables', fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(ticks=range(len(df_method.index)), labels=df_method.index)
plt.tight_layout()
plt.savefig('f1_trend.png', dpi=300, bbox_inches='tight')

# 绘制召回率柱状图
fig, ax = plt.subplots(figsize=(12, 8))
df['Recall'].unstack().plot(kind='bar', ax=ax, width=0.8, 
                          color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
ax.set_ylabel('Recall', fontweight='bold')
ax.set_xlabel('Table', fontweight='bold')
ax.set_ylim(0, 1.0)
ax.set_title('Recall Comparison Across Tables', fontweight='bold')
ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('recall_comparison.png', dpi=300, bbox_inches='tight')

# ------------ 输出数值表格 ------------
print("\n数值表格（LaTeX格式）：")
print(df.unstack().to_latex(float_format="%.3f", caption="Experimental Results", label="tab:results"))

plt.show()