import pandas as pd
import numpy as np
from Levenshtein import distance as levenshtein_distance
from collections import defaultdict
import itertools 
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import time

class StrictSINDImputer:
    def __init__(self, table1, table2, sind_pairs=('A', 'A\''), key_cols=['B', 'C'], 
             K=2, delta=0.8, non_sind_method='approximate'):

        # 初始化 SIND 候选值存储结构，用于存储 SIND 列的候选填补值
        self.sind_candidates = defaultdict(list)  
        
        # 初始化非 SIND 候选值存储结构，用于存储非 SIND 列的候选填补值
        self.non_sind_candidates = defaultdict(dict)
                
        self.t1 = table1.copy()
        self.t2 = table2.copy()
        
        # 在原始表中新增 '_missing_mask' 列，标记每行是否包含缺失值
        self.t1['_missing_mask'] = self.t1.isnull().any(axis=1)
        self.t2['_missing_mask'] = self.t2.isnull().any(axis=1)
        
        # 解包 SIND 列的标识符对
        self.A, self.A_prime = sind_pairs
        
        # 将关键列转换为列表
        self.key_cols = list(key_cols)
        
        # 设置控制阈值或数量的参数
        self.K = K
        
        # 设置相似度阈值
        self.delta = delta
        
        # 设置非 SIND 列的填补方法
        self.non_sind_method = non_sind_method
        
        # 初始化相似度缓存字典，用于存储已计算的相似度
        self.sim_cache = {}
        
        # 初始化链接映射存储结构，用于存储表之间的连接关系
        self.link_map = defaultdict(list)
        
        # 初始化非 SIND 候选值存储结构
        self.non_sind_candidates = defaultdict(dict)
        
        # 保留原始数据表的数据类型信息，便于后续引用
        self.table1_original_dtypes = table1.dtypes.to_dict()
        self.table2_original_dtypes = table2.dtypes.to_dict()
        
        # 保留原始数据表的副本，便于后续引用
        self.original_table1 = table1.copy()
        self.original_table2 = table2.copy()
        
        # 统计并保留原始数据表中的缺失值数量
        self.original_missing_t1 = table1.isnull().sum().sum()
        self.original_missing_t2 = table2.isnull().sum().sum()

        self.original_missing_mask_t1 = table1.isnull()  # 新增
        self.original_missing_mask_t2 = table2.isnull()  # 新增

        # 新增：存储标准化器与分类列标识
        self.scaler = StandardScaler()
        self.categorical_cols = {
            't1': [col for col in table1.columns if not pd.api.types.is_numeric_dtype(table1[col])],
            't2': [col for col in table2.columns if not pd.api.types.is_numeric_dtype(table2[col])]
        }

    # ------------------- 第一步：候选值选取 -------------------
    def _generate_sind_candidates(self):

        # 处理表1中 SIND 列缺失的情况
        for idx, row in self.t1[self.t1[self.A].isnull()].iterrows():
            # 调用 _get_knn_candidates_for_sind 方法生成候选值
            # 参数说明：
            # - row: 当前行
            # - src_table: 当前表（表1）
            # - tgt_table: 目标表（表2）
            # - src_col: 当前表的 SIND 列
            # - tgt_col: 目标表的 SIND 列
            candidates = self._get_knn_candidates_for_sind(row, src_table=self.t1, tgt_table=self.t2, 
                                                        src_col=self.A, tgt_col=self.A_prime)
            # 将生成的候选值存储到 self.sind_candidates 字典中
            # 键为 ('t1', idx)，表示表1中索引为 idx 的行的候选值
            self.sind_candidates[('t1', idx)] = candidates

        # 处理表2中 SIND 列缺失的情况
        for idx, row in self.t2[self.t2[self.A_prime].isnull()].iterrows():
            candidates = self._get_knn_candidates_for_sind(row, src_table=self.t2, tgt_table=self.t1,
                                                        src_col=self.A_prime, tgt_col=self.A)
            # 将生成的候选值存储到 self.sind_candidates 字典中
            # 键为 ('t2', idx)，表示表2中索引为 idx 的行的候选值
            self.sind_candidates[('t2', idx)] = candidates

    def _get_knn_candidates_for_sind(self, row, src_table, tgt_table, src_col, tgt_col):

        # 筛选目标表中共同属性和 SIND 列均非缺失值的行
        valid_targets = tgt_table.dropna(subset=self.key_cols + [tgt_col])
        candidates = []

        # 遍历所有有效的候选行
        for _, target_row in valid_targets.iterrows():
            # 计算当前行与目标行在共同属性上的相似度
            key_sim = self._calc_similarity(row[self.key_cols], target_row[self.key_cols])
            
            # 如果相似度大于或等于阈值 delta，则将目标行的 SIND 列值作为候选值
            if key_sim >= self.delta:
                candidates.append({
                    'value': target_row[tgt_col],  # 候选值
                    'src_idx': target_row.name,   # 候选值所在的行索引
                    'key_sim': key_sim           # 关键列的相似度
                })

        # 按关键列相似度降序排序，并返回前 K 个候选值
        return sorted(candidates, key=lambda x: x['key_sim'], reverse=True)[:self.K]
    
    def _calc_similarity(self, vec1, vec2):
        
        total = 0  # 用于累加所有特征的相似度
        valid_features = 0  # 用于计数有效特征的数量

        # 遍历两个向量中的每个值（v1 和 v2），并获取对应的列名
        for i, (v1, v2) in enumerate(zip(vec1, vec2)):
            col_name = self.key_cols[i]  # 获取当前列的名称

            # 如果任一值为缺失值，则跳过当前特征的计算
            if pd.isnull(v1) or pd.isnull(v2):
                continue

            # 根据值的类型计算相似度
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # 如果两个值都是数值类型，计算数值相似度
                max_val = max(abs(v1), abs(v2)) or 1  # 避免除以零
                sim = 1 - abs(v1 - v2) / max_val  # 相似度公式：1 - 差值 / 最大值
            else:
                # 如果两个值是字符串类型，计算字符串相似度
                str1 = str(v1).lower()  # 将字符串转为小写以忽略大小写差异
                str2 = str(v2).lower()
                max_len = max(len(str1), len(str2)) or 1  # 避免除以零
                sim = 1 - levenshtein_distance(str1, str2) / max_len  # 使用 Levenshtein 距离计算相似度

            total += sim  # 将当前特征的相似度累加到总和中
            valid_features += 1  # 增加有效特征计数

        # 如果有有效特征，则返回平均相似度；否则返回0
        return total / valid_features if valid_features > 0 else 0
    
    def _generate_precise_non_sind_candidates(self):
        
        # 处理表1的非SIND列（排除SIND列 self.A）
        for col in set(self.t1.columns) - {self.A}:
            # 获取当前列中缺失值所在的行索引
            missing_indices = self.t1[self.t1[col].isnull()].index
            for idx in missing_indices:
                # 调用 _get_knn_candidates_for_non_sind 方法为当前缺失值生成候选值
                candidates = self._get_knn_candidates_for_non_sind(self.t1, col, idx)
                # 将候选值存储到 self.non_sind_candidates['t1'] 中
                # 键为 (col, idx)，表示表1中列 col、索引为 idx 的缺失值的候选值
                self.non_sind_candidates['t1'][(col, idx)] = candidates

        # 处理表2的非SIND列（排除SIND列 self.A_prime）
        for col in set(self.t2.columns) - {self.A_prime}:
            # 获取当前列中缺失值所在的行索引
            missing_indices = self.t2[self.t2[col].isnull()].index
            for idx in missing_indices:
                # 调用 _get_knn_candidates_for_non_sind 方法为当前缺失值生成候选值
                candidates = self._get_knn_candidates_for_non_sind(self.t2, col, idx)
                # 将候选值存储到 self.non_sind_candidates['t2'] 中
                # 键为 (col, idx)，表示表2中列 col、索引为 idx 的缺失值的候选值
                self.non_sind_candidates['t2'][(col, idx)] = candidates

    def _get_knn_candidates_for_non_sind(self, table, col, idx):

        candidates_df = table.dropna(subset=[col])
        
        # 如果候选数据为空，直接返回空列表
        if candidates_df.empty:
            return []
        
        # 获取当前缺失值所在的行
        current_row = table.loc[idx]
        similarities = []
        
        # 遍历候选数据，计算当前行与每行候选数据的相似度
        for cand_idx, cand_row in candidates_df.iterrows():
            # 计算相似度，忽略当前列（col）
            sim = self._calc_similarity_with_missing(current_row, cand_row, exclude_col=col)
            # 获取候选值
            value = cand_row[col]
            # 将相似度、候选值及其来源索引存储为三元组
            similarities.append((sim, value, cand_idx))
        
        # 按相似度降序排序并取前K个候选值
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = similarities[:self.K]
        
        # 返回格式：[(候选值, 来源索引), ...]
        return [(v, src_idx) for sim, v, src_idx in top_k]
    
    def _calc_similarity_with_missing(self, row1, row2, exclude_col=None):

        total_sim = 0  # 初始化总相似度
        valid_features = 0  # 初始化有效特征计数
        
        # 动态获取所有有效列（排除后缀冲突）
        # 选择在两行中都存在的列，并且排除以'_t1'或'_t2'结尾的列（这些是合并时生成的后缀）
        cols = [col for col in row1.index if col in row2.index and not col.endswith(('_t1', '_t2'))]
        
        # 遍历每个有效列
        for col in cols:
            # 跳过需要排除的列和标记缺失值的列
            if col == exclude_col or col == '_missing_mask':
                continue
            v1 = row1[col]  # 获取第一行的值
            v2 = row2[col]  # 获取第二行的值
            
            # 跳过存在缺失值或特殊编码缺失值（-1）的列
            if pd.isnull(v1) or pd.isnull(v2) or v1 == -1 or v2 == -1:
                continue
            
            # 计算当前列的相似度
            sim = self._calc_single_feature_similarity(col, v1, v2)
            total_sim += sim  # 累加相似度
            valid_features += 1  # 增加有效特征计数
        
        # 返回平均相似度，如果有效特征数为0，返回0
        return total_sim / valid_features if valid_features > 0 else 0

    def _calc_single_feature_similarity(self, col, v1, v2):
       
        # 检查是否任一值为空，如果为空则直接返回相似度为0
        if pd.isnull(v1) or pd.isnull(v2):
            return 0

        # 根据值的类型计算相似度
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            # 如果两个值都是数值类型，计算数值相似度
            max_val = max(abs(v1), abs(v2)) or 1  # 避免除以零
            # 相似度公式：1 - 差值 / 最大值
            return 1 - abs(v1 - v2) / max_val
        else:
            # 如果两个值是字符串类型，计算字符串相似度
            str1 = str(v1).lower()  # 将字符串转为小写以忽略大小写差异
            str2 = str(v2).lower()
            max_len = max(len(str1), len(str2)) or 1  # 避免除以零
            # 使用 Levenshtein 距离计算相似度
            return 1 - levenshtein_distance(str1, str2) / max_len

    # ------------------- 第二步：连接多表 -------------------
    def _link_tables(self):

        self.link_combinations = []
        
        # 生成SIND候选组合时直接创建合并视图
        sind_combinations = self._generate_sind_combinations()
        
        for combination in sind_combinations:
            # 创建带原始索引的临时表并正确重命名索引列
            temp_t1 = self.t1.copy().reset_index().rename(
                columns=lambda x: f"{x}_t1" if x != 'index' else 'index_t1'
            )
            temp_t2 = self.t2.copy().reset_index().rename(
                columns=lambda x: f"{x}_t2" if x != 'index' else 'index_t2'
            )
            
            # 应用当前SIND候选值
            self._apply_sind_combination(temp_t1, temp_t2, combination)
            
            # 执行内连接，指定重命名后的列名
            merged = pd.merge(
                temp_t1, temp_t2,
                left_on=[f"{self.A}_t1"],  # 重命名后的列名
                right_on=[f"{self.A_prime}_t2"],  # 重命名后的列名
                how='inner'
            )
            
            if not merged.empty:
                # 构建包含完整连接上下文的link_map
                link_map = defaultdict(list)
                # 使用正确的列名进行分组
                merged_grouped = merged.groupby('index_t1')['index_t2'].apply(list)
                for t1_idx, t2_indices in merged_grouped.items():
                    link_map[t1_idx].extend(t2_indices)
                
                # 记录完整的合并视图
                self.link_combinations.append({
                    'sind_values': combination,
                    'link_map': link_map,
                    'merged_view': merged
                })

    def _apply_sind_combination(self, temp_t1, temp_t2, combination):
        
        for (table_key, idx), candidate in combination.items():
            if table_key == 't1':
                # 使用正确的索引列名 'index_t1'
                temp_t1.loc[temp_t1['index_t1'] == idx, self.A] = candidate['value']
            elif table_key == 't2':
                # 使用正确的索引列名 'index_t2'
                temp_t2.loc[temp_t2['index_t2'] == idx, self.A_prime] = candidate['value']

    
    def _generate_sind_combinations(self):
        combinations = []  # 存储生成的候选组合
        items = list(self.sind_candidates.items())  # 获取候选项的键值对列表

        if not items:
            return [{}]  # 如果没有候选项，返回一个空组合

        # 逐步构建组合避免内存爆炸
        current = {}  # 当前组合
        for key, candidates in items:
            if not combinations:
                # 初始化为每个候选单独的组合
                for cand in candidates:
                    new_combo = current.copy()  # 复制当前组合
                    new_combo[key] = cand  # 添加当前候选值
                    combinations.append(new_combo)  # 将新组合添加到列表中
            else:
                # 逐步扩展现有组合
                new_combinations = []
                for combo in combinations:  # 遍历现有组合
                    for cand in candidates:  # 遍历当前候选值
                        new_combo = combo.copy()  # 复制现有组合
                        new_combo[key] = cand  # 添加当前候选值
                        new_combinations.append(new_combo)  # 将新组合添加到新列表中
                combinations = new_combinations  # 更新组合列表为新生成的组合

        return combinations  # 返回所有生成的候选组合
    
    # ------------------- 第三步：选择最优填补方案 -------------------   
    def _approximate_impute(self):
        # 初始化最佳全局相似度和对应的结果矩阵
        best_global_sim = -np.inf
        best_t1 = self.t1.copy()
        best_t2 = self.t2.copy()
        
        # 并行计算每个连接方案的相似度
        # 使用所有可用CPU核心加速计算
        results = Parallel(n_jobs=-1)(
            delayed(self._evaluate_link_info)(link_info)
            for link_info in self.link_combinations
        )
        
        # 遍历所有计算结果，选择具有最高相似度的方案
        for global_sim, temp_t1, temp_t2 in results:
            if global_sim > best_global_sim:
                best_global_sim = global_sim
                best_t1, best_t2 = temp_t1, temp_t2
        
        # 返回最佳匹配结果
        return best_t1, best_t2

    def _evaluate_link_info(self, link_info):
        # 获取合并后的视图数据
        merged = link_info['merged_view']
        
        # 生成候选匹配（近似填补策略）
        candidates = self._generate_approximate_candidates(merged)
        
        # 计算全局相似度
        global_sim = self._calc_approximate_similarity(merged, candidates)
        
        # 应用候选匹配到原始数据
        temp_t1, temp_t2 = self._apply_approximate_candidates(merged, candidates)
        
        # 返回相似度评分和处理后的数据
        return global_sim, temp_t1, temp_t2

    def _generate_approximate_candidates(self, merged):
        # 初始化候选字典
        candidates = defaultdict(list)
        
        # 筛选完整行（没有缺失值的行）
        complete_mask = (~merged['_missing_mask_t1']) & (~merged['_missing_mask_t2'])
        complete_rows = merged[complete_mask]

        # 遍历所有合并行
        for idx, row in merged.iterrows():
            # 跳过已经完整的行
            if not (row['_missing_mask_t1'] or row['_missing_mask_t2']):
                continue
                    
            # 计算当前行与所有完整行的相似度
            similarities = []
            for comp_idx, comp_row in complete_rows.iterrows():
                sim = self._calc_merged_row_similarity(row, comp_row)
                similarities.append((sim, comp_idx))
            
            # 按相似度排序并选择最相似的行
            similarities.sort(reverse=True, key=lambda x: x[0])
            
            # 为当前缺失行选择最佳匹配
            if similarities:  # 添加空列表保护
                candidates[idx] = [similarities[0][1]]  # 只保留最相似的索引
        
        # 返回候选匹配字典
        return candidates
    
    def _calc_approximate_similarity(self, merged, candidates):
        total_sim = 0  # 初始化总相似度
        # 遍历每个缺失行及其对应的候选完整行索引
        for miss_idx, cand_indices in candidates.items():
            max_sim = 0  # 初始化当前缺失行的最大相似度
            miss_row = merged.loc[miss_idx]  # 获取缺失行的数据
            # 遍历每个候选行索引
            for cand_idx in cand_indices:
                cand_row = merged.loc[cand_idx]  # 获取候选行的数据
                # 计算缺失行与候选行的相似度
                sim = self._calc_merged_row_similarity(miss_row, cand_row)
                # 更新最大相似度
                if sim > max_sim:
                    max_sim = sim
            # 累加当前缺失行的最大相似度到总相似度
            total_sim += max_sim
        # 返回总相似度
        return total_sim

    def _apply_approximate_candidates(self, merged, candidates):
        # 创建表1和表2的副本，用于存储填补后的数据
        filled_t1 = self.t1.copy()
        filled_t2 = self.t2.copy()

        # 遍历每个缺失行及其对应的候选完整行索引
        for miss_idx, cand_indices in candidates.items():
            if not cand_indices:
                continue  # 跳过没有候选的缺失行

            best_cand_idx = cand_indices[0]  # 获取最相似的完整行索引
            best_cand = merged.loc[best_cand_idx]  # 获取最相似的完整行数据

            # 填补表1：确保列名带 _t1 后缀
            if merged.loc[miss_idx, '_missing_mask_t1']:
                t1_idx = merged.loc[miss_idx, 'index_t1']  # 获取表1中对应的索引
                for col in filled_t1.columns:
                    t1_col_in_merged = f"{col}_t1"  # 正确引用合并后的列名
                    # 检查表1中当前列是否为空且在完整行中有对应值
                    if pd.isnull(filled_t1.loc[t1_idx, col]) and t1_col_in_merged in best_cand:
                        dtype = self.table1_original_dtypes[col]  # 获取原始数据类型
                        value = best_cand[t1_col_in_merged]  # 获取填补值
                        # 根据数据类型进行转换
                        if pd.api.types.is_bool_dtype(dtype):
                            value = bool(value)
                        filled_t1.loc[t1_idx, col] = value  # 应用填补值

            # 填补表2：确保列名带 _t2 后缀
            if merged.loc[miss_idx, '_missing_mask_t2']:
                t2_idx = merged.loc[miss_idx, 'index_t2']  # 获取表2中对应的索引
                for col in filled_t2.columns:
                    t2_col_in_merged = f"{col}_t2"  # 正确引用合并后的列名
                    # 检查表2中当前列是否为空且在完整行中有对应值
                    if pd.isnull(filled_t2.loc[t2_idx, col]) and t2_col_in_merged in best_cand:
                        dtype = self.table2_original_dtypes[col]  # 获取原始数据类型
                        value = best_cand[t2_col_in_merged]  # 获取填补值
                        # 根据数据类型进行转换
                        if pd.api.types.is_bool_dtype(dtype):
                            value = bool(value)
                        filled_t2.loc[t2_idx, col] = value  # 应用填补值

        # 返回填补后的表1和表2
        return filled_t1, filled_t2
 
    def _precise_impute(self):
        """并行化的精确算法"""
        # 并行处理每个连接方案
        results = Parallel(n_jobs=-1)(
            delayed(self._process_single_link_info)(link_info)
            for link_info in self.link_combinations
        )

        # 选择全局最优结果
        best_score = -np.inf
        best_tables = (self.t1.copy(), self.t2.copy())
        for score, temp_t1, temp_t2 in results:
            if score > best_score:
                best_score = score
                best_tables = (temp_t1, temp_t2)

        # 恢复原始表的格式
        best_t1, best_t2 = best_tables
        best_t1 = best_t1.reindex_like(self.original_table1).fillna(method='ffill')
        best_t2 = best_t2.reindex_like(self.original_table2).fillna(method='ffill')
        return best_t1, best_t2

    def _process_single_link_info(self, link_info):
        """处理单个连接方案并返回最优结果（并行任务函数）"""
        merged = link_info['merged_view'].copy()
        candidate_map = self._build_candidate_mapping(link_info['link_map'], merged)
        candidate_combos = itertools.product(*candidate_map.values())

        best_local_score = -np.inf
        best_local_t1, best_local_t2 = None, None

        for combo in candidate_combos:
            temp_merged = merged.copy()
            total_sim = 0

            for entry in combo:
                key, value, src_idx = entry
                col, idx, table = key

                # 应用填补值
                if table == 't1':
                    temp_merged.loc[temp_merged['index_t1'] == idx, f'{col}_t1'] = value
                else:
                    temp_merged.loc[temp_merged['index_t2'] == idx, f'{col}_t2'] = value

                # 计算相似度
                miss_row_idx = temp_merged[temp_merged[f'index_{table}'] == idx].index[0]
                src_row = temp_merged.loc[src_idx]
                sim = self._calc_merged_row_similarity(temp_merged.loc[miss_row_idx], src_row)
                total_sim += sim

            # 更新当前连接方案下的最优结果
            if total_sim > best_local_score:
                best_local_score = total_sim
                best_local_t1 = self._extract_table_from_merged(temp_merged, 't1')
                best_local_t2 = self._extract_table_from_merged(temp_merged, 't2')

        return (best_local_score, best_local_t1, best_local_t2)



    def _build_candidate_mapping(self, link_map, merged):
        candidate_map = defaultdict(list)
        
        # 处理表1候选
        for (col, idx), cands in self.non_sind_candidates['t1'].items():
            if idx in link_map:
                for (value, src_idx) in cands:
                    src_in_merged = merged[merged['index_t1'] == src_idx].index
                    if src_in_merged.empty: continue
                    key = (col, idx, 't1')
                    candidate_map[key].append( (key, value, src_in_merged[0]) )  # 添加key到值中
        
        # 处理表2候选
        for (col, idx), cands in self.non_sind_candidates['t2'].items():
            if any(idx in t2_indices for t2_indices in link_map.values()):
                for (value, src_idx) in cands:
                    src_in_merged = merged[merged['index_t2'] == src_idx].index
                    if src_in_merged.empty: continue
                    key = (col, idx, 't2')
                    candidate_map[key].append( (key, value, src_in_merged[0]) )
        
        return candidate_map

    def _extract_table_from_merged(self, merged, table_name):
        """从合并视图提取表（修复索引列问题）"""
        suffix = '_t1' if table_name == 't1' else '_t2'
        
        # 获取所有有效列（排除索引列）
        cols = [
            col.replace(suffix, '') 
            for col in merged.columns 
            if col.endswith(suffix) and not col.startswith('index_')
        ]
        
        # 提取数据列（带后缀）
        extracted = merged[[col + suffix for col in cols]].copy()
        extracted.columns = cols  # 移除后缀
        
        # 恢复原始数据类型（仅处理原始存在的列）
        dtypes = self.table1_original_dtypes if table_name == 't1' else self.table2_original_dtypes
        for col in cols:
            if col in dtypes:  # 添加存在性检查
                extracted[col] = extracted[col].astype(dtypes[col])
        
        return extracted
    
    def _calc_merged_row_similarity(self, row1, row2):
        total = 0
        valid = 0
        
        # 定义需要排除的列：SIND列和关键列
        exclude_cols = {self.A, self.A_prime} | set(self.key_cols)
        exclude_cols = {'index_t1', 'index_t2', '_missing_mask_t1', '_missing_mask_t2'}
        
        for col in row1.index:
            if col in exclude_cols:
                continue
                
            v1 = row1[col]
            v2 = row2[col]
            
            if pd.isnull(v1) or pd.isnull(v2):
                continue
                
            # 计算单列相似度
            sim = self._calc_single_feature_similarity(col, v1, v2)
            total += sim
            valid += 1
        
        return total / valid if valid > 0 else 0
    
    # ------------------- 执行入口 -------------------
    def execute(self):

        start_time = time.time()
        
        print("开始填充SIND相关缺失值...")
        self._generate_sind_candidates()
        print("SIND缺失值填充完成.")

        if self.non_sind_method == 'full_enum':
            print("生成非SIND候选值...")
            self._generate_precise_non_sind_candidates()  # 新增步骤
            print("非SIND候选值生成完成.")

        print("开始连接表格...")
        self._link_tables()
        print("表格连接完成.")

        print("开始填充非SIND缺失值...")
        if self.non_sind_method == 'full_enum':
            self.t1, self.t2 = self._precise_impute()
        elif self.non_sind_method == 'approximate':
            self.t1, self.t2 = self._approximate_impute()  # 调用独立近似算法
        print("非SIND缺失值填充完成.")

        # 直接使用原始表并移除 '_missing_mask' 列
        result_t1 = self.t1.drop(columns=['_missing_mask'], errors='ignore')
        result_t2 = self.t2.drop(columns=['_missing_mask'], errors='ignore')
        
        print("填补完成.")

        end_time = time.time()
        print(f"填补完成，耗时：{end_time - start_time:.2f} 秒")
        return result_t1, result_t2



if __name__ == "__main__":
    table1 = pd.read_csv('table7_with_missing.csv', sep=',')
    table2 = pd.read_csv('table8_with_missing.csv', sep=',')

    imputer = StrictSINDImputer(
        table1, table2,
        sind_pairs=('ID', 'ID'),
        key_cols=['Name'],
        K=1,
        delta=1,
        non_sind_method='approximate', 
    )
    filled_t1, filled_t2 = imputer.execute()

    print("填补结果（非编码模式）：")
    print("表1：\n", filled_t1)
    print("\n表2：\n", filled_t2)

    filled_t1.to_csv('table7_imputed.csv', index=False)
    filled_t2.to_csv('table8_imputed.csv', index=False)