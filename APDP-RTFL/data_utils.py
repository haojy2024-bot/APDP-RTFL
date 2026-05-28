import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from collections import defaultdict

DATA_FILE_APPLICATION = 'application_record.csv'
DATA_FILE_CREDIT = 'credit_record.csv'

def load_and_preprocess_data():
    try:
        app_df = pd.read_csv(DATA_FILE_APPLICATION)
        cred_df = pd.read_csv(DATA_FILE_CREDIT)
    except FileNotFoundError:
        print(f"Error: Ensure '{DATA_FILE_APPLICATION}' and '{DATA_FILE_CREDIT}' are in the current directory.")
        print("Download from:")
        print("https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction")
        return None, None, None, None

    df = pd.merge(app_df, cred_df, on='ID', how='inner')

    def determine_credit_risk(group):
        if any(s in ['2', '3', '4', '5'] for s in group['STATUS'].astype(str)):
            return 1 # Bad risk
        return 0 # Good risk

    target_df = df.groupby('ID').apply(determine_credit_risk).reset_index(name='TARGET')
    unique_app_df = df.drop_duplicates(subset=['ID'], keep='first')
    df = pd.merge(unique_app_df, target_df, on='ID', how='left')
    
    df = df.drop(['ID', 'MONTHS_BALANCE', 'STATUS', 'FLAG_MOBIL'], axis=1)
    df = df.dropna(subset=['TARGET'])
    
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
        
    df['DAYS_BIRTH'] = np.abs(df['DAYS_BIRTH']) / 365
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: 0 if x > 0 else np.abs(x) / 365)

    X = df.drop('TARGET', axis=1)
    y = df['TARGET'].astype(int)
    # 诊断信息打印
    print("\n=== Dataset Class Distribution ===")
    print(f"Total samples: {len(y)}")
    print(f"Positive (bad risk, TARGET=1): {y.sum()} ({y.mean():.3%})")
    print(f"Negative (good risk, TARGET=0): {len(y) - y.sum()} ({1 - y.mean():.3%})")
    # 预处理
    categorical_features = X.select_dtypes(include='object').columns
    numerical_features = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_processed = preprocessor.fit_transform(X)
    print(f"After preprocessing: {X_processed.shape[1]} features")

    try:
        feature_names_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    except AttributeError:
         feature_names_cat = list(preprocessor.named_transformers_['cat'].get_feature_names_out())

    feature_names = numerical_features.tolist() + feature_names_cat.tolist()
    # 拆分
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    # 转为DataFrame/Series（保持原有逻辑）
    X_train_full = pd.DataFrame(X_train_full).reset_index(drop=True) if not isinstance(X_train_full, pd.DataFrame) else X_train_full.reset_index(drop=True)
    y_train_full = pd.Series(y_train_full).reset_index(drop=True) if not isinstance(y_train_full, pd.Series) else y_train_full.reset_index(drop=True)
    X_test = pd.DataFrame(X_test).reset_index(drop=True) if not isinstance(X_test, pd.DataFrame) else X_test.reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True) if not isinstance(y_test, pd.Series) else y_test.reset_index(drop=True)
    return X_train_full, y_train_full, X_test, y_test, feature_names

def split_data_for_clients(X_train_full, y_train_full, num_clients,
                           size_ratios=None, # 每个客户端相对大小的比例
                           min_samples=500, # 防止某个客户端样本太少
                           stratified=True): # 新增参数，默认启用分层
    if size_ratios is None: # 默认：严重不均衡，例如常见设置
        size_ratios = [0.35, 0.25, 0.18, 0.12, 0.08] # 和=1.0
        # 或随机生成（更真实）
        # size_ratios = np.random.dirichlet(np.ones(num_clients) * 0.3)
        # size_ratios = size_ratios / size_ratios.sum()
    # 关键修复：转为numpy数组
    X_np = np.asarray(X_train_full)
    y_np = np.asarray(y_train_full)
    n_total = len(y_np)
    # 计算每个客户端目标大小
    split_sizes = [int(r * n_total) for r in size_ratios]
    # 修正最后一个，让总和精确等于n_total
    split_sizes = [max(min_samples, s) for s in split_sizes]
    # 保证最小样本量
    split_sizes[-1] = n_total - sum(split_sizes[:-1])
    #如果总和超了，就从最大的减
    excess = sum(split_sizes) - n_total
    if excess > 0:
        idx = np.argmax(split_sizes)
        split_sizes[idx] -= excess

    client_data = []
    if stratified:
        # 按类别分层
        pos_idx = np.where(y_np == 1)[0]
        neg_idx = np.where(y_np == 0)[0]
        # 为每个客户端分配大致比例的正负样本
        for size in split_sizes:
            # 计算该客户端应得正样本数（按全局比例）
            pos_target = int(size * (len(pos_idx) / n_total))
            pos_target = max(10, min(pos_target, len(pos_idx) // num_clients)) # 至少5个正样本
            if len(pos_idx) > 0 and pos_target > 0:
                # 采样正样本
                pos_sample = np.random.choice(pos_idx, pos_target, replace=False)
                pos_idx = np.setdiff1d(pos_idx, pos_sample)
            else:
                pos_sample = np.array([], dtype=int)

            # 剩余用负样本补足
            neg_target = size - len(pos_sample)
            neg_target = min(neg_target, len(neg_idx)) # 关键修复：不能超过剩余数量
            if len(neg_idx) > 0 and neg_target > 0:
                neg_sample = np.random.choice(neg_idx, neg_target, replace=False)
                neg_idx = np.setdiff1d(neg_idx, neg_sample)
            else:
                neg_sample = np.array([], dtype=int)
            client_idx = np.concatenate([pos_sample, neg_sample])
            np.random.shuffle(client_idx)
            client_data.append((X_np[client_idx], y_np[client_idx]))
            # 调试打印（上线时可注释掉）
            client_id = len(client_data) - 1
            pos_count =  np.sum(y_np[client_idx] == 1)
            neg_count = np.sum(y_np[client_idx] == 0)
            pos_ratio = pos_count / len(client_idx) if len(client_idx) > 0 else 0
            print(f"Client {client_id}: size={len(client_idx)},"
                  f"pos={pos_count} ({pos_ratio:.3%}),"
                  f"neg+{neg_count},"
                  f"remaining_pos={len(pos_idx)}, remaining_neg={len(neg_idx)}")

            #处理剩余未分配的样本
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                remaining_idx = np.setdiff1d(pos_idx, neg_idx)
                if len(remaining_idx) > 0:
                    np.random.shuffle(remaining_idx)
                    # 加到最后一个客户端（合并）
                    last_X, last_y = client_data[-1]
                    client_data[-1] = (
                        np.concatenate([last_X, X_np[remaining_idx]]),
                        np.concatenate([last_y, y_np[remaining_idx]])
                    )
    else:
        # 原随机逻辑
        shuffled_indices = np.random.permutation(n_total)
        start = 0
        for size in split_sizes:
            end = start + size
            client_idx = shuffled_indices[start:end]
            client_data.append((X_np[client_idx], y_np[client_idx]))
            start = end
    return client_data