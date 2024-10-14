import sys
print("Python version: {}". format(sys.version))

import numpy as np
print("numpy version: {}". format(np.__version__))

import pandas as pd
print("pandas version: {}". format(pd.__version__))

import sklearn
print("scikit-learn version: {}". format(sklearn.__version__))
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import argparse
import time

import tensorflow as tf
print("tensorflow version: {}". format(tf.__version__))

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import deepctr
print("deepctr version: {}". format(deepctr.__version__))
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from mmoe import MMOE
from two_tower import TwoTower

from util import reduce_mem_usage

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

import warnings
warnings.filterwarnings('ignore')


# needed for deterministic output
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)


def huber_loss(y_true, y_pred, delta=0.001):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    small_error_loss = tf.square(error) / 2
    big_error_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, small_error_loss, big_error_loss)


# 缺失值填充, 并转换为整形特征
def fillna_df(df, features):
    for feat in tqdm([x for x in features if x not in ['user_id', 'video_id']]):
        df[feat] = df[feat].astype('str')
        df[feat] = df[feat].fillna('-1', )
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])


def cross_join(user_ids, video_ids):
    df1 = pd.DataFrame(user_ids, columns=['user_id'])
    df2 = pd.DataFrame(video_ids, columns=['video_id'])

    # 使用 merge 方法生成笛卡尔积
    df = pd.merge(df1, df2, how='cross')
    return df


def get_tower_emb(label='is_click'):
    # 根据层的名字获取输出, 用于生成双塔召回的embedding
    user_tower_output = model_prerank.get_layer(f'user_tower_{label}').output
    item_tower_output = model_prerank.get_layer(f'item_tower_{label}').output
    user_tower_model = Model(inputs=model_prerank.inputs, outputs=user_tower_output)
    item_tower_model = Model(inputs=model_prerank.inputs, outputs=item_tower_output)

    temp_df = cross_join(train['user_id'].unique(), [0])
    temp_df = pd.merge(temp_df, user_features, on='user_id', how='left')
    temp_df = pd.merge(temp_df, video_features_basic, on='video_id', how='left')
    temp_df = pd.merge(temp_df, video_features_statistics, on='video_id', how='left')
    temp_model_input = {name: temp_df[name].values for name in sparse_features}
    user_tower_output_data = user_tower_model.predict(temp_model_input, batch_size=test_batch_size)
    print(user_tower_output_data.shape)
    # 对每个用户生成粗排embedding
    predict_df = pd.DataFrame(user_tower_output_data, columns=[f"user_tower_{label}_{i}" for i in range(64)])
    predict_df = pd.concat([temp_df[['user_id']], predict_df], axis=1)
    predict_df.to_pickle(f'./results/user_tower_{label}_{data_type}.pkl')

    temp_df = cross_join([0], train['video_id'].unique())
    temp_df = pd.merge(temp_df, user_features, on='user_id', how='left')
    temp_df = pd.merge(temp_df, video_features_basic, on='video_id', how='left')
    temp_df = pd.merge(temp_df, video_features_statistics, on='video_id', how='left')
    temp_model_input = {name: temp_df[name].values for name in sparse_features}
    item_tower_output_data = item_tower_model.predict(temp_model_input, batch_size=test_batch_size)
    print(item_tower_output_data.shape)
    # 对每个item生成粗排embedding
    predict_df = pd.DataFrame(item_tower_output_data, columns=[f"item_tower_{label}_{i}" for i in range(64)])
    predict_df = pd.concat([temp_df[['video_id']], predict_df], axis=1)
    predict_df.to_pickle(f'./results/item_tower_{label}_{data_type}.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("WWW-2025...")
    parser.add_argument("--data_path", type=str, default="KuaiRand-1K")
    parser.add_argument("--split_date", type=str, default="2022-05-05")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=65536)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument('--bottom_dnn_hidden_units', nargs='*', default=[256, 128], type=int)
    parser.add_argument('--tower_dnn_hidden_units', nargs='*', default=[64], type=int)
    parser.add_argument("--prerank_topk_num", type=int, default=1000)
    parser.add_argument("--fullrank_topk_num", type=int, default=1000)
    parser.add_argument("--per_select_num", type=int, default=1, help="每次遍历 user_id 类别数, 太大会爆内存")
    parser.add_argument("--seed", type=int, default=2023)

    args = parser.parse_args()

    data_path = args.data_path
    split_date = args.split_date
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    embedding_dim = args.embedding_dim
    num_experts = args.num_experts
    bottom_dnn_hidden_units = args.bottom_dnn_hidden_units
    tower_dnn_hidden_units = args.tower_dnn_hidden_units
    prerank_topk_num = args.prerank_topk_num
    fullrank_topk_num = args.fullrank_topk_num
    per_select_num = args.per_select_num
    seed = args.seed

    seed_all(seed)

    t0 = time.time()
    # 读取原始数据
    data_type = data_path.split('-')[-1].lower()
    nrows = 1e12

    # # df_rand = pd.read_csv(os.path.join(data_path, f"data/log_random_4_22_to_5_08_{data_type}.csv"))
    # df1 = pd.read_csv(os.path.join(data_path, f"data/log_standard_4_08_to_4_21_{data_type}.csv"), nrows=nrows, parse_dates=['date'])
    # df2 = pd.read_csv(os.path.join(data_path, f"data/log_standard_4_22_to_5_08_{data_type}.csv"), nrows=nrows, parse_dates=['date'])
    # user_features = pd.read_csv(os.path.join(data_path, f"data/user_features_{data_type}.csv"))
    # video_features_basic = pd.read_csv(os.path.join(data_path, f"data/video_features_basic_{data_type}.csv"), parse_dates=['upload_dt'])
    # video_features_statistics = pd.read_csv(os.path.join(data_path, f"data/video_features_statistic_{data_type}.csv"))

    # 直接读取 pickle 文件 (速度快)
    df1 = pd.read_pickle(os.path.join(data_path, f"data/log_standard_4_08_to_4_21_{data_type}.pkl"))
    df2 = pd.read_pickle(os.path.join(data_path, f"data/log_standard_4_22_to_5_08_{data_type}.pkl"))
    user_features = pd.read_pickle(os.path.join(data_path, f"data/user_features_{data_type}.pkl"))
    video_features_basic = pd.read_pickle(os.path.join(data_path, f"data/video_features_basic_{data_type}.pkl"))
    video_features_statistics = pd.read_pickle(os.path.join(data_path, f"data/video_features_statistic_{data_type}.pkl"))

    data = pd.concat([df1, df2], axis=0, ignore_index=True)
    data = data[(data['tab'] == 1) & (data['play_time_ms']) > 0]
    data.drop(['tab'], axis=1, inplace=True)
    data = data[data['duration_ms'] > 0]
    data['duration_s'] = data['duration_ms'] / 1000
    data['play_time_s'] = data['play_time_ms'] / 1000
    data['play_rate'] = data['play_time_ms'] / data['duration_ms']
    data = data[data['play_rate'] <= 5]
    # data['play_time_label'] = np.log(data['play_rate'] + 1) / np.log(5)
    data['play_time_label'] = np.log(data['play_time_s'] + 1) / 7

    user_features.drop(['follow_user_num'], axis=1, inplace=True)

    video_features_basic = video_features_basic[video_features_basic['video_duration'].notnull()]

    del df1, df2
    gc.collect()

    t1 = time.time()
    print(f"原始数据读取完毕, 用时: {t1-t0:.1f}s")

    labels = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view', 'play_time_label']
    task_types = ['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'regression']

    # 数据预处理
    # dense 特征分箱转换为类别特征
    basic_dense_features = ["video_duration"]
    statistics_dense_features = [col for col in video_features_statistics.columns.tolist() if col != "video_id"]
    for col in basic_dense_features:
        # video_features_basic[col] = pd.cut(video_features_basic[col], bins=16, labels=False)
        video_features_basic[col] = pd.qcut(video_features_basic[col], q=16, labels=False, duplicates='drop')
    for col in statistics_dense_features:
        # video_features_statistics[col] = pd.cut(video_features_statistics[col], bins=16, labels=False)
        video_features_statistics[col] = pd.qcut(video_features_statistics[col], q=16, labels=False, duplicates='drop')

    user_sparse_features = ["user_id", "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author", "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range"] + [f"onehot_feat{i}" for i in range(1, 18)]
    basic_item_sparse_features = ["video_id", "author_id", "video_type", "upload_type", "visible_status", "music_id", "music_type"] + ["video_duration", "server_width", "server_height"]
    statistics_item_sparse_features = [col for col in video_features_statistics.columns.tolist() if col != "video_id"]
    item_sparse_features = basic_item_sparse_features + statistics_item_sparse_features
    sparse_features = user_sparse_features + item_sparse_features

    fillna_df(user_features, user_sparse_features)
    fillna_df(video_features_basic, basic_item_sparse_features)
    fillna_df(video_features_statistics, statistics_item_sparse_features)

    print(f"共使用 {len(user_sparse_features)} 个 user 稀疏特征: {user_sparse_features}")
    print(f"共使用 {len(item_sparse_features)} 个 item 稀疏特征: {item_sparse_features}")

    # 合并数据
    data = pd.merge(data, user_features, on='user_id', how='left')
    data = pd.merge(data, video_features_basic, on='video_id', how='left')
    data = pd.merge(data, video_features_statistics, on='video_id', how='left')

    # # 观看时距离视频上传的天数
    # data["upload_days"] = (data["date"] - data["upload_dt"]).dt.days

    user_dnn_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=embedding_dim) for feat in user_sparse_features]
    item_dnn_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=embedding_dim) for feat in item_sparse_features]
    dnn_feature_columns = user_dnn_feature_columns + item_dnn_feature_columns

    # generate input data for model
    train = data[data['date'] <= split_date].reset_index(drop=True)
    # test = data[data['date'] > split_date].reset_index(drop=True)
    # train_model_input = {name: train[name].values for name in sparse_features}
    # test_model_input = {name: test[name].values for name in sparse_features}

    print(f"训练集大小: {train.shape}, 起止日期: {train['date'].min()} ~ {train['date'].max()}")
    # print(f"测试集大小: {test.shape}, 起止日期: {test['date'].min()} ~ {test['date'].max()}")

    t2 = time.time()
    print(f"数据预处理完成, 耗时: {t2-t1:.1f}s")

    # Stage 1. 双塔召回
    model_prerank = TwoTower(
        user_dnn_feature_columns=user_dnn_feature_columns,
        item_dnn_feature_columns=item_dnn_feature_columns,
        bottom_dnn_hidden_units=bottom_dnn_hidden_units,
        tower_dnn_hidden_units=tower_dnn_hidden_units,
        task_types=task_types,
        task_names=labels,
        seed=seed,
    )

    model_prerank.load_weights(f"./models/prerank_{data_type}.h5")

    # get_tower_emb()


    # Stage 2. MMOE排序
    model_fullrank = MMOE(
        dnn_feature_columns=dnn_feature_columns,
        num_experts=num_experts,
        expert_dnn_hidden_units=bottom_dnn_hidden_units,
        tower_dnn_hidden_units=tower_dnn_hidden_units,
        task_types=task_types,
        task_names=labels,
        seed=seed,
    )

    model_fullrank.load_weights(f"./models/fullrank_{data_type}.h5")

    unique_user_ids = train['user_id'].unique()  # 找出所有不重复的user_id
    unique_video_ids = train['video_id'].unique()  # 找出所有不重复的video_id

    bucket_num = int(len(unique_user_ids) / 10)

    
    for k in range(0, 10):
        prerank_topk_outputs = []
        fullrank_topk_outputs = []

        unique_user_ids_bucket = unique_user_ids[(k * bucket_num): ( (k+1) * bucket_num)]

        for i in tqdm(range(0, len(unique_user_ids_bucket), per_select_num)):  # 循环遍历 unique_user_ids
        # for i in tqdm(range(0, 500, per_select_num)):  # 循环遍历 unique_user_ids
            current_user_ids = unique_user_ids_bucket[i:i+per_select_num]

            temp_df = cross_join(current_user_ids, unique_video_ids)
            # print(f"召回候选集大小: {temp_df.shape[0]}")

            temp_df = pd.merge(temp_df, user_features, on='user_id', how='left')
            temp_df = pd.merge(temp_df, video_features_basic, on='video_id', how='left')
            temp_df = pd.merge(temp_df, video_features_statistics, on='video_id', how='left')
            temp_model_input = {name: temp_df[name].values for name in sparse_features}

            # 对每个用户生成粗排结果
            pred_ans = model_prerank.predict(temp_model_input, batch_size=test_batch_size)
            predict_df = pd.DataFrame(np.hstack(pred_ans), columns=labels)
            predict_df = pd.concat([temp_df[['user_id', 'video_id']], predict_df], axis=1)
            predict_df['play_time_s'] = np.exp(predict_df['play_time_label'] * 7) - 1
            predict_df.drop(['play_time_label'], axis=1, inplace=True)
            prerank_topk_outputs.append(predict_df.round(6))

            # 对每个用户生成精排结果
            pred_ans = model_fullrank.predict(temp_model_input, batch_size=test_batch_size)
            predict_df = pd.DataFrame(np.hstack(pred_ans), columns=labels)
            predict_df = pd.concat([temp_df[['user_id', 'video_id']], predict_df], axis=1)
            predict_df['play_time_s'] = np.exp(predict_df['play_time_label'] * 7) - 1
            predict_df.drop(['play_time_label'], axis=1, inplace=True)
            fullrank_topk_outputs.append(predict_df.round(6))

            # print(f"排序输出集大小: {fullrank_topk_video_df.shape[0]}")

        
        prerank_topk_df = pd.concat(prerank_topk_outputs, axis=0, ignore_index=True)
        prerank_topk_df.to_pickle(f'./results/prerank_{data_type}_{k}.pkl')

        del prerank_topk_outputs, prerank_topk_df
        gc.collect()

        fullrank_topk_df = pd.concat(fullrank_topk_outputs, axis=0, ignore_index=True)
        fullrank_topk_df.to_pickle(f'./results/fullrank_{data_type}_{k}.pkl')

        del fullrank_topk_outputs, fullrank_topk_df
        gc.collect()

        del temp_df, temp_model_input, pred_ans, predict_df
        gc.collect()


