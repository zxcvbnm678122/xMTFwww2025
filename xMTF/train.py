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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AAAI-2023...")
    parser.add_argument("--data_path", type=str, default="KuaiRand-1K")
    parser.add_argument("--split_date", type=str, default="2022-05-05")
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=65536)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument('--bottom_dnn_hidden_units', nargs='*', default=[256, 128], type=int)
    parser.add_argument('--tower_dnn_hidden_units', nargs='*', default=[64], type=int)
    parser.add_argument("--seed", type=int, default=2023)

    args = parser.parse_args()

    data_path = args.data_path
    split_date = args.split_date
    validation_split = args.validation_split
    epochs = args.epochs
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    embedding_dim = args.embedding_dim
    num_experts = args.num_experts
    bottom_dnn_hidden_units = args.bottom_dnn_hidden_units
    tower_dnn_hidden_units = args.tower_dnn_hidden_units
    seed = args.seed

    seed_all(seed)

    t0 = time.time()
    # 读取原始数据
    data_type = data_path.split('-')[-1].lower()
    # nrows = 1e12

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

    del df1, df2, user_features, video_features_basic, video_features_statistics
    gc.collect()

    # # 观看时距离视频上传的天数
    # data["upload_days"] = (data["date"] - data["upload_dt"]).dt.days

    user_dnn_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=embedding_dim) for feat in user_sparse_features]
    item_dnn_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=embedding_dim) for feat in item_sparse_features]
    dnn_feature_columns = user_dnn_feature_columns + item_dnn_feature_columns

    # generate input data for model
    train = data[data['date'] <= split_date].reset_index(drop=True)
    test = data[data['date'] > split_date].reset_index(drop=True)
    train_model_input = {name: train[name].values for name in sparse_features}
    test_model_input = {name: test[name].values for name in sparse_features}

    print(f"训练集大小: {train.shape}, 起止日期: {train['date'].min()} ~ {train['date'].max()}")
    print(f"测试集大小: {test.shape}, 起止日期: {test['date'].min()} ~ {test['date'].max()}")

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

    model_prerank.compile(
        optimizer="adam",
        loss=[
            "binary_crossentropy", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy",
            # 'mean_squared_error',
            huber_loss,
        ],
        loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0],
    )

    ckpt = ModelCheckpoint(f'./models/prerank_{data_type}.h5', save_best_only=True, save_weights_only=True, verbose=1, monitor='val_loss', mode='min')

    history = model_prerank.fit(
        x=train_model_input,
        y=[train[label].values for label in labels],
        batch_size=train_batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=validation_split,
        callbacks=[
            ckpt,
        ],
    )

    pred_ans = model_prerank.predict(test_model_input, batch_size=test_batch_size)

    for i, label in enumerate(labels):
        if i <=5:
            print("prerank test %s AUC:" % label, round(roc_auc_score(test[label], pred_ans[i]), 4))
        elif i ==6:
            print("prerank test %s MSE:" % label, round(mean_squared_error(test[label], pred_ans[i]), 4))

    t3 = time.time()
    print(f"双塔模型训练完成, 耗时: {t3-t2:.1f}s")

    predict_df = pd.DataFrame(np.hstack(pred_ans), columns=[f"prerank_{label}" for label in labels])
    predict_df = pd.concat([test[['user_id', 'video_id'] + labels + ['play_time_s', 'duration_s']].reset_index(drop=True), predict_df], axis=1)
    # predict_df['prerank_play_time_s'] = (np.exp(predict_df['prerank_play_time_label'] * np.log(5)) - 1) * predict_df['duration_s']
    predict_df['prerank_play_time_s'] = np.exp(predict_df['prerank_play_time_label'] * 7) - 1
    predict_df.loc[predict_df['prerank_play_time_s'] < 0, 'prerank_play_time_s'] = 0
    predict_df.drop(['prerank_play_time_label'], axis=1, inplace=True)
    predict_df.to_pickle(f'./results/prerank_predict_{data_type}.pkl')

    # for i, label in enumerate(labels):
    #     if i <=5:
    #         # 搜寻最优阈值
    #         best_f1 = 0
    #         thresholds = np.arange(0.01, 1.0, 0.01)
    #         for threshold in thresholds:
    #             predict_df[f'prerank_{label}_pred'] = predict_df[f'prerank_{label}'].apply(lambda x: 1 if x >= threshold else 0)
    #             f1 = f1_score(predict_df[label], predict_df[f'prerank_{label}_pred'])
    #             print(f"prerank test {label} F1: {f1:.4f}, threshold: {threshold}")
    #             if f1 > best_f1:
    #                 best_f1 = f1
    #                 best_threshold = threshold
    #         print(f"prerank test {label} F1: {best_f1:.4f}, best threshold: {best_threshold}")


    # print(predict_df['prerank_play_time_s'].describe())
    # print(mean_absolute_error(predict_df['play_time_s'], predict_df['prerank_play_time_s']))
    # fig = plt.figure(figsize=(15, 5))
    # sns.distplot(predict_df['play_time_s'], kde=False, bins=100)
    # sns.distplot(predict_df['prerank_play_time_s'], kde=False, bins=100)
    # plt.legend(['ground truth', 'prediction'])
    # # plt.show()
    # plt.savefig(f'./figs/prerank_{data_type}.png')

    # Stage 2. MMOE 排序
    model_fullrank = MMOE(
        dnn_feature_columns=dnn_feature_columns,
        num_experts=num_experts,
        expert_dnn_hidden_units=bottom_dnn_hidden_units,
        tower_dnn_hidden_units=tower_dnn_hidden_units,
        task_types=task_types,
        task_names=labels,
        seed=seed,
    )

    model_fullrank.compile(
        optimizer="adam",
        loss=[
            "binary_crossentropy", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy",
            # 'mean_squared_error',
            huber_loss,
        ],
        loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0],
    )

    ckpt = ModelCheckpoint(f'./models/fullrank_{data_type}.h5', save_best_only=True, save_weights_only=True, verbose=1, monitor='val_loss', mode='min')

    history = model_fullrank.fit(
        x=train_model_input,
        y=[train[label].values for label in labels],
        batch_size=train_batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=validation_split,
        callbacks=[
            ckpt,
        ],
    )

    pred_ans = model_fullrank.predict(test_model_input, batch_size=test_batch_size)

    for i, label in enumerate(labels):
        if i <=5:
            print("fullrank test %s AUC:" % label, round(roc_auc_score(test[label], pred_ans[i]), 4))
        elif i ==6:
            print("fullrank test %s MSE:" % label, round(mean_squared_error(test[label], pred_ans[i]), 4))

    t4 = time.time()
    print(f"fullrank模型训练完成, 耗时: {t4-t3:.1f}s")

    predict_df = pd.DataFrame(np.hstack(pred_ans), columns=[f"fullrank_{label}" for label in labels])
    predict_df = pd.concat([test[['user_id', 'video_id'] + labels + ['play_time_s', 'duration_s']].reset_index(drop=True), predict_df], axis=1)
    # predict_df['fullrank_play_time_s'] = (np.exp(predict_df['fullrank_play_time_label'] * np.log(5)) -1) * predict_df['duration_s']
    predict_df['fullrank_play_time_s'] = np.exp(predict_df['fullrank_play_time_label'] * 7) - 1
    predict_df.loc[predict_df['fullrank_play_time_s'] < 0, 'fullrank_play_time_s'] = 0
    predict_df.drop(['fullrank_play_time_label'], axis=1, inplace=True)
    predict_df.to_pickle(f'./results/fullrank_predict_{data_type}.pkl')

    # for i, label in enumerate(labels):
    #     if i <=5:
    #         # 搜寻最优阈值
    #         best_f1 = 0
    #         thresholds = np.arange(0.01, 1.0, 0.01)
    #         for threshold in thresholds:
    #             predict_df[f'fullrank_{label}_pred'] = predict_df[f'fullrank_{label}'].apply(lambda x: 1 if x >= threshold else 0)
    #             f1 = f1_score(predict_df[label], predict_df[f'fullrank_{label}_pred'])
    #             print(f"fullrank test {label} F1: {f1:.4f}, threshold: {threshold}")
    #             if f1 > best_f1:
    #                 best_f1 = f1
    #                 best_threshold = threshold
    #         print(f"fullrank test {label} F1: {best_f1:.4f}, best threshold: {best_threshold}")

    # print(mean_absolute_error(predict_df['play_time_s'], predict_df['fullrank_play_time_s']))
    # print(predict_df['fullrank_play_time_s'].describe())
    # fig = plt.figure(figsize=(15, 5))
    # sns.distplot(predict_df['play_time_s'], kde=False, bins=100)
    # sns.distplot(predict_df['fullrank_play_time_s'], kde=False, bins=100)
    # plt.legend(['ground truth', 'prediction'])
    # # plt.show()
    # plt.savefig(f'./figs/fullrank_{data_type}.png')
