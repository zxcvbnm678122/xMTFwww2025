import sys
print("Python version: {}". format(sys.version))
import glob
import numpy as np
print("numpy version: {}". format(np.__version__))

import pandas as pd
print("pandas version: {}". format(pd.__version__))

import sklearn
print("scikit-learn version: {}". format(sklearn.__version__))
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import GlorotNormal
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import argparse
import time
from datetime import datetime

#import tensorflow.compat.v1 as tf
import tensorflow as tf
print("tensorflow version: {}". format(tf.__version__))
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow.compat.v1 as tf
import deepctr
print("deepctr version: {}". format(deepctr.__version__))
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from mmoe import MMOE
from two_tower import TwoTower

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from mfc import MfcBaseModel
from mfc import DataGenerator
import warnings
warnings.filterwarnings('ignore')


def handle_input(dense_input):
    epsilon=0.00001
    dense_feature_num = 6
    input_feas_reshape = tf.reshape(dense_input, [1 * list_num, dense_feature_num])  # [-1, ]
    list_feas_trans = tf.clip_by_value(input_feas_reshape, clip_value_min=epsilon, clip_value_max=1.0-epsilon)
    log_x = list_feas_trans + 1.0
    tanh_x = list_feas_trans * 4.0
    reverse_x = - tf.math.log(1 / (list_feas_trans + epsilon) - 1.0)

    ## nonlinear
    log_v1 = tf.math.log(log_x)  #(batch_size, label_num)
    tanh_v1 = tf.tanh(tanh_x)
    sigmoid_v1 = tf.sigmoid(reverse_x + 1.0)
    sigmoid_v2 = tf.sigmoid(reverse_x + 2.0)
    sigmoid_v3 = tf.sigmoid(reverse_x + 3.0)
    sigmoid_v4 = tf.sigmoid(reverse_x + 4.0)

    all_pxtr_lst = [list_feas_trans, log_v1, 
                    tanh_v1, 
                    sigmoid_v1, sigmoid_v2, sigmoid_v3, sigmoid_v4]
    change_size = len(all_pxtr_lst)
    all_pxtr_input = tf.concat(all_pxtr_lst, axis=1)
    all_pxtr_input = tf.clip_by_value(all_pxtr_input, clip_value_min=epsilon, clip_value_max=1.0-epsilon)
    model_input = tf.reshape(all_pxtr_input, [1, list_num, dense_feature_num * change_size])
    return model_input

def train():
    num_epochs = 10
    batch_size = 64
    df = pd.read_pickle(f'./predict/{data_type}/fullrank_set_train.pkl') # 读取pkl的数据，速度快
    print(df)
    # 假设你的输入特征维度是 num_features
    # num_features = 6  # 除去 'uid' 和 'label'
    
    # 构建模型
    MfcModel = MfcBaseModel()
    #MfcModel.build(input_shape=(64,500,49))
    # 定义优化器和损失函数
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=0)
    learning_rate=0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # 定义评估指标
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    # 定义保存模型的 checkpoint
    ckpt = tf.train.Checkpoint(model=MfcModel, optimizer=optimizer)
    #ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    checkpoint_dir = "./checkpoints"
    checkpoint_prefix = f"{checkpoint_dir}/ckpt"
    
    # 初始化数据生成器
    train_data_gen = DataGenerator(df, batch_size)

    dir_name="./mfc_checkpoints/mfc_checkpoint"
    step=0
    steps_per_save = 100  # 每100个 step 保存一次
    for epoch in range(num_epochs):
        train_loss.reset_states()

        for batch in range(len(train_data_gen)):
            features, labels = train_data_gen[batch]
            print(features.shape)
            #[BS,500,6]
            #[BS,500,1]
            MfcModel.train_step(features, labels) 
            print(step)
            step+=1
            if step % steps_per_save == 0:
                # for var in MfcModel.variables:
                #     print(f"Variable: {var.name}, Shape: {var.shape}, Value: {var.numpy()}")
                # checkpoint_name = f'{dir_name}/checkpoint_step_{step}.ckpt'
                # MfcModel.save_weights(checkpoint_name)
                # MfcModel.save('path_to_save_weights', save_format="tf")
                print(f'Step {step}: Model checkpoint saved.')
        
        print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}')

    # 最后保存模型
    MfcModel.save_weights(f'./models/{data_type}/mfc_model')


def test():
    # 设置超参数
    df = pd.read_pickle(f'./predict/{data_type}/fullrank_set_original.pkl') # 读取pkl的数据，速度快
    grouped_data = df.groupby('user_id')
    # 假设你的输入特征维度是 num_features

    # 构建模型
    MfcModel = MfcBaseModel()
    MfcModel.build(input_shape=(None, list_num, 42))
    # 生成模拟的测试数据
    # num_users = 10
    # num_rows_per_user = 500
    # num_predictions = 7
    # test_data = np.random.rand(num_users * num_rows_per_user, num_features)
    # uids = np.repeat(np.arange(num_users), num_rows_per_user)
    # vids = np.random.randint(0,10000,size=(5000))

    # save_dir_name="./mfc_checkpoints/mfc_checkpoint"
    # checkpoint_files = glob.glob(os.path.join(save_dir_name, 'checkpoint_step_*.ckpt.index'))
    # checkpoint_files.sort(key=os.path.getmtime)
    # if checkpoint_files:
    #     latest_checkpoint = checkpoint_files[-1].replace('.index', '')
    #     print(f'Loading latest checkpoint: {latest_checkpoint}')
    #     MfcModel.load_weights(f'../models/{data_type}/mfc_model')
    # else:
    #     print('No checkpoints found.')
    
    MfcModel.load_weights(f'./models/{data_type}/mfc_model')
    for var in MfcModel.variables:
        print(f"mfcVariable: {var.name}, Shape: {var.shape}, Value: {var.numpy()}")
    # MfcModel.load_weights(save_dir_name)
    # for var in MfcModel.variables:
    #     print(f"mfcVariable: {var.name}, Shape: {var.shape}, Value: {var.numpy()}")

    predictions = []
    i=0
    for uid, data in grouped_data:
        dense_input = data[['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view']].values
        uid=np.expand_dims(data['user_id'],axis=-1)
        vid=np.expand_dims(data['video_id'],axis=-1)
        # 将数据重塑为[1, 500, 6]的形状
        dense_input = dense_input.reshape(1,list_num, num_features)
        model_input = handle_input(dense_input)
        #[bs,500,7*6]
        #print(test_data)
        # 加载模型
        # 进行预测
        #print("model_input.,shape",model_input.shape)
        user_predictions = MfcModel(model_input)
        # list
        user_predictions=tf.concat(user_predictions,axis=-1)
        user_predictions = np.squeeze(user_predictions)
        #print(user_predictions.shape)
        # test_data = dense_input
        # test_data = np.squeeze(test_data)
        #print(test_data.shape)
        # 将每个用户的预测结果添加到总体预测结果中
        predictions=np.concatenate((uid,vid,user_predictions),axis=1)
        columns = ['user_id', 'video_id'] + ['is_click_mfc', 'is_like_mfc', 'is_follow_mfc', 'is_comment_mfc', 'is_forward_mfc', 'long_view_mfc'] + ['final_output']
        df = pd.DataFrame(predictions,columns=columns)
        df['user_id'] = df['user_id'].astype('int64')
        df['video_id'] = df['video_id'].astype('int64')
        
        # 将当前用户的预测结果写入结果文件
        df.to_csv('results.csv', mode='a', header=(i==0), index=False)
        i+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser("WWW-2025...")
    parser.add_argument("--data_path", type=str, default="KuaiRand-1K")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=65536)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()

    data_path = args.data_path
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    embedding_dim = args.embedding_dim

    data_type = data_path.split('-')[-1].lower()

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    t0 = time.time()

    list_num=1100
    num_features=6

    if args.train:  # train
        train()


    if args.test:  # test
        test()