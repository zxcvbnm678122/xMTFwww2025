###www2025
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

import warnings
warnings.filterwarnings('ignore')
class MfcBaseModel(Model):
    def __init__(self):
        super(MfcBaseModel, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.bn=64
        self.dim_hidden=128
        self.dim_in=42
        self.seq_len=100
        self.task_num=6
        self.I = tf.Variable(tf.random.truncated_normal([1, 50, self.dim_hidden],mean=0,stddev=1),name= 'inducing_points')  # [1, num_inds, hidden_dim]
        self.S = tf.Variable(tf.random.truncated_normal([1, 50, self.dim_hidden],mean=0,stddev=1),name= 'seed1_vectors')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.Attention()
        self.denseQ1 = tf.keras.layers.Dense(self.dim_hidden,use_bias=False,  kernel_initializer='he_normal',name='denseq')
        self.denseK1 = tf.keras.layers.Dense(self.dim_hidden,use_bias=False, kernel_initializer='he_normal', name='densek')
        self.denseV1 = tf.keras.layers.Dense(self.dim_hidden,use_bias=False,  kernel_initializer='he_normal',name='densev')
        self.denseQ2 = tf.keras.layers.Dense(self.dim_hidden,use_bias=False, kernel_initializer='RandomNormal', name='denseq2')
        self.denseK2 = tf.keras.layers.Dense(self.dim_hidden,use_bias=False, kernel_initializer='RandomNormal', name='densek2')
        self.denseV2 = tf.keras.layers.Dense(self.dim_hidden,use_bias=False, kernel_initializer='RandomNormal', name='densev2')
        self.denseagg = tf.keras.layers.Dense(self.dim_hidden, use_bias=False, kernel_initializer='RandomNormal', name='denseagg')
        self.denses=[]
        for i in range(self.task_num):
            layers = [
                # tf.keras.layers.Dense(256, activation='relu', name=f'dense1_task_{i}'),
                tf.keras.layers.Dense(32, activation='relu', use_bias=True,name=f'dense2_task_{i}'),
                tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True,name=f'dense3_task_{i}')
            ]
            self.denses.append(layers)
            
        self.dense1 = tf.keras.layers.Dense(256, use_bias=True, activation=tf.nn.relu, kernel_initializer='RandomNormal', name='dense1')
        self.dense2 = tf.keras.layers.Dense(64, use_bias=True, activation=tf.nn.relu, kernel_initializer='RandomNormal', name='dense2')
        self.dense3 = tf.keras.layers.Dense(1, use_bias=True, activation=tf.nn.relu, kernel_initializer='RandomNormal', name='dense3')
        self.densefinal = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,kernel_initializer='he_normal', name='densefinal')

    def handle_input(self,dense_input,batch_size):
        batch_size=batch_size
        list_num= 100
        epsilon=0.00001
        dense_feature_num = 6
        input_feas_reshape = tf.reshape(dense_input, [batch_size * list_num, dense_feature_num])  # [-1, ]
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
        model_input = tf.reshape(all_pxtr_input, [batch_size, 100, dense_feature_num * change_size])
        return model_input
    
    def call(self,model_input,dim_in=6,hidden_dim=128,dim_out=1,training=True):
        
        dim_in=model_input.shape[-1]
        #tf.print(model_input[0],summarize=1000)
        xtr_outputs_list = self.SetTransformer(model_input, dim_in, hidden_dim, dim_out, num_inds=50, 
                                num_seeds=100, enc_layer=0, dec_layer=0, 
                                ln=False, skip_connection=False, rff=True, task_num=6, name="s2s_isab_xtr")
        
        dim_in=model_input.shape[-1]
        [vtr_output, ltr_output, wdsr_output, ftr_output, fpr_output, evr_p60_output,final_output] = xtr_outputs_list
        return vtr_output,ltr_output, wdsr_output, ftr_output, fpr_output, evr_p60_output,final_output

    def train_step(self,model_input,labels):
        mfc_loss_weight=0.4
        splits = tf.split(model_input, 6, axis=2)
        #原始的各pxtr的输入，可以直接当label [BS,500,1]
        batch_size=model_input.shape[0]
        pxtr_list_0, pxtr_list_1, pxtr_list_2, pxtr_list_3,pxtr_list_4,pxtr_list_5 = [split for split in splits]
        model_input=self.handle_input(model_input,batch_size)
        with tf.GradientTape() as tape:
            vtr_output,ltr_output, wdsr_output, ftr_output, fpr_output, evr_p60_output,final_output = self(model_input)
            labels = tf.cast(labels,tf.float32)
            loss_final = self.loss_object(labels,final_output)
            #tf.print("loss_final",loss_final)
            loss_0 = self.loss_object(labels,vtr_output)
            tf.print("long view",evr_p60_output[0])
            tf.print("final_output",final_output[0])
            loss_1 = self.loss_object(labels,ltr_output)
            loss_2 = self.loss_object(labels,wdsr_output)
            loss_3 = self.loss_object(labels,ftr_output)
            loss_4 = self.loss_object(labels,fpr_output)
            loss_5 = self.loss_object(labels,evr_p60_output)
            loss_pw_0 = pairwise_loss(vtr_output,pxtr_list_0)
            loss_pw_1 = pairwise_loss(ltr_output,pxtr_list_1)
            loss_pw_2 = pairwise_loss(wdsr_output,pxtr_list_2)
            loss_pw_3 = pairwise_loss(ftr_output,pxtr_list_3)
            loss_pw_4 = pairwise_loss(fpr_output,pxtr_list_4)
            loss_pw_5 = pairwise_loss(evr_p60_output,pxtr_list_5)
            tf.print("loss_0",loss_0)
            tf.print("loss_1",loss_1)
            tf.print("loss_2",loss_2)
            tf.print("loss_3",loss_3)
            tf.print("loss_4",loss_4)
            tf.print("loss_5",loss_5)
            tf.print("loss_pw_0",loss_pw_0)
            tf.print("loss_pw_1",loss_pw_1)
            tf.print("loss_pw_2",loss_pw_2)
            tf.print("loss_pw_3",loss_pw_3)
            tf.print("loss_pw_4",loss_pw_4)
            tf.print("loss_pw_5",loss_pw_5)
            loss =(1-mfc_loss_weight)*(loss_final+ loss_0+ loss_1+ loss_2+ loss_3+ loss_4+ loss_5)+mfc_loss_weight*(loss_pw_0+loss_pw_1+loss_pw_2+loss_pw_3+loss_pw_4+loss_pw_5)
            tf.print("loss_all",loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        # for var in MfcModel.trainable_variables:
        #     print(f"Variable: {var.name}, Shape: {var.shape}")
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.train_loss(loss)
        
    def split_heads(self,x, batch_size, num_heads, d_model):
        """Split the last dimension into (num_heads, d_model).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, d_model)
        """
        x = tf.reshape(x, (batch_size, -1, num_heads, d_model))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self,Q, K, V):
        matmul_qk = tf.matmul(Q, K, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, V)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def MSA_self(self,X, Y, dim_V=128,  ln=False, skip_connection=False, name="msa-SELF"):
        num_heads=4
        # NOT a standard MSA, with skip connection from Q_.
        #print(X.shape)
        Q = self.denseQ1(X)  # (batch_size, <key/value dimensions>, key_dim)
        #tf.print("Q",Q)
        K = self.denseK1(Y)  # (batch_size, <key/value dimensions>, key_dim)
        #tf.print("K",K)
        V = self.denseV1(Y)  # (batch_size, <key/value dimensions>, value_dim)
        #tf.print("V",V)
        # d_model = dim_V // num_heads
        batch_size = tf.shape(X)[0]
        seq_len = X.shape.as_list()[1]
        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V)
        output = scaled_attention
        output = self.layer_norm(output)
        return output

    def MSA(self,X, Y, dim_Q, dim_K, dim_V,  ln=False, skip_connection=False, name="msa"):
        num_heads=1
        # NOT a standard MSA, with skip connection from Q_.
        #print(X.shape)
        if name.startswith("s2s_isab_xtr_enc0_mab0"):
            #print("Q1",name)
            #tf.print("xshape",X.shape)
            Q = self.denseQ1(X)  # (batch_size, <key/value dimensions>, key_dim)
            K = self.denseK1(Y)  # (batch_size, <key/value dimensions>, key_dim)
            V = self.denseV1(Y)  # (batch_size, <key/value dimensions>, value_dim)
        else:
            #print("Q2",name)
            #tf.print("xshape",X.shape)
            Q = self.denseQ2(X)  # (batch_size, <key/value dimensions>, key_dim)
            K = self.denseK2(Y)  # (batch_size, <key/value dimensions>, key_dim)
            V = self.denseV2(Y)  # (batch_size, <key/value dimensions>, value_dim)
        d_model = dim_V // num_heads
        batch_size = tf.shape(X)[0]
        seq_len = X.shape.as_list()[1]
        Q_ = self.split_heads(Q, batch_size, num_heads, d_model)
        K_ = self.split_heads(K, batch_size, num_heads, d_model)
        V_ = self.split_heads(V, batch_size, num_heads, d_model)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q_, K_, V_)
        # scaled_attention = scaled_attention + Q_  # skip connection
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, d_model)
        output = tf.reshape(scaled_attention, (batch_size, seq_len, dim_V))  # (batch_size, seq_len_q, d_model)
        # output = dense(concat_attention, dim_V, dim_V, name=name+'_agg')  # (batch_size, seq_len_q, d_model)
        if ln:
            output = self.layer_norm(output)
        output_agg = self.denseagg(output)
        output = output + output_agg if skip_connection else output_agg
        return output
        
    def MAB(self,X, Y, dim_Q, dim_K, dim_V,  ln=False, skip_connection=False, name="mab"):
        h = self.MSA(X, Y, dim_Q, dim_K, dim_V, ln, skip_connection, name=name+'_msa')
        if ln:
            h = self.layer_norm(h)
        return h

    def SAB(self,X, dim_V, ln=False, skip_connection=False, name="sab"):
        return self.MAB(X, X, dim_V, dim_V, dim_V, ln, skip_connection, name)

    def ISAB(self,X, dim_in, dim_out,  num_inds=50, ln=False, 
            skip_connection=False, name="isab"):
        if num_inds < 1:
            return self.MAB(X, X, dim_in, dim_in, dim_out, ln, skip_connection, name)
        H = self.MAB(tf.tile(self.I, [tf.shape(X)[0], 1, 1]), X, dim_out, dim_in, dim_out, ln, 
                skip_connection, name+"_mab0")  # [b, num_inds, hidden_num]
        return self.MAB(X, H, dim_in, dim_out, dim_out,ln, skip_connection, name+"_mab1")  # [b, seq_len, hidden_num]

    def PMA(self,X, hidden_num, num_seeds, ln=False, skip_connection=False, name = "pma"):
        initializers = GlorotNormal()
        # X = rFF(X, hidden_num, hidden_num, name=name+"_rff")
        batch_size = tf.shape(X)[0]
        return self.MAB(tf.tile(self.S, [batch_size, 1, 1]), X, hidden_num, hidden_num, hidden_num, ln, skip_connection, name+"_mab")

    def SetTransformer(self, X, dim_in, dim_hidden, dim_out, num_inds=-1, num_seeds=2,
                    enc_layer=0, dec_layer=1, ln=True, skip_connection=True, 
                    rff=False, task_num=1, name="s2s"):
        X = self.MSA_self(X,X)
        outputs_list = []
        for layers in self.denses:
            x = X
            for layer in layers:
                x = layer(x)
            outputs_list.append(x)

        xtr_outputs=tf.concat(outputs_list,axis=-1)
        final_output=self.densefinal(xtr_outputs)
        outputs_list.append(final_output)
        #{}每一个元素都是[bs,100,1]*7
        return outputs_list

    def scaled_sigmoid(self,x):
        return 3*(1/(1+tf.math.exp(-0.025*(x-400))))

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size):
        self.df = df
        self.uids = df['new_uid'].unique()
        self.batch_size = batch_size
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.uids) // self.batch_size

    def __getitem__(self, index):
        # 获取一个批次的用户 
        uid_batch = self.uids[index * self.batch_size:(index + 1) * self.batch_size]
        batch_features = []
        batch_labels = []

        for uid in uid_batch:
            # 获取这个UID所有相关的样本的索引
            user_indices = self.df[self.df['new_uid'] == uid].index
            # 随机选出100个样本（有可能小于100）
            sampled_indices = np.random.choice(user_indices, size=min(100, len(user_indices)), replace=True)
            # 提取这些样本的特征和标签
            batch_features.append(self.df.loc[sampled_indices, self.df.columns != 'label'].drop(['new_uid','video_id','score','play_time_s'], axis=1).values)
            batch_labels.append(self.df.loc[sampled_indices, 'label'].values)

        # 当前批次是否已经取得到
        batch_features = np.array(batch_features)
        batch_labels = np.array(batch_labels)

        # 返回形状 [bs*100, fea_dim] 和 [bs*100, 1]
        # print(batch_features.shape)
        # print(batch_labels.dtype)
        return batch_features,batch_labels[..., np.newaxis]  # 去掉 uid 和 label 的特征
    
    def on_epoch_end(self):
        np.random.shuffle(self.uids)
    
def pairwise_loss(pair_outputs,list_pxtr_labels):
    #print("===============split================")
    hinge=0.1
    mask_weight=0.5
    pair_loss_weight = 0.01
    list_num=100
    pred_pair = tf.reshape(pair_outputs, (-1, list_num, 1))
    #tf.print("pred_pair",pred_pair)
    labels = tf.reshape(list_pxtr_labels, (-1, list_num, 1))
    print(pred_pair.shape)
    # 局部loss
    pairwise_loss_matrix = pred_pair - tf.transpose(pred_pair, (0,2,1))
    print(pairwise_loss_matrix.shape)
    pairwise_loss = tf.nn.relu(pairwise_loss_matrix + hinge)
    print("pw_loss",tf.reduce_sum(pairwise_loss))
    # 1. labels_i - labels_j < -margin <=> labels_i < labels_j -margin  j比i排名靠后比如j是4，i是0
    cond1 = tf.cast(tf.less(labels - tf.transpose(labels, (0,2,1)), -0.00000001), dtype=tf.float32)
    # 2. 随机取样 shape=[batch_size,list_num,list_num]   element = pred_pair_difference
    cond_shape=tf.shape(cond1)
    random_tensor = tf.random.uniform(shape=cond_shape, minval=0, maxval=1, dtype=tf.float32)
    cond2 = tf.cast(tf.less(random_tensor, mask_weight), dtype=tf.float32)
    pairwise_weights = cond1 * cond2
    pw_loss = pair_loss_weight * tf.reduce_sum(pairwise_loss * pairwise_weights)
    return pw_loss
