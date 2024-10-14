"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.

Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0
-------------
"""


import os
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

class DDPG(object):
    """
    DDPG class
    """
    def __init__(self, model_type, data_type, reward_type, stage, a_dim, s_dim, a_bound, lr_a=0.0001, lr_c=0.0002, gamma=0.9, tau=0.01, memory_capacity=1000, batch_size=32, summary_writer=None):
        """
        lr_a                     # learning rate for actor
        lr_c                     # learning rate for critic
        gamma                    # reward discount
        tau                      # soft replacement
        memory_capacity          # size of replay buffer
        batch_size               # update batchsize
        """

        self.model_type = model_type
        self.data_type = data_type
        self.reward_type = reward_type
        self.stage = stage
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.tau = tau
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.summary_writer = summary_writer

        # memory用于储存跑的数据的数组：
        # 保存个数MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：分别是两个state，一个action，和一个reward
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.step = 0

        W_init = tf.random_normal_initializer(mean=0, stddev=0.1)
        b_init = tf.constant_initializer(0.1)

        # 建立actor网络，输入s，输出a
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=32, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l2')(x)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.sigmoid, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x)            #注意这里，先用tanh把范围限定在[-1,1]之间，再进行映射
            return tl.models.Model(inputs=inputs, outputs=x, name=f'{self.stage}_Actor' + name)

        #建立Critic网络，输入s，a。输出Q值
        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=128, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name=f'{self.stage}_Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        #更新参数，只用于首次赋值，之后就没用了
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        #建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        #建立critic_target网络，并和actor参数一致，不能训练
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        #建立ema，滑动平均值
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(self.lr_a)
        self.critic_opt = tf.optimizers.Adam(self.lr_c)


    def ema_update(self):
        """
        滑动平均更新
        """
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        paras = self.actor.trainable_weights + self.critic.trainable_weights    #获取要更新的参数包括actor和critic的
        self.ema.apply(paras)                                                   #主要是建立影子参数
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))                                       # 用滑动平均赋值

    # 选择动作，把s带进入，输出a
    def choose_action(self, s, avg_pxtr, prerank_a=[]):
        """
        Choose action
        :param s: state
        :return: act
        """
        # return self.actor(np.array([s], dtype=np.float32))[0]
        return self.actor(np.array([np.concatenate((s, avg_pxtr, prerank_a))], dtype=np.float32))[0]

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)    #随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]                    #根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]                         #从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  #从bt获得数据a
        br = bt[:, -self.s_dim - 1:-self.s_dim]         #从bt获得数据r
        bs_ = bt[:, -self.s_dim:]                       #从bt获得数据s'

        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + self.gamma * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Writing TD error to TensorBoard
        with self.summary_writer.as_default():
            tf.summary.scalar(f'{self.stage}_td_error', tf.reduce_mean(td_error), step=self.step)

        # Actor：
        # Actor的目标就是获取最多Q值的。
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # 【敲黑板】：注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        # Writing Actor loss to TensorBoard
        with self.summary_writer.as_default():
            tf.summary.scalar(f'{self.stage}_actor_loss', tf.reduce_mean(a_loss), step=self.step)

        self.ema_update()

        self.step += 1

        print(f"模型经过了 {self.step} 次训练")

    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        # 整理s，s_,方便直接输入网络计算
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        #把s, a, [r], s_横向堆叠
        transition = np.hstack((s, a, [r], s_))

        #pointer是记录了曾经有多少数据进来。
        #index是记录当前最新进来的数据位置。
        #所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        #把transition，也就是s, a, [r], s_存进去。
        self.memory[index, :] = transition
        self.pointer += 1
        print(f"memory 位置 {index} 的数据被替换")

        # print(self.memory[:self.pointer, self.s_dim:self.s_dim + self.a_dim].mean(axis=0))

    def save_ckpt(self, mfc_mode, exit_mode, step_id, formula):
        """
        save trained weights
        :return: None
        """
        if mfc_mode == "mfc":
            model_path = f"models/{self.data_type}/{self.model_type}_{self.reward_type}_{exit_mode}_{mfc_mode}_{step_id}"
        else:
            model_path = f"models/{self.data_type}/{self.model_type}_{self.reward_type}_{exit_mode}_{formula}"
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        tl.files.save_weights_to_hdf5(os.path.join(model_path, f'{self.stage}_actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(model_path, f'{self.stage}_actor_target.hdf5'), self.actor_target)
        tl.files.save_weights_to_hdf5(os.path.join(model_path, f'{self.stage}_critic.hdf5'), self.critic)
        tl.files.save_weights_to_hdf5(os.path.join(model_path, f'{self.stage}_critic_target.hdf5'), self.critic_target)

        # self.actor.save_weights(os.path.join(model_path, f'{self.stage}_actor.h5'))
        # self.actor_target.save_weights(os.path.join(model_path, f'{self.stage}_actor_target.h5'))
        # self.critic.save_weights(os.path.join(model_path, f'{self.stage}_critic.h5'))
        # self.critic_target.save_weights(os.path.join(model_path, f'{self.stage}_critic_target.h5'))

    def load_ckpt(self, mfc_mode, exit_mode, step_id, formula):
        """
        load trained weights
        :return: None
        """
        if mfc_mode == "mfc":
            model_path = f"models/{self.data_type}/{self.model_type}_{self.reward_type}_{exit_mode}_{mfc_mode}_{step_id}"
        else:
            model_path = f"models/{self.data_type}/{self.model_type}_{self.reward_type}_{exit_mode}_{formula}"
        
        tl.files.load_hdf5_to_weights_in_order(os.path.join(model_path, f'{self.stage}_actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(model_path, f'{self.stage}_actor_target.hdf5'), self.actor_target)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(model_path, f'{self.stage}_critic.hdf5'), self.critic)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(model_path, f'{self.stage}_critic_target.hdf5'), self.critic_target)
        
        # self.actor.load_weights(os.path.join(model_path, f'{self.stage}_actor.h5'))
        # self.actor_target.load_weights(os.path.join(model_path, f'{self.stage}_actor_target.h5'))
        # self.critic.load_weights(os.path.join(model_path, f'{self.stage}_critic.h5'))
        # self.critic_target.load_weights(os.path.join(model_path, f'{self.stage}_critic_target.h5'))