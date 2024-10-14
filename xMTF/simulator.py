import sys
print("Python version: {}". format(sys.version))

import numpy as np
print("numpy version: {}". format(np.__version__))

import pandas as pd
print("pandas version: {}". format(pd.__version__))

import gym
print("gym version: {}". format(gym.__version__))
from gym import spaces

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
# os.environ["OMP_NUM_THREADS"] = "3"
import argparse
import time
from datetime import datetime

import tensorflow as tf
print("tensorflow version: {}". format(tf.__version__))

import tensorlayer as tl
print("tensorlayer version: {}". format(tl.__version__))

from ddpg import DDPG
from td3 import TD3
from rule_agent import RuleAgent

import optuna

import copy

from util import reduce_mem_usage, is_empty_file

import gc
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def get_single_fullrank_pxtr(fullrank_df, video_id, mode='binary'):
    """
    直接从精排预测文件中取出单个视频的 pxtr 值
    """
    row = fullrank_df.loc[fullrank_df['video_id']==video_id]
    if mode == 'binary':
        values = row[[f"{label}_binary" for label in satisfaction_labels]].values.tolist()
    else:
        values = row[satisfaction_labels].values.tolist()

    try:
        return values[0]
    except:
        raise Exception(f"该用户找不到 video_id={video_id} 的数据")


class User(object):
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_tower_feature = user_tower_is_click_all.loc[user_tower_is_click_all['user_id']==user_id].drop(['user_id'], axis=1).values.tolist()[0]  # 从粗排双塔取用户emb特征
        self.view_count = 0  # 用户浏览视频的次数
        self.viewed_videos = []  # 用户浏览过的视频列表
        self.satisfaction = user_view.loc[user_view['user_id']==user_id, 'user_init_satisfaction'].item()  # 用户初始满意度

        # self.prerank_df = prerank_df_all.loc[prerank_df_all['user_id']==self.user_id]  # 用户全部的粗排预测数据
        # self.fullrank_df = fullrank_df_all.loc[fullrank_df_all['user_id']==self.user_id]  # 用户全部的精排预测数据
        # 注意prerank_df和fullrank_df的读取路径
        self.prerank_df = pd.read_parquet(f'./predict/{data_type}/prerank_mfc/{user_id}.parquet')  # 用户全部的粗排预测数据
        self.fullrank_df = pd.read_parquet(f'./predict/{data_type}/fullrank_mfc_{step_id}/{user_id}.parquet')  # 用户全部的精排预测数据

        self.prerank_df['play_time_s_norm'] = self.prerank_df['play_time_s'].rank() / len(self.prerank_df)
        self.fullrank_df['play_time_s_norm'] = self.fullrank_df['play_time_s'].rank() / len(self.fullrank_df)

        # for label, threshold in zip(prerank_labels, prerank_thresholds):
        #     self.prerank_df[f'{label}_binary'] = self.prerank_df[label].apply(lambda x: 1 if x >= threshold else 0)
        for label, threshold in zip(['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view'], fullrank_thresholds):
            self.fullrank_df[f'{label}_binary'] = self.fullrank_df[label].apply(lambda x: 1 if x >= threshold else 0)

        # print(f"用户 {self.user_id} 的粗排预测数据: {self.prerank_df.shape}")
        # print(f"用户 {self.user_id} 的精排预测数据: {self.fullrank_df.shape}")
        self.selected_prerank_df = None  # 用户粗排 topk 视频数据(1000)
        self.selected_fullrank_df = None  # 根据粗排 topk 的结果筛选精排视频数据(这里是 1000, 非 1)
        self.final_fullrank_df = None  # 用户精排排 topk 视频数据(1)
        self.watch_time = 0  # 用户观看视频的总时长

        # print(f"用户 {self.user_id} 初始满意度: {self.satisfaction}")

    def _update_selected_fullrank_df(self):
        """
        更新用户的精排 topk 视频数据
        """
        self.selected_fullrank_df = self.fullrank_df.loc[self.fullrank_df['video_id'].isin(self.selected_prerank_df['video_id'])]

    def get_avg_prerank_pxtr(self):
        """
        粗排的 avg_pxtr 对所有视频做计算
        """
        return self.prerank_df.mean()[prerank_labels].values.tolist()

    def get_avg_selected_fullrank_pxtr(self):
        """
        精排的 avg_pxtr 对粗排召回的视频做计算
        """
        return self.selected_fullrank_df.mean()[fullrank_labels].values.tolist()

    def _calculate_reward(self, video_id, reward_type='pxtr'):
        """
        计算用户的奖励, 这里是真实奖励
        """
        reward = 0

        if 'watch_time' in reward_type:
            reward += self.final_fullrank_df['play_time_s'].item() / 400
        elif reward_type == 'pxtr':
            pxtres = get_single_fullrank_pxtr(self.fullrank_df, video_id)
            for i, pxtr in enumerate(pxtres):
                reward += pxtr * weights[i]

        return reward

    def view(self, video_id):
        """
        用户观看视频, 返回 done 标志
        """
        if exit_mode == 'satisfy':
            if self.satisfaction < 0.1 or self.view_count >= 97:
                # 退出模拟
                print(f"用户 {self.user_id} 满意度过低, 已退出本次模拟, 总观看视频数: {self.view_count}, 总观看时间: {self.watch_time:0.1f} s")
                return True, 0
        elif exit_mode == 'vv':
            if self.view_count >= 50:  # 用户vv达到一定次数
                # 退出模拟
                print(f"用户 {self.user_id} 达到vv上限, 已退出本次模拟, 总观看视频数: {self.view_count}, 总观看时间: {self.watch_time:0.1f} s")
                return True, 0
        else:
            raise RuntimeError('Invalid Exit Mode')
        
        # 用户浏览一次，更新满意度和浏览次数
        self.view_count += 1
        self.viewed_videos.append(video_id)
        video_watch_time = self.final_fullrank_df['play_time_s'].item()
        self.watch_time += video_watch_time

        reward = self._calculate_reward(video_id, reward_type=reward_type)
        new_satisfaction = self.satisfaction * 0.95 + 0.05 * reward

        print(f"用户 {self.user_id} 浏览了视频 {video_id}, 观看时长为: {video_watch_time:0.1f} s, reward 数值为: {reward:0.4f}, 满意度变化: {self.satisfaction:0.4f} => {new_satisfaction:0.4f}")
        print(f"用户 {self.user_id} 总计浏览视频次数: {self.view_count}, 总计观看时长为: {self.watch_time:0.1f} s, 精排候选集数目: {len(self.selected_fullrank_df)}")

        if len(self.selected_fullrank_df) <= 950:
            raise RuntimeError('Invalid Fullrank')
        
        self.satisfaction = new_satisfaction
        return False, reward

class Simulator(object):
    def __init__(self):
        self.users = {}  # 用字典来保存所有用户

    def add_user(self, user_id):
        # 添加一个新用户
        self.users[user_id] = User(user_id)

    def get_user(self, user_id):
        # 获取一个用户，如果不存在就新建一个
        if user_id not in self.users:
            self.add_user(user_id)
        return self.users[user_id]

    # def user_view(self, user_id, video_id):
    #     user = self.get_user(user_id)
    #     return user.view(video_id)

    def reset(self):
        self.users = {}


class ESEnv(gym.Env):
    def __init__(self):
        super(ESEnv, self).__init__()

        self.simulator = Simulator()

    def step(self, user_id, reward_type='pxtr'):
        """
        定义环境的转移动态和奖励函数, 这里是模拟的奖励
        """
        user = self.simulator.get_user(user_id)

        video_id = user.final_fullrank_df['video_id'].item()  # 取出精排 top1 的 video_id
        done, true_reward = user.view(video_id)

        # 计算奖励(同 User 中的计算方式)
        reward = 0
        if reward_type =='watch_time':
            reward += user.final_fullrank_df['play_time_s'].item() / 400
        elif reward_type =='watch_time_norm':
            reward += user.final_fullrank_df['play_time_s_norm'].item()
        elif reward_type == 'pxtr':
            pxtres = get_single_fullrank_pxtr(user.final_fullrank_df, video_id)
            for i, pxtr in enumerate(pxtres):
                reward += pxtr * weights[i]


        # if mode == 'binary':
        #     for i, label in enumerate(fullrank_labels):
        #         reward += user.final_fullrank_df[f"{label}_binary"].item() * weights[i]
        # else:
        #     for i, label in enumerate(fullrank_labels):
        #         reward += user.final_fullrank_df[label].item() * weights[i]

        view_count = user.view_count

        return np.array(user.user_tower_feature + [view_count/100]), reward, done, {}, true_reward

    def reset(self, user_id):
        """
        定义环境的初始状态
        """
        self.simulator.reset()
        user = self.simulator.get_user(user_id)
        return np.array(user.user_tower_feature + [0.0])

    def render(self):
        """
        定义环境的渲染方法
        """
        pass

    def prerank(self, user_id, action):
        """
        根据用户的历史浏览记录，对 prerank_df 进行过滤，计算 prerank_es_score, 并返回前 prerank_topk_num 个视频
        """
        user = self.simulator.get_user(user_id)

        user.prerank_df = user.prerank_df.loc[~user.prerank_df['video_id'].isin(user.viewed_videos)]
        user.prerank_df['prerank_es_score'] = 0
        for i, label in enumerate(prerank_labels):
            user.prerank_df['prerank_es_score'] += user.prerank_df[label] * np.float32(action[i])

        user.selected_prerank_df = user.prerank_df.nlargest(prerank_topk_num, 'prerank_es_score')

        user._update_selected_fullrank_df()

    def fullrank(self, user_id, action):
        """
        对 selected_fullrank_df 计算 fullrank_es_score, 并返回前 fullrank_topk_num 个视频
        """
        user = self.simulator.get_user(user_id)

        user.selected_fullrank_df[f'fullrank_es_score'] = 0
        for i, label in enumerate(fullrank_labels):
            
            if formula == "add":
                user.selected_fullrank_df[f'fullrank_es_score'] += user.selected_fullrank_df[label] * np.float32(action[i])
            elif formula == 'multiply':
                user.selected_fullrank_df[f'fullrank_es_score'] += np.log(user.selected_fullrank_df[label] + 0.1) * np.float32(action[i])
            else:
                raise RuntimeError('Invalid Formula')


        user.final_fullrank_df = user.selected_fullrank_df.nlargest(fullrank_topk_num, 'fullrank_es_score')


def get_agents(policy, a1=[0.134543, 0.904333, 0.408954], a2=[0.581594, 0.120548, 0.431726, 0.692834, 0.095083, 0.120036], summary_writer=None):
    """
    根据 policy 选择不同的 agent
    """

    if policy == 'random':
        prerank_agent = RuleAgent(a_dim=3, a_bound=1.0, policy='random')
        fullrank_agent = RuleAgent(a_dim=6, a_bound=1.0, policy='random')
    elif policy == 'one':
        prerank_agent = RuleAgent(a_dim=3, a_bound=1.0, policy='one')
        fullrank_agent = RuleAgent(a_dim=6, a_bound=1.0, policy='one')
    elif policy == 'fixed':
        prerank_agent = RuleAgent(a_dim=3, a=a1, policy='fixed')
        fullrank_agent = RuleAgent(a_dim=6, a=a2, policy='fixed')
    elif policy == 'ddpg':
        prerank_agent = RuleAgent(a_dim=3, a_bound=1.0, policy='one')
        fullrank_agent = DDPG(model_type='ddpg', data_type=data_type, reward_type=reward_type, stage='fullrank',a_dim=6, s_dim=64+1+6, a_bound=1.0, memory_capacity=memory_capacity, batch_size=batch_size, summary_writer=summary_writer)
    elif policy == 'td3':
        prerank_agent = RuleAgent(a_dim=3, a_bound=1.0, policy='one')
        fullrank_agent = TD3(model_type='td3', data_type=data_type, reward_type=reward_type, stage='fullrank',a_dim=6, s_dim=64+1+6, a_bound=1.0, memory_capacity=memory_capacity, batch_size=batch_size, summary_writer=summary_writer)
    else:
        raise RuntimeError('Invalid Class')

    print(f'Using policy {policy}')
    return prerank_agent, fullrank_agent


def train(do_test_step=False):

    summary_writer = tf.summary.create_file_writer(f"logs/{policy}_{now}")  # Replace with your log directory

    train_env = ESEnv()

    prerank_agent, fullrank_agent = get_agents(policy=policy, summary_writer=summary_writer)

    reward_buffer = [] # 用于记录每个EP的reward，统计变化

    np.random.seed(seed=seed)
    selected_user_ids = np.random.choice(unique_user_ids, size=len(unique_user_ids), replace=False)

    pbar = tqdm(range(len(selected_user_ids)))
    # pbar = tqdm(range(1, 10))
    # pbar = tqdm([1] * 10000)
    already_printed = False
    already_trained = False
    for i, idx in  enumerate(pbar):
        current_user_id = unique_user_ids[idx]  # 当前user_id

        s = train_env.reset(current_user_id)
        ep_reward = 0       #记录当前EP的reward
        done = False
        eval_step = 0

        while not done:

            user = train_env.simulator.get_user(current_user_id)

            user_prerank_avg_pxtr = user.get_avg_prerank_pxtr()
            user_prerank_a = prerank_agent.choose_action(s=s, avg_pxtr=user_prerank_avg_pxtr)
            if policy in ['maddpg', 'maddpg_cascade']:
                user_prerank_a = np.clip(np.random.normal(user_prerank_a, var/np.sqrt(i+1)), 0.05, 0.95) # 增加探索
            train_env.prerank(user_id=current_user_id, action=user_prerank_a)

            user_selected_fullrank_avg_pxtr = user.get_avg_selected_fullrank_pxtr()
            prerank_a = np.array(user_prerank_a) if policy == 'maddpg_cascade' else []
            user_fullrank_a = fullrank_agent.choose_action(s=s, avg_pxtr=user_selected_fullrank_avg_pxtr, prerank_a=prerank_a)
            user_fullrank_a = np.clip(np.random.normal(user_fullrank_a, var/np.sqrt(i+1)), 0.05, 0.95) # 增加探索
            train_env.fullrank(user_id=current_user_id, action=user_fullrank_a)

            # 与环境进行互动
            s_, r, done, info, true_r = train_env.step(user_id=current_user_id, reward_type=reward_type)
            # print(f"a1:{user_prerank_a}, a2: {user_fullrank_a}, s: {s}, s_: {s_}, r: {r:0.4f}, done: {done}")
            print(f"i:{i}, pointer:{fullrank_agent.pointer}, var:{var/np.sqrt(i+1):.4f}, a1:{np.round(user_prerank_a, 2)}, a2: {np.round(user_fullrank_a, 2)}, r: {r:.4f}, true_r:{true_r:.4f}, done: {done}")

            # 保存s，a，r，s_
            prerank_agent.store_transition(np.concatenate((s, user_prerank_avg_pxtr)), user_prerank_a, r / 10, np.concatenate((s_, user_prerank_avg_pxtr)))
            fullrank_agent.store_transition(np.concatenate((s, user_selected_fullrank_avg_pxtr, prerank_a)), user_fullrank_a, r / 10, np.concatenate((s_, user_selected_fullrank_avg_pxtr, prerank_a)))

            # 第一次数据满了，就可以开始学习
            if policy != 'random' and fullrank_agent.pointer >= memory_capacity + fullrank_agent.step * batch_size:
                if not already_printed:
                    print(f"积攒数据已满, 开始训练, 当前EP: {i}/{len(selected_user_ids)}")
                    already_printed = True

                if eval_step < max_steps:
                    if do_test_step and fullrank_agent.step % 5 == 0:
                        print(f"开始用户 {current_user_id} 的第 {eval_step} 轮测试, 使用第 {fullrank_agent.step} 步的模型")
                        test_env = copy.deepcopy(train_env)
                        test_step(test_env=test_env, prerank_agent=prerank_agent, fullrank_agent=fullrank_agent, step=fullrank_agent.step)
                    prerank_agent.learn()
                    fullrank_agent.learn()
                    eval_step += 1
                else:
                    already_trained = True
                    break

            #输出数据记录
            s = s_
            ep_reward += r  #记录当前EP的总reward

        reward_buffer.append(ep_reward)
        pbar.set_postfix({"Episode Reward": ep_reward}, refresh=True)
        # print(f'Episode: {i}/{len(unique_user_ids)}  | Episode Reward: {ep_reward:.4f}')

        if already_trained:
            break

    prerank_agent.save_ckpt(mfc_mode=mfc_mode, exit_mode=exit_mode, step_id=step_id, formula=formula)
    fullrank_agent.save_ckpt(mfc_mode=mfc_mode, exit_mode=exit_mode, step_id=step_id, formula=formula)

    t2 = time.time()
    print(f"训练用时: {t2-t1:0.1f} 秒")


def test_step(test_env, prerank_agent, fullrank_agent, step):

    t0 = time.time()
    np.random.seed(seed=seed)

    reward_buffer = []

    view_count_all = 0
    watch_time_all = 0

    results_file = os.path.join(f'./results/{data_type}', f'results_{policy}_{reward_type}_step.csv')

    with open(results_file, "a") as gp:
        if is_empty_file(results_file):
            gp.write("exp_time,policy,reward_type,user_id,step,view_count,watch_time,ep_reward\n")
            gp.flush()

        # for i in range(len(unique_user_ids)):
        for i in tqdm(range(1, 2)):
        # selected_user_ids = np.random.choice(unique_user_ids, size=1, replace=False)
        # for i in tqdm(range(1, len(selected_user_ids))):
            current_user_id = unique_user_ids[i]  # 当前user_id
            s = test_env.reset(user_id=current_user_id)
            user = test_env.simulator.get_user(current_user_id)

            ep_reward = 0
            done = False
            while not done:

                user_prerank_avg_pxtr = user.get_avg_prerank_pxtr()
                user_prerank_a = prerank_agent.choose_action(s=s, avg_pxtr=user_prerank_avg_pxtr)
                test_env.prerank(user_id=current_user_id, action=user_prerank_a)

                user_selected_fullrank_avg_pxtr = user.get_avg_selected_fullrank_pxtr()
                prerank_a = np.array(user_prerank_a) if policy == 'maddpg_cascade' else []
                user_fullrank_a = fullrank_agent.choose_action(s=s, avg_pxtr=user_selected_fullrank_avg_pxtr, prerank_a=prerank_a)
                test_env.fullrank(user_id=current_user_id, action=user_fullrank_a)

                s, r, done, info, true_r = test_env.step(user_id=current_user_id, reward_type=reward_type)
                # print(f"s:{s}, r:{r}, done:{done}")
                print(f"i:{i}, a1:{np.round(user_prerank_a, 2)}, a2:{np.round(user_fullrank_a, 2)}, r:{r:.4f}, true_r:{true_r:.4f}, done:{done}")

                ep_reward += true_r

                if done:
                    break
            reward_buffer.append(ep_reward)
            watch_time_all += user.watch_time
            view_count_all += user.view_count
            # print(f"user_id {current_user_id}, 观看次数: {user.view_count}, 播放时长: {user.watch_time:.1f} s, reward: {ep_reward:.1f}")

            gp.write(f"{now},{policy},{reward_type},{current_user_id},{step},{user.view_count},{user.watch_time},{ep_reward}\n")
            gp.flush()

    t1 = time.time()
    print(f"测试用时: {t1-t0:.1f} 秒, 这些用户的平均观看次数为: {view_count_all/1}, 平均播放时长: {watch_time_all/1:.1f} s, 平均reward: {sum(reward_buffer)/1:.1f}")


def test():

    test_env = ESEnv()

    prerank_agent, fullrank_agent = get_agents(policy=policy)

    t0 = time.time()
    np.random.seed(seed=seed)

    reward_buffer = []

    prerank_agent.load_ckpt(mfc_mode=mfc_mode, exit_mode=exit_mode, step_id=step_id, formula=formula)
    fullrank_agent.load_ckpt(mfc_mode=mfc_mode, exit_mode=exit_mode, step_id=step_id, formula=formula)

    view_count_all = 0
    watch_time_all = 0

    results_file = os.path.join(f'./results/{data_type}', f'results_{policy}_{reward_type}.csv')

    with open(results_file, "a") as gp:
        if is_empty_file(results_file):
            gp.write("policy,reward_type,user_id,view_count,watch_time,ep_reward\n")
            gp.flush()

        # for i in range(len(unique_user_ids)):
        # for i in tqdm(range(0, 10)):
        selected_user_ids = np.random.choice(unique_user_ids, size=1, replace=False)
        for i in tqdm(range(0, len(selected_user_ids))):
            current_user_id = selected_user_ids[i]  # 当前user_id
            s = test_env.reset(user_id=current_user_id)
            user = test_env.simulator.get_user(current_user_id)

            ep_reward = 0
            done = False
            while not done:

                user_prerank_avg_pxtr = user.get_avg_prerank_pxtr()
                user_prerank_a = prerank_agent.choose_action(s=s, avg_pxtr=user_prerank_avg_pxtr)
                test_env.prerank(user_id=current_user_id, action=user_prerank_a)

                user_selected_fullrank_avg_pxtr = user.get_avg_selected_fullrank_pxtr()
                prerank_a = np.array(user_prerank_a) if policy == 'maddpg_cascade' else []
                user_fullrank_a = fullrank_agent.choose_action(s=s, avg_pxtr=user_selected_fullrank_avg_pxtr, prerank_a=prerank_a)
                test_env.fullrank(user_id=current_user_id, action=user_fullrank_a)

                s, r, done, info, true_r = test_env.step(user_id=current_user_id, reward_type=reward_type)
                # print(f"s:{s}, r:{r}, done:{done}")
                print(f"i:{i}, a1:{np.round(user_prerank_a, 2)}, a2:{np.round(user_fullrank_a, 2)}, r:{r:.4f}, true_r:{true_r:.4f}, done:{done}")

                ep_reward += true_r

                if done:
                    break
            reward_buffer.append(ep_reward)
            watch_time_all += user.watch_time
            view_count_all += user.view_count
            # print(f"user_id {current_user_id}, 观看次数: {user.view_count}, 播放时长: {user.watch_time:.1f} s, reward: {ep_reward:.1f}")

            gp.write(f"{policy},{reward_type},{current_user_id},{user.view_count},{user.watch_time},{ep_reward}\n")
            gp.flush()

    t1 = time.time()
    print(f"测试用时: {t1-t0:.1f} 秒, 这些用户的平均观看次数为: {view_count_all/len(selected_user_ids)}, 平均播放时长: {watch_time_all/len(selected_user_ids):.1f} s, 平均reward: {sum(reward_buffer)/len(selected_user_ids):.1f}")


def search():
    # Define an objective function to be minimized.
    def objective(trial):

        # Invoke suggest methods of a Trial object to generate hyperparameters.
        pre_a = [trial.suggest_float(f'pre_a_{i}', 0, 1) for i in range(3)]
        full_a = [trial.suggest_float(f'full_a_{i}', 0, 1) for i in range(6)]
        # print(f"pre_a: {pre_a}, full_a: {full_a}")

        search_env = ESEnv()

        prerank_agent, fullrank_agent = get_agents(policy=policy, a1=pre_a, a2=full_a)

        t0 = time.time()
        np.random.seed(seed=seed)

        reward_buffer = []

        prerank_agent.load_ckpt(mfc_mode=mfc_mode, exit_mode=exit_mode,step_id=step_id, formula=formula)
        fullrank_agent.load_ckpt(mfc_mode=mfc_mode, exit_mode=exit_mode,step_id=step_id, formula=formula)
        view_count_all = 0
        watch_time_all = 0


        # for i in range(len(unique_user_ids)):
        # for i in tqdm(range(0, 10)):
        selected_user_ids = np.random.choice(unique_user_ids, size=10, replace=False)
        for i in tqdm(range(len(selected_user_ids))):
            current_user_id = selected_user_ids[i]  # 当前user_id
            s = search_env.reset(user_id=current_user_id)
            user = search_env.simulator.get_user(current_user_id)

            ep_reward = 0
            done = False
            while not done:

                user_prerank_avg_pxtr = user.get_avg_prerank_pxtr()
                user_prerank_a = prerank_agent.choose_action(s=s, avg_pxtr=user_prerank_avg_pxtr)
                search_env.prerank(user_id=current_user_id, action=user_prerank_a)

                user_selected_fullrank_avg_pxtr = user.get_avg_selected_fullrank_pxtr()
                prerank_a = np.array(user_prerank_a) if policy == 'maddpg_cascade' else []
                user_fullrank_a = fullrank_agent.choose_action(s=s, avg_pxtr=user_selected_fullrank_avg_pxtr, prerank_a=prerank_a)
                search_env.fullrank(user_id=current_user_id, action=user_fullrank_a)

                s, r, done, info, true_r = search_env.step(user_id=current_user_id, reward_type=reward_type)
                # print(f"s:{s}, r:{r}, done:{done}")
                # print(f"i:{i}, a1:{np.round(user_prerank_a, 2)}, a2:{np.round(user_fullrank_a, 2)}, r:{r:.4f}, done:{done}")

                ep_reward += true_r

                if done:
                    break
            reward_buffer.append(ep_reward)
            watch_time_all += user.watch_time
            view_count_all += user.view_count
            # print(f"user_id {current_user_id}, 观看次数: {user.view_count}, 播放时长: {user.watch_time:.1f} s, reward: {ep_reward:.1f}")

        t1 = time.time()
        print(f"测试用时: {t1-t0:.1f} 秒, 这些用户的平均观看次数为: {view_count_all/len(selected_user_ids)}, 平均播放时长: {watch_time_all/len(selected_user_ids):.1f} s, 平均reward: {sum(reward_buffer)/len(selected_user_ids):.1f}")

        error = sum(reward_buffer) / len(selected_user_ids)

        return error  # An objective value linked with the Trial object.

    study = optuna.create_study(direction='maximize')  # Create a new study.
    study.optimize(objective, n_trials=100, n_jobs=4)  # Invoke optimization of the objective function.


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AAAI-2023...")
    parser.add_argument("--mfc_mode", type=str, default="")
    parser.add_argument("--exit_mode", type=str, default="satisfy")  # 默认模式为"satisfy"，另一种模型为"vv"
    parser.add_argument('--step_id', type=str, default="step1")
    parser.add_argument('--formula', type=str, default="add")
    parser.add_argument("--policy", type=str, default="maddpg")
    parser.add_argument("--reward_type", type=str, default="watch_time")
    parser.add_argument("--data_path", type=str, default="KuaiRand-1K")
    parser.add_argument("--prerank_topk_num", type=int, default=1000)
    parser.add_argument("--fullrank_topk_num", type=int, default=1)
    parser.add_argument("--memory_capacity", type=int, default=256, help='size of replay buffer')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--var", type=float, default=1.0, help='control exploration')
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--search', dest='search', action='store_true')
    args = parser.parse_args()

    mfc_mode = args.mfc_mode
    print(mfc_mode)
    exit_mode = args.exit_mode
    step_id = args.step_id
    formula = args.formula
    policy = args.policy
    reward_type = args.reward_type
    data_path = args.data_path
    prerank_topk_num = args.prerank_topk_num
    fullrank_topk_num = args.fullrank_topk_num
    memory_capacity = args.memory_capacity
    batch_size = args.batch_size
    var = args.var
    seed = args.seed
    max_steps = args.max_steps

    data_type = data_path.split('-')[-1].lower()

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(step_id)
    t0 = time.time()
    df1 = pd.read_pickle(os.path.join(data_path, f"data/log_standard_4_08_to_4_21_{data_type}.pkl"))
    df2 = pd.read_pickle(os.path.join(data_path, f"data/log_standard_4_22_to_5_08_{data_type}.pkl"))

    data = pd.concat([df1, df2], axis=0, ignore_index=True)
    data = data[(data['tab'] == 1) & (data['play_time_ms']) > 0]
    data.drop(['tab'], axis=1, inplace=True)
    data = data[data['duration_ms'] > 0]
    data['play_rate'] = data['play_time_ms'] / data['duration_ms']
    data = data[data['play_rate'] <= 5]
    data['play_time_s'] = data['play_time_ms'] / 1000

    del df1, df2
    gc.collect()

    results_path = f'./predict/{data_type}'

    # # 全量数据预测结果（每个 user-item 对）
    # prerank_df_all = reduce_mem_usage(pd.read_pickle(f'./results/prerank_{data_type}.pkl'), verbose=True)
    # fullrank_df_all = reduce_mem_usage(pd.read_pickle(f'./results/fullrank_{data_type}.pkl'), verbose=True)

    # 双塔预测 embedding
    user_tower_is_click_all = pd.read_pickle(f'./results/user_tower_is_click_{data_type}.pkl')
    # item_tower_is_click_all = pd.read_pickle(f'./results/{data_type}/item_tower_is_click.pkl')

    # 根据用户历史浏览确定初始满意度
    # user_view = data.groupby('user_id')['video_id'].count().reset_index().rename(columns={'video_id': 'view_count'})
    # user_view['user_init_satisfaction'] = user_view['view_count'] / 1000
    # user_view['user_init_satisfaction'] = user_view['user_init_satisfaction'].apply(lambda x: 0.15 if x < 0.15 else x)
    user_view = data.groupby('user_id')['play_time_s'].sum().reset_index()
    user_view['user_init_satisfaction'] = user_view['play_time_s'] / 10000
    user_view['user_init_satisfaction'] = user_view['user_init_satisfaction'].apply(lambda x: 0.11 if x < 0.11 else x)

    satisfaction_labels = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view']  # 用于计算满意度
    prerank_labels = ['is_click', 'is_like', 'is_follow']  # 粗排排序指标
    # prerank_thresholds = [0.3463, 0.1598, 0.0416]  # 粗排阈值

    if mfc_mode == 'mfc':
        fullrank_labels = ['is_click_mfc', 'is_like_mfc', 'is_follow_mfc', 'is_comment_mfc', 'is_forward_mfc', 'long_view_mfc']
    else:
        fullrank_labels = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view']  # 精排排序指标
    
    
    fullrank_thresholds = [0.3728, 0.1863, 0.0462, 0.0402, 0.0170, 0.3177]  # 精排阈值

    if reward_type == 'pxtr':
        weights = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]  # 系统内部权重参数, RL不知道这部分参数

    # unique_user_ids = fullrank_df_all['user_id'].unique()  # 找出所有不重复的user_id
    unique_user_ids = np.sort([int(filename.split('.')[0]) for filename in os.listdir(os.path.join(results_path, 'prerank_mfc'))])
    # unique_user_ids = [0]

    t1 = time.time()
    print(f"数据读取完毕, 共有 {len(unique_user_ids)} 个用户, 用时 {t1-t0:.1f} s")


    if args.train:  # train
        train()


    if args.test:  # test
        test()


    if args.search:  # search
        search()
