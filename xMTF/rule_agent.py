import numpy as np


class RuleAgent(object):
    def __init__(self, a_dim=3, a_bound=1.0, a=[], policy='random'):
        self.a_dim = a_dim
        self.a_bound = a_bound
        self.a = a
        self.policy = policy

    def get_actor(input_state_shape, name=''):
        pass

    def get_critic(input_state_shape, input_action_shape, name=''):
        pass

    def copy_para(from_model, to_model):
        pass


    def ema_update(self):
        pass

    def choose_action(self, s, avg_pxtr, prerank_a=[]):
        """
        根据策略选择动作
        """
        if self.policy == 'random':
            return np.random.uniform(low=0, high=self.a_bound, size=self.a_dim)
        elif self.policy == 'one':
            return np.full(shape=self.a_dim, fill_value=self.a_bound)
        elif self.policy == 'fixed':
            return np.array(self.a)

    def learn(self):
        pass

    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        pass

    def save_ckpt(self, mfc_mode, exit_mode, step_id, formula):
        pass

    def load_ckpt(self, mfc_mode, exit_mode, step_id, formula):
        pass
