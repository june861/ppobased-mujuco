# -*- encoding: utf-8 -*-
'''
@File       :for_obs.py
@Description:
@Date       :2024/12/14 15:31:19
@Author     :junweiluo
@Version    :python
'''

import numpy as np
import gymnasium as gym



class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_actions):
        super().__init__(env)
        self.n_actions = n_actions
        self.action_bins = np.linspace(env.action_space.low[0], env.action_space.high[0], n_actions)
        self.action_space = gym.spaces.Discrete(self.n_actions * env.action_space.shape[0])
    
    def action(self, action):
        # 根据离散动作索引解码出每个维度的动作
        action_indices = np.unravel_index(action, [self.n_actions] * len(self.action_space_shape))
        continuous_action = np.array([self.action_bins[i][action_indices[i]] for i in range(len(self.action_space_shape))])
        return continuous_action

env_ids = [
    "HalfCheetah-v4",
    "HumanoidStandup-v4",
    "Humanoid-v4",
    "Walker2d-v4",
    "Ant-v4",
    "InvertedDoublePendulum-v4",
    "Hopper-v4",
    "Reacher-v4",
    "InvertedPendulum-v4",
    ]

for env_id in env_ids:
    env = gym.make(env_id)
    print(env_id, env.action_space, env.action_space.low, env.action_space.high, sep="\t")

# terminal = False
# while not terminal:
#     action = np.tanh(np.random.randn(17,)) * 0.4
#     action =np.tanh(np.random.randn(1,)) * 0.4
#     obs_, reward, terminal, trunc, env_info =  env.step(action)


# Humanoid-v4 Reward Function
# ============================================================
# def step(self, action):
#     xy_position_before = mass_center(self.model, self.data)
#     self.do_simulation(action, self.frame_skip)
#     xy_position_after = mass_center(self.model, self.data)

#     xy_velocity = (xy_position_after - xy_position_before) / self.dt
#     x_velocity, y_velocity = xy_velocity

#     ctrl_cost = self.control_cost(action)

#     forward_reward = self._forward_reward_weight * x_velocity
#     healthy_reward = self.healthy_reward

#     rewards = forward_reward + healthy_reward

#     observation = self._get_obs()
#     reward = rewards - ctrl_cost
#     terminated = self.terminated
#     info = {
#         "reward_linvel": forward_reward,
#         "reward_quadctrl": -ctrl_cost,
#         "reward_alive": healthy_reward,
#         "x_position": xy_position_after[0],
#         "y_position": xy_position_after[1],
#         "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
#         "x_velocity": x_velocity,
#         "y_velocity": y_velocity,
#         "forward_reward": forward_reward,
#     }

#     if self.render_mode == "human":
#         self.render()
#     return observation, reward, terminated, False, info