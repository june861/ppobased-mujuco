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

env_id = "Humanoid-v4"
env = gym.make(env_id)
env.reset()
terminal = False
while not terminal:
    action = np.tanh(np.random.randn(17,)) * 0.4
    obs_, reward, terminal, trunc, env_info =  env.step(action)


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