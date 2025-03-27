# -*- encoding: utf-8 -*-
'''
@File    :   mujuco_main.py
@Time    :   2025/03/25 22:18:00
@Author  :   junewluo 
'''

# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
import gymnasium as gym
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import torch.optim as optim
from utils import get_conf
from agent import Agent
from trainer import MujocoTrainer
from buffer import MujocoBuffer
from runner import MujocoRunner


if __name__ == "__main__":
    # args = tyro.cli(Args)
    args = get_conf()
    if args.sample_action_num == None or (not isinstance(args.sample_action_num, int)) or args.sample_action_num <= 0:
        args.logger.error(f'param of "sample_action_num" must be a integer which is required larger than 0')
        sys.exit(args.sample_action_num)
    
    config_ = {
        "all_args" : args,
        "trainer": None,
        "buffer": None,
        "envs": None,   
    }
    
    runner = MujocoRunner(config_)


    envs = gym.vector.SyncVectorEnv(
        [runner.make_env(i, runner.run_name) for i in range(runner.all_args.num_envs)]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        runner.all_args.logger.error(f"only continuous action space is supported")
        raise TypeError(f'{type(envs.single_action_space)} != {gym.spaces.Box}')

    runner.all_args.single_observation_space = envs.single_observation_space
    runner.all_args.single_action_space = envs.single_action_space
    runner.envs = envs

    # logger.info(f"env is {args.env_id}, n_rollout_thread is {args.num_envs}, sample action num is {args.sample_action_num}")
    agent = Agent(runner.envs, sample_action_num = args.sample_action_num).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    trainer = MujocoTrainer(args, agent, optimizer, runner.writer)
    replay_buffer = MujocoBuffer(args)
    
    runner.trainer = trainer
    runner.buffer = replay_buffer
    runner.env_reset()
    runner.run()

    runner.envs.close()
    runner.writer.close()
