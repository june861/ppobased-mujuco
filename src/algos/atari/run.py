 # -*- encoding: utf-8 -*-
'''
@File       :atari_main.py
@Description:atari main function
@Date       :2025/03/26 18:44:45
@Author     :junweiluo
@Version    :python
'''

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import gymnasium as gym
import torch.optim as optim
from utils import get_conf
from .agent import Agent
from trainer import AtariTrainer
from buffer import AtariBuffer
from runner import AtariRunner

if __name__ == "__main__":
    # args = tyro.cli(Args)
    args = get_conf()
    config_ = {
        "all_args" : args,
        "trainer": None,
        "buffer": None,
        "envs": None,   
    }
    
    runner = AtariRunner(config_)
    envs = gym.vector.SyncVectorEnv(
        [runner.make_envs(i) for i in range(runner.all_args.num_envs)]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        runner.all_args.logger.error(f"only discrete action space is supported")
        raise TypeError(f'{type(envs.single_action_space)} != {gym.spaces.Discrete}')

    runner.all_args.single_observation_space = envs.single_observation_space
    runner.all_args.single_action_space = envs.single_action_space
    runner.all_args.discrete_action_space_n = envs.single_action_space.n
    runner.envs = envs

    # logger.info(f"env is {args.env_id}, n_rollout_thread is {args.num_envs}, sample action num is {args.sample_action_num}")
    agent = Agent(num_actions = runner.all_args.discrete_action_space_n).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    trainer = AtariTrainer(runner.all_args, agent, optimizer, runner.writer)
    replay_buffer = AtariBuffer(args)
    
    runner.trainer = trainer
    runner.buffer = replay_buffer
    runner.env_reset()
    
    runner.run()

    runner.envs.close()
    runner.writer.close()


