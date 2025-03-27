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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import random
import time
import wandb
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import compute_advantages, atari_conf
from agent import Agent
from trainer import AtariTrainer
from buffer import AtariBuffer
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk


def launch_log(args):
    run_name = f"{args.exp_name}_{int(time.time())}_{os.getppid()}"
    if args.track:
        wandb_group = args.exp_name
        args.logger.info(f"use wandb to log. Project is {args.wandb_project_name}, Group is {wandb_group}, Name is {run_name}")
        wandb.init(
            project=args.wandb_project_name,
            group=wandb_group,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    return writer, run_name


if __name__ == "__main__":

    args = atari_conf()

    writer, run_name = launch_log(args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    replay_buffer = AtariBuffer(args = args)
    trainer = AtariTrainer(args = args, agent = agent, optimizer = optimizer, writer = writer)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, total_logits = agent.get_action_and_value(next_obs)
            
            # TRY NOT TO MODIFY: execute the game and log data.
            obs_, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            replay_buffer.push(next_obs, action, torch.tensor(reward).to(device).view(-1), next_done, logprob, value.flatten(), total_logits, step)
            next_obs = obs_
            next_done = np.logical_or(terminations, truncations)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        
        obs, actions, rewards, dones, logprobs, values, total_logits = replay_buffer.pop()

        returns, advantages = compute_advantages(
            args = args, 
            agent = agent, 
            rewards = rewards, 
            values = values, 
            next_obs = next_obs, 
            next_done = next_done, 
            dones = dones
        )
        
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_total_logits = total_logits.reshape((-1,) + envs.single_action_space.shape)
        
        buffer = (b_obs, b_actions, b_logprobs, b_returns, b_advantages, b_values, b_total_logits)

        trainer.train(global_step = global_step, buffer = buffer)

    envs.close()
    writer.close()

