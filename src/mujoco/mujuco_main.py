# -*- encoding: utf-8 -*-
'''
@File    :   mujuco_main.py
@Time    :   2025/03/25 22:18:00
@Author  :   junewluo 
'''

# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import wandb
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from utils import compute_advantages, get_mujuco_config
from agent import Agent
from trainer import MujocoTrainer
from buffer import MujocoBuffer
from torch.utils.tensorboard import SummaryWriter

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def launch_log(args):
    run_name = f"{args.env_id}__{args.exp_name}__seed{args.seed}_{int(time.time())}_ratio2clamp_grad"
    if args.track:
        wandb_group = args.wandb_group if args.wandb_group != None else f"{args.env_id}__{args.exp_name}_clipcoef{str(args.clip_coef)}__ratio2clamp_v1_grad"
        args.logger.info(f"use wandb to log.Project is {args.wandb_project_name}, Group is {wandb_group}, Name is {run_name}")
        wandb.init(
            project=args.wandb_project_name,
            group=wandb_group,
            # entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    return writer, run_name


if __name__ == "__main__":
    # args = tyro.cli(Args)
    args = get_mujuco_config()
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


    writer, run_name = launch_log(args)
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    if args.sample_action_num == None:
        args.sample_action_num = envs.single_action_space.shape[0]

    args.single_observation_space = envs.single_observation_space
    args.single_action_space = envs.single_action_space
    
    # logger.info(f"env is {args.env_id}, n_rollout_thread is {args.num_envs}, sample action num is {args.sample_action_num}")
    agent = Agent(envs, sample_action_num = args.sample_action_num, max_scale = envs.single_action_space.high[0]).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    trainer = MujocoTrainer(args, agent, optimizer, writer)
    replay_buffer = MujocoBuffer(args)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)

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
                action, logprob, _, value, mean_std = trainer.agent.get_action_and_value(next_obs)

            
            # TRY NOT TO MODIFY: execute the game and log data.
            obs_, reward, terminations, truncations, infos = envs.step(action[:,0,:].cpu().numpy())
            data = (next_obs, action, logprob, torch.tensor(reward).to(args.device).view(-1), next_done, value.flatten().detach(), mean_std.loc, mean_std.scale)
            replay_buffer.push(data, step)
            next_obs = obs_
            next_done = np.logical_or(terminations, truncations)
            next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(next_done).to(args.device)

            # v4 version
            if "final_info" in infos:
                for index, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        args.logger.info(f"index = {index}, global_step = {global_step}, episodic_return = {info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"] , global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar("charts/global_step", global_step)

            # v5 version
            # if "episode" in infos:
            #     episode_return = np.sum(infos["episode"]["r"] * infos["episode"]["_r"]) / args.num_envs
            #     episode_length = np.sum(infos["episode"]["l"] * infos["episode"]["_l"]) / args.num_envs
            #     args.logger.info(f"global_step = {global_step}, episodic_return = {episode_return}, episodic_length = {episode_length}")
            #     writer.add_scalar("charts/episodic_return", episode_return, global_step)
            #     writer.add_scalar("charts/episodic_length", episode_length, global_step)
        
        obs, actions, logprobs, rewards, dones, values, means, stds = replay_buffer.pop()

        returns, advantages = compute_advantages(
            args = args, 
            agent = agent, 
            rewards = rewards, 
            values = values, 
            next_obs = next_obs, 
            next_done = next_done, 
            dones = dones
        )

        buffer = (
            obs, logprobs, actions, advantages, returns, values, means, stds
        )
        trainer.train(buffer, global_step)

    envs.close()
    writer.close()
