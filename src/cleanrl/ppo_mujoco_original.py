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
import torch.nn as nn
import torch.optim as optim
import tyro
from utils.mujoco_config import get_config
from torch.distributions.normal import Normal
from utils.mujoco_config import get_config
from utils.utils import compute_advantages
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
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space = env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    # DONE(junweiluo)：增加一个离散化动作的参数
    def __init__(self, envs, sample_action_num = 1, max_scale = 1.0):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        # junweiluo: 增加参数，
        self.sample_action_num = sample_action_num 
        self.scale = max_scale
    
    # junweiluo: 增加函数
    def sample_action(self, probs):
        actions = []
        for _ in range(self.sample_action_num):
            # i_action = torch.tanh(probs.sample()) * self.scale
            # actions.append(i_action)
            i_action = probs.sample()
            actions.append(i_action)
        actions = torch.stack(actions, dim = 1)
        log_probs = self.get_logprobs(actions, probs)

        return actions, log_probs

    def get_logprobs(self, actions, probs):
        """ actions shape is [num_envs, self.sample_action_num, action_dim] """
        log_probs = []
        for i in range(self.sample_action_num):
            log_probs.append(probs.log_prob(actions[:,i,:]))
        
        return torch.stack(log_probs, dim = 1).sum(2)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            # action shape is (num_envs, sample_action_num, action_dim)
            action, log_probs = self.sample_action(probs)
            return action, log_probs, probs.entropy().sum(1), self.critic(x), probs
            # else:
            #     action = probs.sample()
            #     return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
        
        # ppo更新时计算新的log_probs
        log_probs = self.get_logprobs(actions = action, probs = probs)
        
        return action, log_probs, probs.entropy().sum(1), self.critic(x), probs


def compute_kld(mu_1, sigma_1, mu_2, sigma_2):
    return torch.log(sigma_2 / sigma_1) + ((mu_1 - mu_2) ** 2 + (sigma_1 ** 2 - sigma_2 ** 2)) / (2 * sigma_2 ** 2)


if __name__ == "__main__":
    # args = tyro.cli(Args)
    args = get_config()
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

    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    


    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    if args.sample_action_num == None:
        args.sample_action_num = envs.single_action_space.shape[0]
    
    # logger.info(f"env is {args.env_id}, n_rollout_thread is {args.num_envs}, sample action num is {args.sample_action_num}")


    agent = Agent(envs, sample_action_num = args.sample_action_num, max_scale = envs.single_action_space.high[0]).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # junweiluo: 修改一下replay buffer的形状
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs) + (args.sample_action_num, ) + envs.single_action_space.shape).to(args.device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)+ (args.sample_action_num, )).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    means = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(args.device)
    stds = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(args.device)
    

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)
    
    batch_index = -1

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # record return
        total_reward = 0.0
        
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, mean_std = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

                
            actions[step] = action
            logprobs[step] = logprob
            means[step] = mean_std.loc
            stds[step] = mean_std.scale
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action[:,0,:].cpu().numpy())
            total_reward += reward.sum()
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(args.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(next_done).to(args.device)

            # v4 version
            # if "final_info" in infos:
            #     for index, info in enumerate(infos["final_info"]):
            #         if info and "episode" in info:
            #             args.logger.info(f"index = {index}, global_step = {global_step}, episodic_return = {info['episode']['r']}")
            #             writer.add_scalar("charts/episodic_return", info["episode"]["r"] , global_step)
            #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            #             writer.add_scalar("charts/global_step", global_step)
            #             total_return += info["episode"]["r"]

            # v5 version
            if "episode" in infos:
                episode_return = np.sum(infos["episode"]["r"] * infos["episode"]["_r"]) / args.num_envs
                episode_length = np.sum(infos["episode"]["l"] * infos["episode"]["_l"]) / args.num_envs
                args.logger.info(f"global_step = {global_step}, episodic_return = {episode_return}, episodic_length = {episode_length}")
                writer.add_scalar("charts/episodic_return", episode_return, global_step)
                writer.add_scalar("charts/episodic_length", episode_length, global_step)
        
        # bootstrap value if not done
        # with torch.no_grad():
        #     next_value = agent.get_value(next_obs).reshape(1, -1)
        #     advantages = torch.zeros_like(rewards).to(args.device)
        #     lastgaelam = 0
        #     for t in reversed(range(args.num_steps)):
        #         if t == args.num_steps - 1:
        #             nextnonterminal = 1.0 - next_done
        #             nextvalues = next_value
        #         else:
        #             nextnonterminal = 1.0 - dones[t + 1]
        #             nextvalues = values[t + 1]
        #         delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        #         advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        #     returns = advantages + values
        # args.logger.info(f'reward is {total_reward}')
        # writer.add_scalar("charts/total_reward", total_reward, global_step)
        
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
        # b_logprobs shape is (args.num_steps * args.num_envs, args.sample_action_num)
        b_logprobs = logprobs.reshape((-1,) + (args.sample_action_num,))
        # b_actions shape is (args.num_steps * args.num_envs, args.sample_action_num, action_dim)
        b_actions = actions.reshape((-1,) + (args.sample_action_num,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_means = means.reshape(args.batch_size, -1)
        b_stds = stds.reshape(args.batch_size, -1)
        

        # 采样新旧策略的数据点用于绘制分布
        dist_sample_points_x = np.linspace(-2, 2, 500) 
        dist_sample_points = {}
        
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            ratio_clipfracs, ratio1_clipfracs, ratio2_clipfracs = 0.0, 0.0, 0.0
            min_ratio, max_ratio = 10.0, 0.0
            min_ratio1, max_ratio1 =  10.0, 0.0
            min_ratio2, max_ratio2 = 10.0, 0.0

            for start in range(0, args.batch_size, args.minibatch_size):

                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, new_mean_std = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                total_logratio = newlogprob - b_logprobs[mb_inds]
                logratio = total_logratio[:,0]
                # ratio = logratio.exp()

                # junweiluo：增加指标记录
                ratio1 = logratio.exp()
                if args.sample_action_num > 1:
                    ratio2 = torch.sum(total_logratio[:,1:], dim=1).exp()
                    ratio2 = torch.clamp(ratio2, 1 - args.clip_coef, 1 + args.clip_coef)
                else:
                    ratio2 = torch.ones_like(ratio1).detach()
                    # ratio2 = ratio1.detach()
                ratio = ratio1 * ratio2

                with torch.no_grad():
                    batch_index += 1
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio1 - 1) - logratio).mean()
                    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), batch_index)
                    writer.add_scalar("losses/approx_kl", approx_kl.item(),  batch_index)                
                    
                    # ADD(junweiluo) : 新增数据
                    total_old_approx_kl = (-ratio.log()).mean()
                    total_approx_kl = ((ratio - 1) - ratio.log()).mean()
                    writer.add_scalar("losses/total_old_approx_kl", old_approx_kl.item(),  batch_index)
                    writer.add_scalar("losses/total_approx_kl", approx_kl.item() , batch_index)
                    
                    
                    # use mean and std to calculate kl
                    new_mean = new_mean_std.loc
                    new_std = new_mean_std.scale
                    kl = compute_kld(b_means[mb_inds], b_stds[mb_inds], new_mean, new_std)
                    writer.add_scalar("losses/kl", kl.mean().detach().cpu().item() , batch_index)
                    

                    # junweiluo： 增加指标
                    # ==================== 优势函数ratio ==============================
                    # indice_larger_0 = np.where(b_returns[mb_inds].detach().cpu().numpy() > 0)[0]
                    # indice_smaller_0 = np.where(b_returns[mb_inds].detach().cpu().numpy() < 0)[0]
                    # adv_larger0_ratio1 = np.mean(ratio1.detach().cpu().numpy()[indice_larger_0])
                    # adv_larger0_ratio2 = np.mean(ratio2.detach().cpu().numpy()[indice_larger_0])
                    # adv_larger0_ratio = np.mean(ratio.detach().cpu().numpy()[indice_larger_0])
                    # adv_smaller0_ratio1 = np.mean(ratio1.detach().cpu().numpy()[indice_smaller_0])
                    # adv_smaller0_ratio2 = np.mean(ratio2.detach().cpu().numpy()[indice_smaller_0])
                    # adv_smaller0_ratio = np.mean(ratio.detach().cpu().numpy()[indice_smaller_0])
                    # writer.add_scalar('adv/adv_larger0_ratio1', adv_larger0_ratio1, batch_index)
                    # writer.add_scalar('adv/adv_larger0_ratio2', adv_larger0_ratio2, batch_index)
                    # writer.add_scalar('adv/adv_larger0_ratio', adv_larger0_ratio, batch_index)
                    # writer.add_scalar('adv/adv_smaller0_ratio1', adv_smaller0_ratio1, batch_index)
                    # writer.add_scalar('adv/adv_smaller0_ratio2', adv_smaller0_ratio2, batch_index)
                    # writer.add_scalar('adv/adv_smaller0_ratio', adv_smaller0_ratio, batch_index)

                    min_ratio, max_ratio = min(np.min(ratio.detach().cpu().numpy()), min_ratio), max(np.max(ratio.detach().cpu().numpy()), max_ratio)
                    min_ratio1, max_ratio1 = min(np.min(ratio1.detach().cpu().numpy()), min_ratio1), max(np.max(ratio1.detach().cpu().numpy()), max_ratio1)
                    min_ratio2, max_ratio2 = min(np.min(ratio2.detach().cpu().numpy()), min_ratio2), max(np.max(ratio2.detach().cpu().numpy()), max_ratio2)

                    
                    ratio_clipfracs += (torch.abs(ratio.detach().cpu() - 1.0) < args.clip_coef).float().sum()
                    ratio1_clipfracs += (torch.abs(ratio1.detach().cpu() - 1.0) < args.clip_coef).float().sum()
                    ratio2_clipfracs += (torch.abs(ratio2.detach().cpu() - 1.0) < args.clip_coef).float().sum()
                    

                    writer.add_scalar("imp_weight/ratio", np.mean(ratio.detach().cpu().numpy()), batch_index)
                    writer.add_scalar("imp_weight/ratio1", np.mean(ratio1.detach().cpu().numpy()), batch_index)
                    writer.add_scalar("imp_weight/ratio2", np.mean(ratio2.detach().cpu().numpy()), batch_index)
                    
                    # clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                
                    
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, (1 - args.clip_coef), (1 + args.clip_coef))
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                
                writer.add_scalar("losses/grad_norm", grad, batch_index)

                
            writer.add_scalar('imp_weight/min_ratio', min_ratio)
            writer.add_scalar('imp_weight/max_ratio', max_ratio)
            writer.add_scalar('imp_weight/min_ratio1', min_ratio1)
            writer.add_scalar('imp_weight/max_ratio1', max_ratio1)
            writer.add_scalar('imp_weight/min_ratio2', min_ratio2)
            writer.add_scalar('imp_weight/max_ratio2', max_ratio2)
            writer.add_scalar('losses/ratio_clifracs',  ratio_clipfracs / args.batch_size)
            writer.add_scalar('losses/ratio1_clifracs',  ratio1_clipfracs / args.batch_size)
            writer.add_scalar('losses/ratio2_clipfracs',  ratio2_clipfracs / args.batch_size)
            # writer.add_scalar("losses/act_std", new_std.detach().mean(), batch_index)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        
        
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # logger.info(f"SPS: {int(global_step / (time.time() - start_time))}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    envs.close()
    writer.close()
