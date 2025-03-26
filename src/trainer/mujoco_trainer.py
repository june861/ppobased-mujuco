# -*- encoding: utf-8 -*-
'''
@File    :   trainer.py
@Time    :   2025/03/25 22:32:17
@Author  :   junewluo 
'''


import torch
import torch.nn as nn
import time
import numpy as np
from utils import compute_kld
from .base_trainer import BaseTrainer

class MujocoTrainer(BaseTrainer):
    def __init__(self, args = None, agent = None, optimizer = None, writer = None):
        super().__init__(args, agent, optimizer, writer)
    
    
    def reshape_(self, buffer):
        
        obs, logprobs, actions, advantages, returns, values, means, stds = buffer
        # flatten the batch
        b_obs = obs.reshape((-1,) + self.args.single_observation_space.shape)
        # b_logprobs shape is (args.num_steps * args.num_envs, args.sample_action_num)
        b_logprobs = logprobs.reshape((-1,) + (self.args.sample_action_num,))
        # b_actions shape is (args.num_steps * args.num_envs, args.sample_action_num, action_dim)
        b_actions = actions.reshape((-1,) + (self.args.sample_action_num,) + self.args.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_means = means.reshape(self.args.batch_size, -1)
        b_stds = stds.reshape(self.args.batch_size, -1)
        
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_means, b_stds
    
    
    def compute_ratios_cluster(self, newlogprob, mb_logprobs):
        """ return ratios cluster

        Args:
            newlogprob (_type_): _description_
            mb_logprobs (_type_): _description_
        """

        total_logratio = newlogprob - mb_logprobs
        logratio1 = total_logratio[:,0]
        ratio1 = logratio1.exp()
        if self.args.sample_action_num > 1:
            ratio2 = torch.sum(total_logratio[:,1:], dim=1).exp()
            ratio2 = torch.pow(ratio2, 1 / self.args.sample_action_num)
            ratio2 = torch.clamp(ratio2, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
        else:
            ratio2 = torch.ones_like(ratio1).detach()

        ratio = ratio1 * ratio2
        dict_ = {
            'ratio' : ratio,
            'ratio1' : ratio1,
            'ratio2': ratio2,
        }
        
        return dict_


    
    def compute_value_loss(self, mb_returns, mb_values, newvalue):
        # Value loss
        newvalue = newvalue.view(-1)
        # if self.args.clip_vloss:
        v_loss_unclipped = (newvalue - mb_returns) ** 2
        v_clipped = mb_values + torch.clamp(
            newvalue - mb_values,
            -self.args.clip_coef,
            self.args.clip_coef,
        )
        v_loss_clipped = (v_clipped - mb_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
        
        # else:
        #     v_loss = 0.5 * ((newvalue -mb_returns) ** 2).mean()
        
        return v_loss

    def compute_policy_loss(self, mb_advantages, ratio):
        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, (1 - self.args.clip_coef), (1 + self.args.clip_coef))
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        return pg_loss
    
    
    def train(self, buffer, global_step):
        
        b_inds = np.arange(self.args.batch_size)
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_means, b_stds = self.reshape_(buffer = buffer)
        start_time = time.time()
        
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            ratio_clipfracs, ratio1_clipfracs, ratio2_clipfracs = 0.0, 0.0, 0.0
            min_ratio, max_ratio = 10.0, 0.0
            min_ratio1, max_ratio1 =  10.0, 0.0
            min_ratio2, max_ratio2 = 10.0, 0.0

            for start in range(0, self.args.batch_size, self.args.minibatch_size):

                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, new_mean_std = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                ratio_family = self.compute_ratios_cluster(newlogprob, b_logprobs[mb_inds])
                ratio, ratio1, ratio2 =  ratio_family['ratio'], ratio_family['ratio1'], ratio_family['ratio2']
                logratio1 = ratio1.log()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio1).mean()
                    approx_kl = ((ratio1 - 1) - logratio1).mean()
                    # use mean and std to calculate kl
                    new_mean = new_mean_std.loc
                    new_std = new_mean_std.scale
                    kl = compute_kld(b_means[mb_inds], b_stds[mb_inds], new_mean, new_std)
                    # ratios family
                    min_ratio, max_ratio = min(np.min(ratio.detach().cpu().numpy()), min_ratio), max(np.max(ratio.detach().cpu().numpy()), max_ratio)
                    min_ratio1, max_ratio1 = min(np.min(ratio1.detach().cpu().numpy()), min_ratio1), max(np.max(ratio1.detach().cpu().numpy()), max_ratio1)
                    min_ratio2, max_ratio2 = min(np.min(ratio2.detach().cpu().numpy()), min_ratio2), max(np.max(ratio2.detach().cpu().numpy()), max_ratio2)
                    ratio_clipfracs += (torch.abs(ratio.detach().cpu() - 1.0) < self.args.clip_coef).float().sum()
                    ratio1_clipfracs += (torch.abs(ratio1.detach().cpu() - 1.0) < self.args.clip_coef).float().sum()
                    ratio2_clipfracs += (torch.abs(ratio2.detach().cpu() - 1.0) < self.args.clip_coef).float().sum()                    
                    mini_dict_ = {
                        "losses/old_approx_kl": old_approx_kl.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "losses/kl": kl.mean().detach().cpu().item() ,
                        "imp_weight/ratio": np.mean(ratio.detach().cpu().numpy()),
                        "imp_weight/ratio1": np.mean(ratio1.detach().cpu().numpy()),
                        "imp_weight/ratio2": np.mean(ratio2.detach().cpu().numpy()),
                    }
                    self.log_minibatch(mini_dict_)     

                mb_advantages = b_advantages[mb_inds]
                # if self.args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss = self.compute_policy_loss(mb_advantages = mb_advantages, ratio = ratio)
                # value loss
                v_loss = self.compute_value_loss(mb_returns = b_returns[mb_inds], mb_values = b_values[mb_inds], newvalue = newvalue)
                # entropy loss
                entropy_loss = entropy.mean()
                # total loss
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef
                # param update
                self.optimizer.zero_grad()
                loss.backward()
                grad = nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.batch_index += 1

            dict_ = {
                "losses/grad_norm" : grad,    
                'imp_weight/min_ratio' : min_ratio,
                'imp_weight/max_ratio' : max_ratio,
                'imp_weight/min_ratio1' : min_ratio1,
                'imp_weight/max_ratio1' : max_ratio1,
                'imp_weight/min_ratio2' : min_ratio2,
                'imp_weight/max_ratio2' : max_ratio2,
                'losses/ratio_clifracs' :  ratio_clipfracs / self.args.batch_size,
                'losses/ratio1_clifracs' :  ratio1_clipfracs / self.args.batch_size,
                'losses/ratio2_clipfracs' :  ratio2_clipfracs / self.args.batch_size,
            }
            
            self.log_(dict_)
            

            # 不涉及这个
            # if self.args.target_kl is not None and approx_kl > self.args.target_kl:
            #     break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        dict_ = {
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - start_time)),
        }
        
        self.log_(dict_)
        
