# -*- encoding: utf-8 -*-
'''
@File       :atari_trainer.py
@Description:
@Date       :2025/03/26 19:01:06
@Author     :junweiluo
@Version    :python
'''
import time
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from .base_trainer import BaseTrainer


class AtariTrainer(BaseTrainer):
    def __init__(self, args=None, agent=None, optimizer=None, writer=None):
        super().__init__(args, agent, optimizer, writer)
        self.num_alter_logprobs = self.args.discrete_action_space_n - 1 if self.args.algo == 'appo-all' else 1
        
    def reshape_(self, buffer):
        obs, actions, log_probs, advantages, returns, values, old_logits  = buffer
        # TODO(weijun): 添加observation_shape和discrete_action_space_n到args中
        b_obs = obs.reshape(-1, self.args.observation_shape)
        b_actions = actions.reshape(-1)
        b_log_probs = log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_values = values.reshape(-1)
        b_old_logits = old_logits.reshape((-1, self.args.discrete_action_space_n))
        
        return b_obs, b_actions, b_log_probs, b_returns, b_advantages, b_values, b_old_logits
    
    def _appo_compute_ratio_family(self, new_log_prob, mb_log_probs, new_logits, mb_old_logits, mb_actions):
        log_probs = new_log_prob - mb_log_probs
        ratio1 = log_probs.exp()
        mask_ = torch.ones_like(mb_old_logits)
        mask_[torch.arange(mb_old_logits.shape[0]), mb_actions] = 0.0
        selected_indice = torch.multinomial(mask_, num_samples = self.num_alter_logprobs).squeeze()
        selected_indice_0 = torch.arange(mb_old_logits.shape[0])
        if self.num_alter_logprobs > 1:
            selected_indice_0 = selected_indice_0.unsqueeze(1).expand(-1, self.num_alter_logprobs)
        old_logprobs = mb_old_logits[selected_indice_0, selected_indice]
        new_logprobs = new_logits[selected_indice_0, selected_indice]
        log_ratio2 = new_logprobs - old_logprobs
        if len(log_ratio2.shape) == 1:
            log_ratio2 = log_ratio2.unsqueeze(1)
        raw_ratio2 = torch.sum(log_ratio2, dim=1).exp()
        # raw_ratio2 = torch.pow(raw_ratio2,  1 / num_alter_actions)
        ratio2 = torch.clamp(raw_ratio2, 1 - self.args.epsilon_2, 1 + self.args.epsilon_2)
        ratio = ratio1 * ratio2
        return ratio, ratio1, raw_ratio2
    
    def _ppoclip_compute_ratio_family(self, new_log_prob, mb_log_probs, *args):
        log_ratio1 = new_log_prob - mb_log_probs
        ratio1 = log_ratio1.exp()
        ratio2 = torch.ones_like(ratio1).detach()
        return ratio1, ratio1, ratio2

        
    def map_compute_ratio_func(self, new_log_prob, mb_log_probs, new_logits, mb_old_logits, mb_actions):
        func_dict = {
            "ppo-clip" : self._ppoclip_compute_ratio_family,
            "ppo-all" : self._appo_compute_ratio_family,
            "ppo-two" : self._appo_compute_ratio_family,
        }

        func_ = func_dict.get(self.args.algo, self._not_implemented)
        return func_(new_log_prob, mb_log_probs, new_logits, mb_old_logits, mb_actions)

    def compute_value_loss(self, mb_values, mb_returns, newvalue):
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
        #     v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        return v_loss
    
    def compute_policy_loss(self, ratio, mb_advantages):
        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        return pg_loss
    
        
    def train(self, global_step, buffer):
        b_inds = np.arange(self.args.batch_size)
        b_obs, b_actions, b_log_probs, b_returns, b_advantages, b_values, b_old_logits = self.reshape_(buffer = buffer)

        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            ratio_clipfracs, ratio1_clipfracs, ratio2_clipfracs = 0.0, 0.0, 0.0
            min_ratio, max_ratio = 10.0, 0.0
            min_ratio1, max_ratio1 =  10.0, 0.0
            min_ratio2, max_ratio2 = 10.0, 0.0

            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                # The latest outputs of the policy network and value network
                _, new_log_prob, new_entropy, new_value, new_logits = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                # Probability ratio
                ratios ,ratio1, ratio2 = self.map_compute_ratio_func(
                    new_log_prob = new_log_prob, 
                    mb_log_probs = b_log_probs[mb_inds], 
                    new_logits = new_logits, 
                    mb_old_logits = b_old_logits[mb_inds], 
                    mb_actions = b_actions[mb_inds]
                )

                # Advantage normalization
                mb_advantages = b_advantages[mb_inds]
                # 暂时取消掉这个if判断
                # if self.args.advantage_normalization:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss = self.compute_policy_loss(ratios, mb_advantages)

                # Value loss
                v_loss = self.compute_value_loss(mb_returns = b_returns[mb_inds], new_value = new_value, mb_values = b_values[mb_inds])
                # Policy entropy
                entropy_loss = new_entropy.mean()
                # Total loss
                loss = pg_loss + v_loss * self.args.vf_coef - entropy_loss * self.args.ent_coef

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    logratio1 = ratio1.detach().log()
                    old_approx_kl = (-logratio1).mean()
                    approx_kl = ((ratio1 - 1) - logratio1).mean()

                    # ratios family
                    min_ratio, max_ratio = min(np.min(ratios.detach().cpu().numpy()), min_ratio), max(np.max(ratios.detach().cpu().numpy()), max_ratio)
                    min_ratio1, max_ratio1 = min(np.min(ratio1.detach().cpu().numpy()), min_ratio1), max(np.max(ratio1.detach().cpu().numpy()), max_ratio1)
                    min_ratio2, max_ratio2 = min(np.min(ratio2.detach().cpu().numpy()), min_ratio2), max(np.max(ratio2.detach().cpu().numpy()), max_ratio2)
                    ratio_clipfracs += (torch.abs(ratios.detach().cpu() - 1.0) < self.args.clip_coef).float().sum()
                    ratio1_clipfracs += (torch.abs(ratio1.detach().cpu() - 1.0) < self.args.clip_coef).float().sum()
                    ratio2_clipfracs += (torch.abs(ratio2.detach().cpu() - 1.0) < self.args.clip_coef).float().sum()                    
                    mini_dict_ = {
                        "losses/old_approx_kl": old_approx_kl.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "imp_weight/ratio": np.mean(ratios.detach().cpu().numpy()),
                        "imp_weight/ratio1": np.mean(ratio1.detach().cpu().numpy()),
                        "imp_weight/ratio2": np.mean(ratio2.detach().cpu().numpy()),
                    }
                    self.log_minibatch(mini_dict_) 
                
                min_ratio, max_ratio = np.min(ratios.detach().cpu().numpy()), np.max(ratios.detach().cpu().numpy())
                min_ratio1, max_ratio1 = np.min(ratio1.detach().cpu().numpy()), np.max(ratio1.detach().cpu().numpy())
                min_ratio2, max_ratio2 = np.min(ratio2.detach().cpu().numpy()), np.max(ratio2.detach().cpu().numpy())
                
                if start == 0 and epoch !=0 :
                    # 计算KL
                    kl_divs = torch.distributions.kl.kl_divergence(
                        Categorical(logits=b_old_logits[mb_inds]), 
                        Categorical(logits=new_logits),
                    ).mean()
                    self.writer.add_scalar('losses/kl_div', kl_divs.item())
                
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
        

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        dict_ = {
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - self.start_time)),
        }
        
        self.log_(dict_)

