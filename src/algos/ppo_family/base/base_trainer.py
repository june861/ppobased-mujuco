# -*- encoding: utf-8 -*-
'''
@File       :base_trainer.py
@Description:
@Date       :2025/03/31 10:40:11
@Author     :junweiluo
@Version    :python
'''
import torch
import sys
import inspect

class BaseTrainer(object):
    def __init__(self, args, agent, optimizer):
        
        # common parameters
        self.logger = args.logger
        self.max_grad_norm = args.max_grad_norm
        self.update_epochs = args.update_epochs
        self.norm_adv = args.norm_adv
        self.clip_vloss = args.clip_vloss
        self.vf_coef = args.vf_coef
        self.target_kl = args.target_kl
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.sample_action_num = args.sample_action_num
        self.ent_coef = args.ent_coef
        
        self.agent = agent
        self.optimizer = optimizer
        self.batch_index = 0

    def log_dict_(self, **kwargs):
        mini_dict_ = {}
        for key, value in kwargs.items():
            if "imp_weights" in key:
                log_key = key.replace("_","$",1).replace("_","/",1).replace("$","_")
            else:
                log_key = key.replace("_","/",1)
            mini_dict_[log_key] = value
        return mini_dict_

    def compute_value_loss(self,  mb_returns, mb_values, newvalue):
        # Value loss
        newvalue = newvalue.view(-1)
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + torch.clamp(
                newvalue - mb_values,
                -self.clip_coef,
                self.clip_coef,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
        
        return v_loss  

    def compute_ratios_family(self):
        self.logger.error("function: compute_ratios_family had not yet implement!")
        sys.exit({"ExitCode": 1, "ERROR": "NotImplementedError"})
    
    def compute_policy_loss(self):
        self.logger.error("function: compute_policy_loss had not yet implement!")
        sys.exit({"ExitCode": 1, "ERROR": "NotImplementedError"})
    
    def ppo_update(self):
        self.logger.error("function: ppo_update had not yet implement!")
        sys.exit({"ExitCode": 1, "ERROR": "NotImplementedError"})
    
    def update_one_episode(self):
        self.logger.error("function: update_one_episode had not yet implement!")
        sys.exit({"ExitCode": 1, "ERROR": "NotImplementedError"})
    
    def _not_implemented(self, *args):
        raise NotImplementedError()
    
    
    
    
        