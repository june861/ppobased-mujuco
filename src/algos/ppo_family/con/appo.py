# -*- encoding: utf-8 -*-
'''
@File    :   appo.py
@Time    :   2025/03/29 16:25:25
@Author  :   junewluo 
'''
import torch

class APPO(object):
    def __init__(self):
        pass

    def update(self, ):
        pass
    

    def compute_ratio_family(self, newlogprob, mb_logprobs):
        # ratio1
        total_logratio = newlogprob - mb_logprobs
        logratio1 = total_logratio[:,0]
        ratio1 = logratio1.exp()

        # ratio2
        log_ratio2 = total_logratio[:,1:]
        ratio2 = log_ratio2.exp()
        ratio2 = torch.sum(ratio2, dim = 1) / (self.args.sample_action_num - 1)

        return ratio1, ratio2

    def compute_policy_loss(self, ratio1, ratio2, mb_advantages):
        """ compute appo policy loss

        Args:
            ratio1 (_torch.tensor_):  
            ratio2 (_torch.tensor_): _description_
            mb_advantages (_torch.tensor_): _description_

        Returns:
            _torch.tensor_: policy loss
        """

        # compute pg_loss_1
        pg_loss1 = -mb_advantages * ratio1
        pg_loss2 = -mb_advantages * torch.clamp(ratio1, (1 - self.args.clip_coef), (1 + self.args.clip_coef))
        pg_loss_1 = torch.max(pg_loss1, pg_loss2).mean() 
        
        # compute pg_loss_2
        ratio2_norm = ratio2 / ratio2.mean()
        pg_loss_2 =  (self.args.decay_beta * 0.5 * torch.abs(mb_advantages.detach()) * (ratio2_norm - 1)**2).mean()

        # compute pg_loss
        pg_loss = pg_loss_1 + pg_loss_2.mean()

        return pg_loss, pg_loss_1, pg_loss_2

