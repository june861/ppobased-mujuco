# -*- encoding: utf-8 -*-
'''
@File    :   base_trainer.py
@Time    :   2025/03/25 22:33:09
@Author  :   junewluo 
'''
import time
import torch
import torch.nn as nn
import numpy as np

class BaseTrainer(object):
    def __init__(self, args = None, agent = None, optimizer = None, writer = None):
        self.args = args
        self.agent = agent
        self.optimizer = optimizer
        self.writer = writer
        self.batch_index = 0
        self.start_time = time.time()
    
    def train(self):
        raise NotImplementedError(f"BaseTranier.train hasn't implemented!")

    def update(self):
        raise NotImplementedError(f"BaseTranier.update hasn't implemented!")
    
    def _not_implemented(self):
        self.args.logger.error(f'Not Implemented for such func')
        raise NotImplementedError

    def log_minibatch(self, dict_):
        for tag, value in dict_.items():
            self.writer.add_scalar(tag, value, self.batch_index)
    
    def log_(self, dict_):
        for tag, value in dict_.items():
            self.writer.add_scalar(tag, value)