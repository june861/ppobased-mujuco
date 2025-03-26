# -*- encoding: utf-8 -*-
'''
@File    :   base_trainer.py
@Time    :   2025/03/25 22:33:09
@Author  :   junewluo 
'''

import torch
import torch.nn as nn
import numpy as np

class BaseTranier(object):
    def __init__(self, args = None, agent = None, optimizer = None, writer = None):
        self.args = args
        self.agent = agent
        self.optimizer = optimizer
        self.writer = writer
        self.batch_index = 0
    
    def train(self):
        raise NotImplementedError(f"BaseTranier.train hasn't implemented!")

    def update(self):
        raise NotImplementedError(f"BaseTranier.update hasn't implemented!")
    