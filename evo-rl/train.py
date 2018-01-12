#!/usr/bin/env python3

import os
import time
import argparse
import operator
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
import gym_minigrid

##############################################################################

class Policy(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        num_inputs = reduce(operator.mul, obs_space.shape, 1)
        num_actions = action_space.n

        #self.a_fc1 = nn.Linear(num_inputs, 64)
        #self.a_fc2 = nn.Linear(64, 64)
        # TODO: softmax to pick actions

        #self.v_fc1 = nn.Linear(num_inputs, 64)
        #self.v_fc2 = nn.Linear(64, 64)





    def forward(self, obs):
        pass
        #return vs, va





##############################################################################

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--env-name', default='MiniGrid-Empty-6x6-v0', help='gym environment to use')
args = parser.parse_args()

env = gym.make(args.env_name)










env.reset()
env.seed()











#
