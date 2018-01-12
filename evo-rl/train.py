#!/usr/bin/env python3

"""
Implementation of evolutionary training for RL inspired by OpenAI's
evolution strategies paper:
https://blog.openai.com/evolution-strategies/
https://arxiv.org/pdf/1703.03864.pdf

Phases:
1) Stochastically perturbing the parameters of the policy and
   evaluating the resulting parameters by running an episode in
   the environment
2) Combining the results of these episodes, calculating a
   stochastic gradient estimate, and updating the parameters
"""

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
from torch.distributions import Categorical

import gym
import gym_minigrid

##############################################################################

def weightInit(m):
    className = m.__class__.__name__

    # TODO: check how paper does weight init
    if className.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1)
        m.bias.data.fill_(0)
        return
    elif className == 'Policy':
        return

    assert False, className

class Policy(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        num_inputs = reduce(operator.mul, obs_space.shape, 1)
        num_actions = action_space.n

        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, obs):
        inputs = obs.view(-1)

        x = F.tanh(self.fc1(inputs))
        x = F.tanh(self.fc2(x))

        # Use a softmax to produce pseudo-probabilites for actions
        action_scores = self.fc3(x)
        action_probs = F.softmax(action_scores, dim=0)

        return action_probs

def selectAction(policy, obs):
    obs = torch.from_numpy(obs).float().unsqueeze(0)

    action_probs = policy(Variable(obs))

    m = Categorical(action_probs)
    action = m.sample()
    return action.data[0]

##############################################################################

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--env-name',
    default='MiniGrid-Empty-6x6-v0',
    help='gym environment to use'
)
parser.add_argument(
    '--lr',
    default=0.0001,
    help='learning rate'
)
# TODO:
# sigma, noise standard deviation for mutations

args = parser.parse_args()

# Create an instance of the environment
env = gym.make(args.env_name)







policy = Policy(env.observation_space, env.action_space)
policy.apply(weightInit)

#print(list(policy.parameters()))
modelSize = 0
for p in policy.parameters():
    pSize = reduce(operator.mul, p.size(), 1)
    modelSize += pSize
print(str(policy))
print('Total model size: %d' % modelSize)

env.seed(0)
obs = env.reset()

for i in range(0, 10):
    action = selectAction(policy, obs)
    print(action)


# Param updates weighed by returns of each noise vector







#
