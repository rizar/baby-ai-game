#!/usr/bin/env python3

import argparse
import numpy as np
from itertools import count
from collections import namedtuple
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
import time

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='MiniGrid-Empty-6x6-v0',
                    help='gym environment to load')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):

    def __init__(self, input_size, obs_high, num_actions):
        super(Policy, self).__init__()

        self.num_inputs = reduce(operator.mul, input_size, 1)
        self.obs_high = obs_high

        self.a_fc1 = nn.Linear(self.num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.a_fc3 = nn.Linear(64, num_actions)
        self.v_fc3 = nn.Linear(64, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, inputs):
        # Reshape the input so that it is one-dimensional
        inputs = inputs.view(-1, self.num_inputs)

        # Rescale observation values in [0,1]
        inputs = inputs / self.obs_high

        # Don't put a relu on the last layer, because we want to avoid
        # zero probabilities
        x = F.relu(self.a_fc1(inputs))
        x = F.relu(self.a_fc2(x))
        action_scores = self.a_fc3(x)  # F.tanh(self.a_fc3(x))
        action_probs = F.softmax(action_scores, dim=1)

        state_value = self.v_fc3(x)

        return action_probs, state_value


def select_action(model, state):
    state = torch.from_numpy(state).float().unsqueeze(0)

    action_probs, state_value = model(Variable(state))

    # print(
    #     'action_probs: ',
    #     action_probs.volatile,
    #     'state_value: ',
    #     state_value.volatile)

    # print(action_probs)

    m = Categorical(action_probs)
    action = m.sample()

    """
    # Maxime: hack to force exploration
    import random
    if random.random() < 0.05:
        action = random.randint(0, 3)
        action = torch.LongTensor([action])
        action = Variable(action)
    """

    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    return action.data[0]


def finish_episode(model, optimizer):
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []

    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)

    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / \
        (rewards.std() + np.finfo(np.float32).eps)

    for (log_prob, value), r in zip(saved_actions, rewards):
        # Advantage Estimate: A = R - V(s)
        reward = r - value.data[0, 0]
        policy_losses.append(-log_prob * reward)
        value_losses.append(
            F.smooth_l1_loss(
                value, Variable(
                    torch.Tensor(
                        [r]))))

    optimizer.zero_grad()
    loss = torch.cat(policy_losses).sum() + torch.cat(value_losses).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]
    return loss.cpu().data.numpy()[0]


def main():
    torch.manual_seed(args.seed)

    env = gym.make(args.env)
    env.seed(args.seed)

    # Create the policy/model based on the environment
    model = Policy(
        env.observation_space.shape,
        env.observation_space.high[0][0][0],
        env.action_space.n
    )

    modelSize = 0
    for p in model.parameters():
        pSize = reduce(operator.mul, p.size(), 1)
        modelSize += pSize
    print('Model size: %d' % modelSize)

    # Create the gradient descent optimizer
    #optimizer = optim.Adam(model.parameters(), lr=0.0005)
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.4)

    # This is used by the pytorch-a2c code
    # Initially promising, but then gets stuck failing
    #optimizer = optim.RMSprop(model.parameters(), lr=7e-4, eps=1e-5, alpha=0.99)

    totalFrames = 0
    runningReward = 0

    for i_episode in count(1):

        state = env.reset()

        for t in range(1, 10000):
            action = select_action(model, state)

            state, reward, done, _ = env.step(action)

            model.rewards.append(reward)

            if args.render:
                env.render()

            if done:

                break

        sumReward = sum(model.rewards)
        if sumReward > 0:
            print('SUCCESS')
            time.sleep(2)

        totalFrames += t
        runningReward = runningReward * 0.99 + sumReward * 0.01

        loss = finish_episode(model, optimizer)

        if i_episode % args.log_interval == 0:
            print(
                'Episode {}, l: {} frames: {}, running reward: {:.2f}'.format(
                    i_episode,
                    loss,
                    totalFrames,
                    runningReward))

if __name__ == '__main__':
    main()
