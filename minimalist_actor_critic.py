#!/usr/bin/env python3

import argparse
import numpy as np
from itertools import count
from collections import namedtuple
import operator
from functools import reduce
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from model import Minimalist_ActorCritic_Agent, Checkpoint

import gym
import gym_minigrid
import gym_minigrid.minigrid as minigrid

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed (default: 1234).')
parser.add_argument('--batch-size', type=int, default=32,
                    help='mini-batch size')
parser.add_argument('--env', type=str, default='MiniGrid-Empty-6x6-v0',
                    help='gym environment to load')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help="Coefficient associated with the critic loss.")
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help="Coefficient associated with the policy entropy.")
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='max value of the gradient norm (default: 50)')
parser.add_argument('--max-iters', type=int, default=1000000,
                    help='maximum training iterations (default: 1000000)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number max of forward steps in AC (default: 20).' +
                    ' Use 0 to go through complete episodes before updating.')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--num-checkpoints', default=5,
                    type=int, help="Number of check points (default: 5).")
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--expt-dir', type=str, default='./experiment',
                    help='Path to experiment directory. If load_checkpoint ' +
                    'is True, then path to checkpoint directory has to be ' +
                    'provided')
parser.add_argument('--load-checkpoint', action='store', help='The name of ' +
                    'the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', default=False,
                    help='Indicates if training has to be resumed from ' +
                    'the latest checkpoint')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.num_steps == 0:
    args.num_steps = None
args.checkpoint_every = int(args.max_iters / (args.num_checkpoints + 1)) + 1


def main():

    env = gym.make(args.env)
    # env = gym.make('MiniGrid-Fetch-8x8-v0')

    # Build the vocabulary dictionnary of the agent
    vocab_words = [
        'get', 'a', 'go', 'fetch', 'you', 'must', 'key', 'ball', 'red',
        'green', 'blue', 'purple', 'yellow', 'grey', 'to', 'the', 'square',
        'circle', 'empty', 'wall', 'door', 'locked_door', 'box', 'goal'
    ]
    specials = ['<pad>', '<unk>', '<eos>']
    vocab_words.sort()

    vocab_dict = {}
    vocab_dict.update({tok: i for i, tok in enumerate(specials)})
    vocab_dict.update({tok: i + len(specials)
                       for i, tok in enumerate(vocab_words)})

    def process_string(text):
        if (text is None) or (text == ''):
            return [vocab_dict['<eos>']]
        else:
            words = text.strip().lower().split()
            word_ids = []
            for i in range(len(words)):
                key = words[i].strip()
                if key == '':
                    continue
                elif key in vocab_dict:
                    word_ids.append(vocab_dict[key])
                else:
                    word_ids.append(vocab_dict['<unk>'])
            word_ids.append(vocab_dict['<eos>'])
            return word_ids

    # Create the model arguments (No need in test mode when loading a saved
    # model)
    if args.load_checkpoint is not None:
        model_args = None
    else:
        vocab_size = 10
        if not (vocab_dict is None or len(vocab_dict) == 0):
            vocab_size = len(vocab_dict)
        model_args = {
            'img_shape': [3, 7, 7],
            'action_space': env.action_space.n,
            'channels': [3, 16, 32],
            'kernels': [4, 3],
            'strides': None,
            'langmod': False,
            'vocab_size': vocab_size,
            'embed_dim': 64,
            'langmod_hidden_size': 64,
            'actmod_hidden_size': 128,
            'policy_hidden_size': 64,
            'langmod_pred': False,
        }

    # Create the RL agent based on the environment
    agent = Minimalist_ActorCritic_Agent(
        env, args, device=None,
        expt_dir=args.expt_dir,
        checkpoint_every=args.checkpoint_every,
        log_interval=args.log_interval,
        input_scale=env.observation_space.high[0][0][0],
        reward_scale=1000,
        pad_sym='<pad>',
        process_string_func=process_string,
        model_args=model_args,
    )

    if args.load_checkpoint is not None:
        print(
            "loading checkpoint from {}".format(
                os.path.join(
                    args.expt_dir,
                    Checkpoint.CHECKPOINT_DIR_NAME,
                    args.load_checkpoint)
            )
        )
        checkpoint_path = os.path.join(
            args.expt_dir,
            Checkpoint.CHECKPOINT_DIR_NAME,
            args.load_checkpoint)
        checkpoint = Checkpoint.load(checkpoint_path)
        model = checkpoint.model
        agent.model = model

        print("Finishing Loading")

        # test the agent
        agent.test(max_iters=args.max_iters, render=args.render)

    else:

        # train the agent
        agent.train(
            max_iters=args.max_iters,
            max_fwd_steps=args.num_steps,
            resume=args.resume,
            path='.'
        )


if __name__ == '__main__':
    main()
