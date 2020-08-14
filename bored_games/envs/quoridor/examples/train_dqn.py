#!/usr/bin/env python

from bored_games.envs.quoridor import Quoridor
from bored_games.envs.quoridor.models import QuoridorDQN
from bored_games.algos import DQN, DQNAgent
from bored_games.utils import ReplayBuffer as RB
from bored_games.utils import PrioritizedReplayBuffer as PRB
import torch
import os
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type = str, required = True)
parser.add_argument('--buffer', type = str, default = None)
parser.add_argument('--buffer-size', type = int, default = 1000)
parser.add_argument('--prioritized', action = 'store_true')
parser.add_argument('--batch_size', type = int, default = 32)

args = parser.parse_args()

# Setup environment
env = Quoridor(set_move_prob = True)
action_shape, state_shape = env.action_shape(), env.state_shape()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 # Create a model
model = QuoridorDQN(board_shape = state_shape[0],
                    tile_shape = state_shape[1],
                    output_shape = action_shape).to(device)

print(model)

# Create an agent
agent = DQNAgent(model = model)

# Create a new replay buffer and populate it
buffer_type = RB if not args.prioritized else PRB

if args.buffer:
    rb = buffer_type(N = None, path = args.buffer)
else:
    rb = buffer_type(N = args.buffer_size)
    rb.populate(env)
    rb.save(args.exp)
        
# DQN
dqn = DQN(env = env,
          agent = agent,
          exp_dir = args.exp,
          target_update = 50,
          batch_size = args.batch_size)


dqn.train(replay_buffer = rb,
          episodes = 100000,
          eval_eps = 50,
          n_eval_games = 50)
