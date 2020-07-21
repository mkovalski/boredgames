#!/usr/bin/env python

from bored_games.envs.quoridor import Quoridor
from bored_games.envs.quoridor.models import QuoridorDQN
from bored_games.algos import DQNAgent
from bored_games.utils import PrioritizedReplayBuffer as PER
import torch
import os
import copy

exp_dir = 'test_per'

# Setup environment
env = Quoridor(random_move_prob = True)
action_shape, state_shape = env.action_shape(), env.state_shape()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 # Create a model
model = QuoridorDQN(board_shape = state_shape[0],
                    tile_shape = state_shape[1],
                    output_shape = action_shape).to(device)
print(model)

target_model = QuoridorDQN(board_shape = state_shape[0],
                    tile_shape = state_shape[1],
                    output_shape = action_shape).to(device)

# Create a new replay buffer and populate it
rb = PER(N = 100000, path = 'rbs/rb_1e6.pkl')
#rb.populate(env)
#rb.save(exp_dir)

# DQN
dqn = DQNAgent(env = env,
               model = model,
               target_model = target_model,
               exp_dir = exp_dir,
               target_update = 20,
               batch_size = 512)

dqn.train(replay_buffer = rb,
          episodes = 10000,
          eval_eps = 50)
