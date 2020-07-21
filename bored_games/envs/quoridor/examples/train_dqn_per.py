#!/usr/bin/env python

from bored_games.envs.quoridor import Quoridor
from bored_games.envs.quoridor.models import QuoridorDQN
from bored_games.algos import DQNAgent
from bored_games.utils import PrioritizedReplayBuffer as PER
import os

exp_dir = 'test_per'

# Setup environment
env = Quoridor(random_move_prob = True)
action_shape, state_shape = env.action_shape(), env.state_shape()

 # Create a model
model = QuoridorDQN(board_shape = state_shape[0],
                    tile_shape = state_shape[1],
                    output_shape = action_shape).to('cpu')

# Create a new replay buffer and populate it
rb = PER(N = 100000, path = os.path.join(exp_dir, 'rb.pkl'))
#rb.populate(env)
#rb.save(exp_dir)

# DQN
dqn = DQNAgent(env = env,
               model = model,
               exp_dir = exp_dir,
               target_update = 20,
               batch_size = 512)

dqn.train(replay_buffer = rb,
          episodes = 10000,
          eval_eps = 50)
