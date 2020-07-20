#!/usr/bin/env python

from bored_games.envs.quoridor import Quoridor
from bored_games.envs.quoridor.models import QuoridorDQN
from bored_games.algos import DQNAgent
from bored_games.utils import ReplayBuffer

exp_dir = 'test'

# Setup environment
env = Quoridor()
action_shape, state_shape = env.action_shape(), env.state_shape()

 # Create a model
model = QuoridorDQN(board_shape = state_shape[0],
                    tile_shape = state_shape[1],
                    output_shape = action_shape).to('cpu')

# Create a new replay buffer and populate it
rb = ReplayBuffer(N = int(1e5))
rb.populate(env)
rb.save(exp_dir)

# DQN
dqn = DQNAgent(env = env,
               model = model,
               exp_dir = exp_dir,
               target_update = 10)

dqn.train(replay_buffer = rb,
          eval_eps = 20)
