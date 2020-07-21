#!/usr/bin/env python

from bored_games import Quoridor, TDQNAgent

env = Quoridor(player = 1)
agent = TDQNAgent(env,
                 exp_dir = 'quoridor_bs_16_priority',
                 buffer_size = 5000,
                 batch_size = 16,
                 priority = True,
                 decay_steps = 10000,
                 n_target_updates = 1e5)

agent.train(episodes = 1000) 
