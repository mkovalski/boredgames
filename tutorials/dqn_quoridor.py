#!/usr/bin/env python

from bored_games import Quoridor, TDQNAgent

env = Quoridor(player = 1)
agent = TDQNAgent(env,
                 exp_dir = 'quoridor_bs_32',
                 buffer_size = 5000,
                 batch_size = 512,
                 priority = True,
                 decay_steps = 10000)

agent.train(episodes = 1000) 
