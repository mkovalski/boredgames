#!/usr/bin/env python

from bored_games import Quoridor, DQNAgent

env = Quoridor(player = 1)
agent = DQNAgent(env,
                 exp_dir = 'test_dqn_quoridor',
                 buffer_size = 100000,
                 batch_size = 32,
                 decay_steps = 10000)

agent.train(episodes = 10000, rb = 'test_dqn_quoridor/rb.pkl') 
