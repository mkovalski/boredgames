# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from quoridor import Quoridor
from dqn import DQNAgent
from nn import NN
import os
import time
import multiprocessing as mp
import argparse
from tqdm import tqdm
import pickle

def populate_rb(N):
    env = Quoridor()
    board_shape, tile_shape = env.state_shape()
    action_shape = env.action_shape()

    agent = DQNAgent(board_shape, tile_shape, action_shape, maxlen = N)
    done = False

    # Fill up the replay buffer
    done = True
    
    print("Populating replay buffer...")
    for i in tqdm(range(agent.memory.maxlen)):
        if done:
            agent.reset()
            state = env.reset(player = 1,
                              move_prob = [np.random.random(), np.random.random()])
            done = False
        
        action = agent.act(state, env)
        
        next_state, reward, done, _ = env.step(1, action)

        agent.memorize(state, action, reward, next_state, done)
        state = next_state
    
    with open('rb.pkl', 'wb') as myFile:
        pickle.dump(agent.memory, myFile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--N', type = int, default = 100000)

    args = parser.parse_args()
    populate_rb(args.N) 
