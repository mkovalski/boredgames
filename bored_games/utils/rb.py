# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from tqdm import tqdm 
import pickle
import os

class ReplayBuffer:
    KEYS = ['state', 'valid_actions', 'action', 'reward',
               'next_state', 'next_valid_actions', 'done', 'idx']

    priority = False

    def __init__(self, N = 1000, path = None):
        self.memory = deque(maxlen = N)
    
    def append(self, batch):
        self.memory.append(batch)

    def populate(self, env, N = None):
        if N is None:
            N = self.memory.maxlen
        if N > self.memory.maxlen:
            print("Cannot populate more memory than size of buffer, will populate {}".format(self.memory.maxlen))
            N = self.memory.maxlen

        done = True
        print("Populating {} items...".format(N))

        for i in tqdm(range(N)):
            if done:
                state, valid_actions = env.reset()

            action = env.sample()
            next_state, next_valid_actions, reward, done, _ = env.step(action)

            batch = dict(state = state,
                         valid_actions = valid_actions,
                         action = action,
                         next_state = next_state,
                         next_valid_actions = next_valid_actions,
                         reward = reward,
                         done = done)

            self.memory.append(batch)

            state = next_state
            valid_actions = next_valid_actions

    @property
    def maxlen(self):
        return self.memory.maxlen

    def get_batch(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        minibatch = self.organize_batch(minibatch)
        minibatch['idx'] = np.asarray([0] * batch_size)

        return minibatch

    def organize_batch(self, minibatch):
        batch_dict = {key: [] for key in minibatch[0].keys()}
        
        for batch in minibatch:
            for key in batch_dict.keys():
                batch_dict[key].append(batch[key])
        
        for key in batch_dict.keys():
            if isinstance(batch_dict[key][0], tuple):
                batches = []
                for i in range(len(batch_dict[key][0])):
                    batches.append(np.stack([x[i] for x in batch_dict[key]]))
                batch_dict[key] = batches
            else:
                batch_dict[key] = np.stack(batch_dict[key], axis = 0)

        return batch_dict

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'rb.pkl'), 'wb') as handle:
            pickle.dump(self.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
