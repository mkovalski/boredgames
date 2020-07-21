# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from tqdm import tqdm
from .rb import ReplayBuffer
from .sum_tree import SumTree
import os
import pickle

# Most of code used from https://github.com/germain-hug/Deep-RL-Keras/blob/master/utils/memory_buffer.py

class PrioritizedReplayBuffer(ReplayBuffer):
    priority = True

    def __init__(self, N, path = None):
        # Prioritized Experience Replay
        self.alpha = 0.5
        self.epsilon = 0.01

        if path:
            self.load(path)
            self.buffer_size = self.buffer.capacity
        else:
            self.buffer = SumTree(N)
            self.buffer_size = N
        
        self.count = 0
    
    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'rb.pkl'), 'wb') as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as handle:
            self.buffer = pickle.load(handle)
        print("Successfully loaded memory!")

    def append(self, batch, error):
        """ Save an experience to memory, optionally with its TD-Error
        """

        priority = self.priority(error)
        self.buffer.add(priority, batch)
        self.count += 1

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (error + self.epsilon) ** self.alpha

    def size(self):
        """ Current Buffer Occupation
        """
        return self.count

    def get_batch(self, batch_size):
        """ Sample a batch
        """
        batch = []
        indices = []

        # Sample using prorities
        T = self.buffer.total() // batch_size
        for i in range(batch_size):
            a, b = T * i, T * (i + 1)
            s = random.uniform(a, b)
            idx, error, data = self.buffer.get(s)
            batch.append(data)
            indices.append(idx)
        
        batch = self.organize_batch(batch)
        batch['idx'] = np.stack(indices)
        
        return batch

    def update(self, idx, new_error):
        """ Update priority for idx (PER)
        """
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        if(self.with_per): self.buffer = SumTree(buffer_size)
        else: self.buffer = deque()
        self.count = 0
 
    @property
    def maxlen(self):
        return self.buffer_size
    
    def populate(self, env, N = None):
        if N is None:
            N = self.buffer_size
        if N > self.buffer_size:
            print("Cannot populate more memory than size of buffer, will populate {}".format(self.buffer_size))
            N = self.buffer_size

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

            self.append(batch, error = np.random.random())

            state = next_state
            valid_actions = next_valid_actions

