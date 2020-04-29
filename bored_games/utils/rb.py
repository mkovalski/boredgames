# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    KEYS = ['state', 'valid_moves', 'action', 'reward',
               'next_state', 'next_valid_moves', 'done', 'idx']

    priority = False

    def __init__(self, N, path = None):
        self.memory = deque(maxlen = N)
    
    def append(self, batch):
        self.memory.append(batch)

    @property
    def maxlen(self):
        return self.memory.maxlen

    def get_batch(self, batch_size):
        
        minibatch = random.sample(self.memory, batch_size)
        minibatch = self.organize_batch(minibatch)
        minibatch['idx'] = np.asarray([0] * batch_size)

        return [minibatch[x] for x in self.KEYS]

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
