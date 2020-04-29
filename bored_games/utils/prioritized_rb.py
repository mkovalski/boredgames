# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from .rb import ReplayBuffer
from .sum_tree import SumTree

# Most of code used from https://github.com/germain-hug/Deep-RL-Keras/blob/master/utils/memory_buffer.py

class PrioritizedReplayBuffer(ReplayBuffer):
    prioritized = True

    def __init__(self, N):
        # Prioritized Experience Replay
        self.alpha = 0.5
        self.epsilon = 0.01
        self.buffer = SumTree(N)
        
        self.count = 0
        self.buffer_size = N

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

        return [batch[x] for x in self.KEYS]

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
