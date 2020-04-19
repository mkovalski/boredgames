# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque

class ReplayBuffer:
	KEYS = set('state', 'valid_moves', 'action', 'reward',
			   'next_state', 'next_valid_moves', 'done')

    def __init__(self, N, path = None):
		if path is not None:
			pass
		else:
	        self.memory = deque(maxlen = N)
    
    def append(self, batch):
        self.memory.append(batch)

    def get_batch(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
		batch_dict = {key: [] for key in minibatch[0].keys()}

        for batch in minibatch:
			for key in batch_dict.keys():
				batch_dict[key].append(batch[key])

        for key in batch_dict.keys():
            if isinstance(batch_dict[key][0], list):
                for i in range(len(batch_dict[key])):
                    batch_dict[key][i] = np.stack(batch_dict[key][i], axis = 0)
            else:
                batch_dict[key] = np.stack(batch_dict[key], axis = 0)
		
		return batch_dict
