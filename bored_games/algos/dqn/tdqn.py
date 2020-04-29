# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import sys
import time
import multiprocessing as mp
import argparse
from tqdm import tqdm
import pickle
import json
from collections import deque
from keras.models import clone_model
from bored_games.utils import ReplayBuffer
from .dqn import DQNAgent

class TDQNAgent(DQNAgent):
    '''TDQN agent. For more complicated DQN, use TQDN agent, Double DQN, etc.'''
   
    def __init__(self, *args, n_target_updates = 1e6, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_model = clone_model(self.model)
        self.n_target_updates = n_target_updates
        self.n_target_steps = 0
    
    def predict_next_target(self, batch):
        return self.target_model.predict(batch)
    
    def update(self):
        super().update()
        
        self.n_target_steps += 1

        if self.n_target_steps % self.n_target_updates == 0:
            print("Updating target model!")
            w = self.model.get_weights()
            self.target_model.set_weights(w)
            self.n_target_steps = 0
