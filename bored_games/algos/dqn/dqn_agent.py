# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
import argparse
from tqdm import tqdm
import pickle
import json
import copy
from collections import deque
from bored_games.utils import ReplayBuffer, PrioritizedReplayBuffer

class DQNAgent:
    '''Base DQN agent. For more complex DQN, use TQDN agent, Double DQN, etc.'''
    
    def __init__(self, model):

        self.model = model

        self.target_model = copy.deepcopy(self.model)
        
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = 0.0001)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)
    
    def act(self, state):
        '''Epsilon greedy policy'''
        state = [np.expand_dims(x, axis = 0) for x in state]
        state = [torch.from_numpy(x).to(self.device) for x in state]
        with torch.no_grad():
            act_values = self.model(*state).cpu()

        return act_values.numpy().squeeze(axis = 0)

    def calculate_error(self, state, valid_actions, action, reward,
                        next_state, next_valid_actions, done, gamma):
        
        with torch.no_grad():
            # Get target value
            next_state = [torch.from_numpy(x).to(self.device) for x in next_state]
            
            next_target = self.target_model(*next_state).cpu().numpy()

            next_target += ((1 - next_valid_actions) * -1e9)
            next_target = np.amax(next_target, axis = 1)

            target = reward + ((1 - done) * (gamma * next_target))
            target = target.reshape(-1, 1)

            # Get the original state output
            tmp_argmax = np.argmax(action, axis = 1)
            indices = np.argmax(action + ((1 - valid_actions) * -1e-9), axis = 1)

            state = [torch.from_numpy(x).to(self.device) for x in state]
            pred = self.model(*state).gather(
                1, torch.from_numpy(indices.reshape(-1, 1)).to(self.device))


            # Update model
            loss = F.smooth_l1_loss(pred,
                                    torch.from_numpy(target).float().to(self.device))

        return loss.cpu().item()
    
    def optimize(self, state, valid_actions, action, reward,
                 next_state, next_valid_actions, done, gamma):
        
        # Get target value
        next_state = [torch.from_numpy(x).to(self.device) for x in next_state]
        
        with torch.no_grad():
            next_target = self.target_model(*next_state).cpu().numpy()

        next_target += ((1 - next_valid_actions) * -1e9)
        next_target = np.amax(next_target, axis = 1)
        
        target = reward + ((1 - done) * (gamma * next_target))
        target = target.reshape(-1, 1)

        # Get the original state output
        indices = np.argmax(action + ((1 - valid_actions) * -1e-9), axis = 1)

        state = [torch.from_numpy(x).to(self.device) for x in state]
        pred = self.model(*state).gather(
            1, torch.from_numpy(indices.reshape(-1, 1)).to(self.device))
    
        
        # Update model
        loss = F.smooth_l1_loss(pred.float(),
                                torch.from_numpy(target).float().to(self.device),
                                reduction = 'none')
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.mean().backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if loss.mean().cpu().item() > 1000.0:
            print("Warning! Huge loss value")
        
        return loss.detach().cpu().numpy(), loss.mean().cpu().item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
