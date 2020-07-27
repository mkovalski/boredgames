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
    
    ALGO_NAME = 'DQN'

    def __init__(self, 
                 env,
                 model,
                 target_model,
                 exp_dir,
                 resume = False,
                 gamma = 0.99,
                 batch_size = 64,
                 target_update = 10,
                 epsilon = 1.0,
                 epsilon_min = 0.1,
                 epsilon_decay = 0.999,
                 decay_steps = 10000):
                
        self.env = env
        self.model = model
        self.target_model = target_model
        self.exp_dir = exp_dir

        self.model_path = None

        self.gamma = gamma    # discount rate
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min  # min exploration rate
        self.epsilon_decay = epsilon_decay
        self.decay_steps = decay_steps # how many training steps in which to decay
        
        # Set up directory
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)
        
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = 0.0001)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)

        self.train_loss = 0
        self.train_steps = 0
        self.nsteps = 0

    def reset(self):
        self.train_loss = 0
        self.nsteps = 0

    def memorize(self, minibatch):
        kwargs = {}
        
        if self.replay_buffer.priority:
            batch = self.replay_buffer.organize_batch([minibatch])

            orig_target, next_target = self.get_target(batch['state'],
                                                       batch['valid_actions'],
                                                       batch['action'],
                                                       batch['reward'],
                                                       batch['next_state'],
                                                       batch['next_valid_actions'],
                                                       batch['done'])

            kwargs['error'] = np.sum((orig_target - next_target)**2)

        self.replay_buffer.append(minibatch, **kwargs)

    def act(self, state, env, train = True):
        '''Epsilon greedy policy'''
        if train and np.random.rand() <= self.epsilon:
            return env.sample(env.player)
        
        state = [np.expand_dims(x, axis = 0) for x in state]
        state = [torch.from_numpy(x).to(self.device) for x in state]
        with torch.no_grad():
            act_values = self.model(*state).cpu()

        return act_values.numpy().squeeze(axis = 0)

    def calculate_error(self, state, valid_actions, action, reward,
                        next_state, next_valid_actions, done):
        
        with torch.no_grad():
            # Get target value
            next_state = [torch.from_numpy(x).to(self.device) for x in next_state]
            
            next_target = self.target_model(*next_state).cpu().numpy()

            next_target += ((1 - next_valid_actions) * -1e9)
            next_target = np.amax(next_target, axis = 1)

            target = reward + ((1 - done) * (self.gamma * next_target))
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
                 next_state, next_valid_actions, done):
        
        # Get target value
        next_state = [torch.from_numpy(x).to(self.device) for x in next_state]
        
        with torch.no_grad():
            next_target = self.target_model(*next_state).cpu().numpy()

        next_target += ((1 - next_valid_actions) * -1e9)
        next_target = np.amax(next_target, axis = 1)
        
        target = reward + ((1 - done) * (self.gamma * next_target))
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

    def get_loss(self):
        if self.nsteps != 0:
            return self.train_loss / self.nsteps
        return "Training hasn't started"

    def evaluate(self, eval_games = 50):
        nwins = 0

        for i in range(eval_games):
            state, _ = self.env.reset()
            done = False

            while not done:
                action = self.act(state, self.env, train = False)

                state, _, reward, done, _ = self.env.step(action)
                if done:
                    nwins += int(self.env.winner == 1)
        
        win_percentage = nwins / eval_games 
        print(" - Win percentage: {}".format(win_percentage))
        return win_percentage

    def train(self, 
             replay_buffer,
             episodes = 1000, 
             eval_eps = 50, 
             n_eval_games = 50, 
             rb = None):
        
        prev_win = 0.0
        writer = SummaryWriter(self.exp_dir)

        # Multiprocessing stuff
        eval_process = None
        eval_dict = {}
        result_queue = mp.Queue()

        for e in range(1, episodes + 1):
            self.reset()
            state, valid_actions = self.env.reset()
            done = False

            while not done:
                action = self.act(state, self.env)
                action *= valid_actions
                
                next_state, next_valid_actions, reward, done, _ = self.env.step(action)
                
                experience = dict(state = state,
                                  valid_actions = valid_actions,
                                  action = action,
                                  reward = reward,
                                  next_state = next_state,
                                  next_valid_actions = next_valid_actions,
                                  done = done)
                
                err_batch = replay_buffer.add_batch_dim(experience)
                 
                buffer_kwargs = {}
                if replay_buffer.priority:
                    error = self.calculate_error(**err_batch)
                    buffer_kwargs['error'] = error
                    
                replay_buffer.append(experience, **buffer_kwargs)

                if done:
                    print("episode: {}/{}, e: {:.2}, nsteps: {}, loss: {}"
                          .format(e, episodes, self.epsilon, self.nsteps, self.get_loss()), flush = True)
                    writer.add_scalar('Train/loss', self.get_loss(), e)
                    writer.add_scalar('Train/steps', self.nsteps, e)
                    writer.add_scalar('Train/epsilon', self.epsilon, e)

                batch = replay_buffer.get_batch(self.batch_size)

                item_loss, loss = self.optimize(state = batch['state'], 
                                               valid_actions = batch['valid_actions'], 
                                               action = batch['action'], 
                                               reward = batch['reward'],
                                               next_state = batch['next_state'], 
                                               next_valid_actions = batch['next_valid_actions'],
                                               done = batch['done'])
                
                if replay_buffer.priority:
                    for idx, rb_idx in enumerate(batch['idx']):
                        replay_buffer.update(rb_idx, item_loss[idx])

                # Update losses
                self.train_loss += loss
                self.nsteps += 1
                self.train_steps += 1

                if self.train_steps % self.decay_steps == 0 and self.epsilon > self.epsilon_min:
                    self.train_steps = 0
                    self.epsilon *= self.epsilon_decay
                
                state = next_state
                valid_actions = next_valid_actions
            
            if e % self.target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if e % eval_eps == 0:
                eval_process = self.check_eval_process(eval_process, result_queue, e, n_eval_games, writer)
        
        self.check_eval_process(eval_process, result_queue, e, n_eval_games, writer, start_new = False)
        
    def check_eval_process(self, eval_process, result_queue, episodes, n_eval_games, writer, start_new = True):
            if eval_process is not None:
                eval_process.join()
                result_dict = result_queue.get()
                print(" - Win percentage step {}: {}".format(
                    result_dict['eval_steps'], result_dict['win_percentage']))
                writer.add_scalar('Eval/win_percent', result_dict['win_percentage'], result_dict['eval_steps'])
            
            return_process = None
            if start_new:
                return_process = mp.Process(target = evaluate_dqn,
                                          args = (copy.deepcopy(self.env),
                                              copy.deepcopy(self.model),
                                              n_eval_games,
                                              episodes,
                                              result_queue,
                                              self.exp_dir))
                return_process.start()
            
            return return_process

def evaluate_dqn(env, model, num_games, eval_steps, result_queue, path):
    '''Separate function so this can be done in a separate process'''
    nwins = 0

    # Single thread execution for pytorch
    torch.set_num_threads(1)

    for i in range(num_games):
        state, _ = env.reset()
        done = False

        while not done:
            state = [np.expand_dims(x, axis = 0) for x in state]
            state = [torch.from_numpy(x).to('cpu') for x in state]
            with torch.no_grad():
                action = model(*state).cpu().numpy().squeeze(axis = 0)

            state, _, reward, done, _ = env.step(action)
            if done:
                nwins += int(env.winner == 1)
    
    win_percentage = nwins / num_games
    torch.save(model,
               os.path.join(path,
                            '{}_{}'.format(eval_steps, win_percentage)))
    
    result_queue.put(dict(win_percentage = win_percentage,
                          eval_steps = eval_steps))

def evaluate(model_path, render = 'ascii'):
    assert(render in ['ascii', 'pillow'])
    
    env = Quoridor(player = 1)
    board_shape, tile_shape = env.state_shape()
    action_shape = env.action_shape()

    agent = DQNAgent(board_shape, tile_shape, action_shape,
                     model_path = model_path)
    
    eval_dir = 'evaluate'
    os.makedirs(eval_dir)

    agent.reset()
    env.reset()

    count = 0

    env.render_ascii(eval_dir, count)

    while not env.done:
        state = env.get_state()
        moves = env.get_all_moves()

        action = agent.act(state, env, train = False)
        tmp_action = action + ((1 - moves) * -1e9)
        print(np.argmax(tmp_action), np.amax(tmp_action))

        env.move(player = PLAYER, move = action)
        count += 1

        env.render(eval_dir, count)

        if not env.done:
            action = env.sample(player = 2)
            env.move(player = 2, move = action)
            count += 1

            env.render(eval_dir, count)
    
    print(env.winner)
