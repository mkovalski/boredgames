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
from bored_games.utils import ReplayBuffer, PrioritizedReplayBuffer

class DQNAgent:
    '''Base DQN agent. For more complicated DQN, use TQDN agent, Double DQN, etc.'''
    
    ALGO_NAME = 'DQN'

    def __init__(self, 
                 env, 
                 exp_dir,
                 model_name = 'model.h5',
                 buffer_size = 20000,
                 priority = False,
                 gamma = 0.99,
                 batch_size = 32,
                 epsilon = 1.0,
                 epsilon_min = 0.1,
                 epsilon_decay = 0.99,
                 decay_steps = 1000):
                
        self.env = env
        self.exp_dir = exp_dir
        self.model_path = None

        self.gamma = gamma    # discount rate
        self.batch_size = batch_size
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min  # min exploration rate
        self.epsilon_decay = epsilon_decay
        self.decay_steps = decay_steps # how many training steps in which to decay
        
        if priority:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Set up directory
        if os.path.isdir(self.exp_dir):
            model_path = os.path.join(self.exp_dir, model_name)
            if os.path.isfile(model_path):
                print("Cannot overwrite an old model, {} already exists!".format(model_path))
                sys.exit(2)
            self.model_path = model_path

        else:
            os.makedirs(self.exp_dir)
            self.model_path = os.path.join(self.exp_dir, model_name)
            
        self.model = self.env.create_model()
        
        self.train_loss = 0
        self.train_steps = 0
        self.nsteps = 0

    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, nsteps, win_percent):
        self.model.save_weights(
            os.path.join(self.exp_dir, 'model_{}_{}.h5'.format(nsteps, win_percent)))

        # TODO: Add json config for latest results
        # make this a parent method / abstract

    def reset(self):
        self.train_loss = 0
        self.nsteps = 0

    def memorize(self, minibatch):
        kwargs = {}
        
        if self.replay_buffer.priority:
            batch = self.replay_buffer.organize_batch([minibatch])

            orig_target, next_target = self.get_target(batch['state'],
                                                       batch['valid_moves'],
                                                       batch['action'],
                                                       batch['reward'],
                                                       batch['next_state'],
                                                       batch['next_valid_moves'],
                                                       batch['done'])

            kwargs['error'] = np.sum((orig_target - next_target)**2)

        self.replay_buffer.append(minibatch, **kwargs)

    def act(self, state, env, train = True):
        '''Epsilon greedy policy'''
        if train and np.random.rand() <= self.epsilon:
            return env.sample(env.player)
        
        state = self.process_state(state)
        
        act_values = np.squeeze(self.model.predict(state), axis = 0)
        return act_values
    
    def process_state(self, state):
        if isinstance(state, tuple):
            state = [np.expand_dims(x, axis = 0) for x in state]
        else:
            state = np.expand_dims(state, axis = 0)

        return state
    
    def predict_next_target(self, batch):
        return self.model.predict(batch)    
    
    def get_target(self, state, valid_moves, action, reward,
                   next_state, next_valid_moves, done):
        
        next_target = self.predict_next_target(next_state)

        next_target += ((1 - next_valid_moves) * -1e9)
        next_target = np.amax(next_target, axis = 1)
        
        target = reward + ((1 - done) * (self.gamma * next_target))

        orig_target = self.model.predict(state)
        target_f = np.copy(orig_target)

        x_indices = np.arange(0, done.shape[0])
        y_indices = np.argmax(action + ((1 - valid_moves) * -1e-9), axis = 1)

        target_f[x_indices, y_indices] = target

        return orig_target, target_f

    def update(self):
        # Stack em, train as batches
        
        state, valid_moves, action, reward, \
            next_state, next_valid_moves, done, idx = self.replay_buffer.get_batch(
                self.batch_size)
       
        orig_target, target_f = self.get_target(state, valid_moves, action, reward,
                                   next_state, next_valid_moves, done)
        


        
        hist = self.model.fit(state, target_f, epochs=1, verbose=0)

        self.train_loss += hist.history['loss'][0]
        self.nsteps += 1
        self.train_steps += 1

        if self.train_steps % self.decay_steps == 0 and self.epsilon > self.epsilon_min:
            self.train_steps = 0
            self.epsilon *= self.epsilon_decay
    
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
                    nwins += int(reward == 1)
        
        win_percentage = nwins / eval_games 
        print(" - Win percentage: {}".format(win_percentage))
        return win_percentage

    def run_sim(self, player_model, opponent_model = None):
        pass

    def populate_rb(self):
        # Fill up the replay buffer
        done = True
        
        print("Populating replay buffer...")
        for i in tqdm(range(self.replay_buffer.maxlen)):
            if done:
                self.reset()
                state, valid_moves = self.env.reset()

            action = self.act(state, self.env)
            
            next_state, next_valid_moves, reward, done, _ = self.env.step(action)

            self.memorize(dict(state = state, 
                               valid_moves = valid_moves, 
                               action = action, 
                               reward = reward, 
                               next_state = next_state, 
                               next_valid_moves = next_valid_moves, 
                               done = done))
            state = next_state
            valid_moves = next_valid_moves
        
        with open(os.path.join(self.exp_dir, 'rb.pkl'), 'wb') as myFile:
            pickle.dump(self.replay_buffer, myFile)

    def train(self, 
             episodes = 1000, 
             eval_eps = 50, 
             n_eval_games = 50, 
             rb = None):

        if rb is not None:
            with open(rb, 'rb') as myFile:
                replay_buffer = pickle.load(myFile)
            self.replay_buffer = replay_buffer
            print("Loaded replay buffer!")
        else:
            self.populate_rb()
        
        prev_win = 0.0

        for e in range(1, episodes + 1):
            self.reset()
            state, valid_moves = self.env.reset()
            done = False

            while not done:
                action = self.act(state, self.env)
                
                next_state, next_valid_moves, reward, done, _ = self.env.step(action)
                self.memorize(dict(state = state, 
                                   valid_moves = valid_moves, 
                                   action = action, 
                                   reward = reward, 
                                   next_state = next_state, 
                                   next_valid_moves = next_valid_moves, 
                                   done = done))

                state = next_state
                valid_moves = next_valid_moves

                if done:
                    print("episode: {}/{}, e: {:.2}, nsteps: {}, loss: {}"
                          .format(e, episodes, self.epsilon, self.nsteps, self.get_loss()), flush = True)

                self.update()
            
            # TODO: Separate process
            if e % eval_eps == 0:
                win_percentage = self.evaluate(n_eval_games)
                self.save(e, win_percentage)

def evaluate(model_path):
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

    env.render(eval_dir, count)

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
