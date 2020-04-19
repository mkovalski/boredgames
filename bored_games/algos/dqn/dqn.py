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
from collections import deque
from keras.models import clone_model
from bored_games.utils import ReplayBuffer

class DQNAgent:
    '''Base DQN agent. For more complicated DQN, use TQDN agent, Double DQN, etc.'''
    def __init__(self, 
                 env, 
                 exp_dir,
                 model_name = 'model.h5',
                 buffer_size = 20000, 
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

        self.replay_buffer = ReplayBuffer(buffer_size)

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
        self.nsteps = 0
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, nsteps, win_percent):
        self.model.save_weights(
            os.path.join(self.exp_dir, 'model_{}_{}.h5'.format(step, win_percent)))

    def reset(self):
        self.train_loss = 0
        self.nsteps = 0

    def memorize(self, batch):
        self.replay_buffer.append(batch)

    def act(self, state, env, train = True):
        '''Epsilon greedy policy'''
        if train and np.random.rand() <= self.epsilon:
            return env.sample(PLAYER)

        act_values = self.model.predict(state)
        return act_values
    
    def update(self):
        # Stack em, train as batches
        batch = self.get_batch(self.batch_size)
        
        next_target = self.model.predict(batch['next_state'])

        next_target += ((1 - batch['next_valid_moves') * -1e9)
        next_target = np.amax(next_target, axis = 1)
        
        target = reward + ((1 - batch['done']) * (self.gamma * next_target))

        target_f = self.model.predict(batch['state'])

        x_indices = np.arange(0, self.batch_size)
        y_indices = np.argmax(batch['action'] + ((1 - batch['valid_moves']) * -1e-9), axis = 1)

        target_f[x_indices, y_indices] = target
        
        hist = self.model.fit(batch['state'], target_f, epochs=1, verbose=0)

        self.train_loss += hist.history['loss'][0]
        self.nsteps += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_loss(self):
        if self.nsteps != 0:
            return self.train_loss / self.nsteps
        return "Training hasn't started"

    def evaluate(self, eval_games = 50):
        nwins = 0

        for i in range(eval_games):
            state, _ = self.env.reset(player = PLAYER, move_prob = np.random.random())
            done = False

            while not done:
                action = self.act(state, env, train = False)

                state, _, reward, done, _ = self.env.step(PLAYER, action)
                if done:
                    nwins += int(reward == PLAYER)
        
        win_percentage = nwins / eval_games 
        print(" - Win percentage: {}".format(win_percentage))
        return win_percentage

    def run_sim(self, player_model, opponent_model = None):
        pass

    def populate_rb(self, path):
        assert(not os.path.isfile(path)), \
            "Path {} for replay buffer already exists!".format(path)

        # Fill up the replay buffer
        done = True
        
        print("Populating replay buffer...")
        for i in tqdm(range(self.memory.maxlen)):
            if done:
                self.reset()
                state, valid_moves = self.env.reset(player = PLAYER,
                                                    move_prob = [np.random.random(), np.random.random()])

            action = self.act(state, self.env)
            
            next_state, next_valid_moves, reward, done, _ = self.env.step(PLAYER, action)

            self.memorize(dict(state = state, 
                               valid_moves = valid_moves, 
                               action = action, 
                               reward = reward, 
                               next_state = next_state, 
                               next_valid_moves = next_valid_moves, 
                               done = done))
            state = next_state
            valid_moves = next_valid_moves
        
        with open(path, 'wb') as myFile:
            pickle.dump(self.memory, myFile)

    def train(self, episodes = 1000, eval_eps = 50, ngames = 50, rb = None):
        if rb is not None:
            with open(rb, 'rb') as myFile:
                replay_buffer = pickle.load(myFile)
            self..memory = replay_buffer
            print("Loaded replay buffer!")
        else:
            
            print("Populating replay buffer...")
            self.populate_rb()
        
        prev_win = 0.0

        for e in range(1, episodes + 1):
            self.reset()
            state, valid_moves = env.reset(player = PLAYER,
                              move_prob = [np.random.random(), np.random.random()])
            done = False

            while not done:
                action = self.act(state, env)
                
                next_state, next_valid_moves, reward, done, _ = env.step(PLAYER, action)
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
                          .format(e, EPISODES, self.epsilon, self.nsteps, self.get_loss()))

                self.update()
            
            # TODO: Parallel
            if e % eval_eps == 0:
                win_percentage = self.evaluate(env, eval_games)
                self.save(win_percentage, e)

def evaluate(model_path):
    env = Quoridor()
    board_shape, tile_shape = env.state_shape()
    action_shape = env.action_shape()

    agent = DQNAgent(board_shape, tile_shape, action_shape,
                     model_path = model_path)
    
    eval_dir = 'evaluate'
    os.makedirs(eval_dir)

    agent.reset()
    env.reset(player = PLAYER, move_prob = 0.1)

    count = 0

    env.render(eval_dir, count)

    while not env.done:
        state = env.get_state(PLAYER)
        moves = env.get_all_moves(PLAYER)

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
