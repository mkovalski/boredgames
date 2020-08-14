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

class DQN:
    '''Base DQN agent. For more complex DQN, use TQDN agent, Double DQN, etc.'''
    
    ALGO_NAME = 'DQN'

    def __init__(self, 
                 env,
                 agent,
                 exp_dir,
                 epsilon = 1.0,
                 epsilon_min = 0.1,
                 epsilon_decay = 0.999,
                 gamma = 0.99,
                 batch_size = 64,
                 target_update = 10,
                 decay_steps = 10000):
                
        self.env = env
        self.agent = agent

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.exp_dir = exp_dir
        self.batch_size = batch_size
        self.target_update = target_update
        self.decay_steps = decay_steps # how many training steps in which to decay
        
        # Set up directory
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)
        
    def reset(self):
        self.train_loss = 0
        self.nsteps = 0

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

        # Setup the history for the number of players
        queues = [deque(maxlen=2) for i in range(self.env.NUMPLAYERS)]

        # Multiprocessing stuff
        eval_process = None
        eval_dict = {}
        result_queue = mp.Queue()

        idx = 0
        done = False
        donecount = 0
        train_steps = 0

        for e in range(1, episodes + 1):
            self.reset()
            state, valid_actions = self.env.reset()

            done = False
            donecount = 0
            idx = 0

            for q in queues:
                while q:
                    q.pop()
            
            while donecount != self.env.NUMPLAYERS:

                # Take an action
                if np.random.random() < self.epsilon:
                    action = self.env.sample()
                else:
                    action = self.agent.act(state)

                action *= valid_actions

                queues[idx].append(dict(state = state,
                                        valid_actions = valid_actions,
                                        action = action,
                                        reward = self.env.get_reward(),
                                        done = done))

                if len(queues[idx]) == queues[idx].maxlen:
                    experience = dict(state = queues[idx][0]['state'],
                                      valid_actions = queues[idx][0]['valid_actions'],
                                      action = queues[idx][0]['action'],
                                      next_state = queues[idx][1]['state'],
                                      next_valid_actions = queues[idx][1]['valid_actions'],
                                      reward = queues[idx][1]['reward'],
                                      done = queues[idx][1]['done'])
                
                    err_batch = replay_buffer.add_batch_dim(experience)
                 
                    buffer_kwargs = {}
                    if replay_buffer.priority:
                        error = self.calculate_error(**err_batch, gamma = self.gamma)
                        buffer_kwargs['error'] = error
                        
                    replay_buffer.append(experience, **buffer_kwargs)

                if done:
                    donecount += 1
                
                # Take the step
                state, valid_actions, _, done, _ = self.env.step(action)

                # Sample a batch
                batch = replay_buffer.get_batch(self.batch_size)
                
                # Update network
                item_loss, loss = self.agent.optimize(state = batch['state'], 
                                                      valid_actions = batch['valid_actions'], 
                                                      action = batch['action'], 
                                                      reward = batch['reward'],
                                                      next_state = batch['next_state'], 
                                                      next_valid_actions = batch['next_valid_actions'],
                                                      done = batch['done'],
                                                      gamma = self.gamma)
                
                if replay_buffer.priority:
                    for idx, rb_idx in enumerate(batch['idx']):
                        replay_buffer.update(rb_idx, item_loss[idx])

                # Update losses
                self.train_loss += loss
                self.nsteps += 1
                train_steps += 1

                if train_steps % self.decay_steps == 0:
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay

                    train_steps = 0
                
                idx = (idx + 1) % self.env.NUMPLAYERS
            
            print("episode: {}/{}, e: {:.2}, nsteps: {}, loss: {}"
                  .format(e, episodes, self.epsilon, self.nsteps, self.get_loss()), flush = True)
            writer.add_scalar('Train/loss', self.get_loss(), e)
            writer.add_scalar('Train/steps', self.nsteps, e)
            writer.add_scalar('Train/epsilon', self.epsilon, e)
            
            if e % self.target_update == 0:
                self.agent.update_target_model()

            if e % eval_eps == 0:
                eval_process = self.check_eval_process(eval_process, result_queue, e, n_eval_games, writer)
        
        self.check_eval_process(eval_process, result_queue, e, n_eval_games, writer, start_new = False)
        
    def check_eval_process(self, eval_process, result_queue, episodes, n_eval_games, writer, start_new = True):
            '''
            if eval_process is not None:
                eval_process.join()
                result_dict = result_queue.get()
                print(" - Win percentage step {}: {}".format(
                    result_dict['eval_steps'], result_dict['win_percentage']))
                writer.add_scalar('Eval/win_percent', result_dict['win_percentage'], result_dict['eval_steps'])
            
            return_process = None
            '''

            if start_new:
                evaluate_dqn(copy.deepcopy(self.env),
                             copy.deepcopy(self.agent),
                             n_eval_games,
                             episodes,
                             result_queue,
                             self.exp_dir)
                result_dict = result_queue.get()
                
                print(" - Win percentage step {}: {}".format(
                    result_dict['eval_steps'], result_dict['win_percentage']))
                writer.add_scalar('Eval/win_percent', result_dict['win_percentage'], result_dict['eval_steps'])

                '''
                return_process = mp.Process(target = evaluate_dqn,
                                          args = (copy.deepcopy(self.env),
                                              copy.deepcopy(self.agent),
                                              n_eval_games,
                                              episodes,
                                              result_queue,
                                              self.exp_dir))
                return_process.start()
                '''
            
            #return return_process

def evaluate_dqn(env, agent, num_games, eval_steps, result_queue, path):
    '''Separate function so this can be done in a separate process'''
    nwins = 0

    # Single thread execution for pytorch
    #torch.set_num_threads(1)

    for i in range(num_games):
        env.reset()
        done = False
        
        # Alternate who goes first
        myturn = True if i % 2 == 0 else False
           
        # EH to this whole thing: TODO
        while True:
            if myturn:
                if done:
                    nwins += int(env.get_reward() == 1)
                    break

                action = agent.act(env.get_state())
                env.step(action)

            else:
                _, _, _, done, _ = env.step(env.sample())

            myturn = not myturn

    win_percentage = nwins / num_games
    torch.save(agent.model,
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
