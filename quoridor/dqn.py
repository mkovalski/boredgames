# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
from quoridor import Quoridor
from nn import NN
import os
import sys
import time
import multiprocessing as mp
import argparse
from tqdm import tqdm
import pickle

EPISODES = 10000
PLAYER = 1

class DQNAgent:
    def __init__(self, board_size, nmoves_size, action_size,
                 model_path = None,
                 buffer_size = 20000, 
                 gamma = 0.99,
                 epsilon = 1.0,
                 epsilon_min = 0.1,
                 batch_size = 32,
                 epsilon_decay = 0.999999,
                 use_pooling = True):

        self.board_size = board_size
        self.nmoves_size = nmoves_size
        self.action_size = action_size
        self.batch_size = batch_size

        self.memory = deque(maxlen=buffer_size)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = NN(board_size, nmoves_size, action_size).get_model()
        
        self.model_path = model_path
        if os.path.isfile(self.model_path):
            print("Loading latest weights!")
            self.load(self.model_path)
            self.model_path = os.path.dirname(os.path.abspath(self.model_path))
            print("Saving in {}".format(self.model_path))

        self.train_loss = 0
        self.nsteps = 0

        self.target_model = clone_model(self.model)
    
    def update_target(self):
        self.target_model = clone_model(self.model)

    def reset(self):
        self.train_loss = 0
        self.nsteps = 0
        self.update_target()

    def memorize(self, state, avail_moves, action, reward, next_state, next_avail_moves, done):
        self.memory.append((state, avail_moves, action, reward, next_state, next_avail_moves, done))

    def act(self, state, env, train = True):
        if train and np.random.rand() <= self.epsilon:
            return env.sample(PLAYER)
        act_values = self.model.predict([np.expand_dims(state[0], axis = 0), 
                                        np.expand_dims(state[1], axis = 0)])[0]
        return act_values
    
    def get_batch(self, batch_size):
        # TODO: Optimize
        minibatch = random.sample(self.memory, batch_size)
        
        batch_dict = {}
        batch_dict['state'] = [[], []]
        batch_dict['avail_moves'] = [[]]
        batch_dict['action'] = [[]]
        batch_dict['reward'] = [[]]
        batch_dict['next_state'] = [[], []]
        batch_dict['next_avail_moves'] = [[]]
        batch_dict['done'] = [[]]

        for batch in minibatch:
            batch_dict['state'][0].append(batch[0][0])
            batch_dict['state'][1].append(batch[0][1])

            batch_dict['avail_moves'][0].append(batch[1])
            
            batch_dict['action'][0].append(batch[2])
            batch_dict['reward'][0].append(batch[3])

            batch_dict['next_state'][0].append(batch[4][0])
            batch_dict['next_state'][1].append(batch[4][1])

            batch_dict['next_avail_moves'][0].append(batch[5])

            batch_dict['done'][0].append(batch[6])

        for key in batch_dict.keys():
            for i in range(len(batch_dict[key])):
                batch_dict[key][i] = np.stack(batch_dict[key][i], axis = 0)
            
            if len(batch_dict[key]) == 1:
                batch_dict[key] = batch_dict[key][0]

        return batch_dict['state'], batch_dict['avail_moves'], batch_dict['action'], \
               batch_dict['reward'], batch_dict['next_state'], batch_dict['next_avail_moves'], \
               batch_dict['done']

    def replay(self):
        # Stack em, train as batches
        state, avail_moves, action, reward, next_state, next_avail_moves, done = self.get_batch(self.batch_size)
        
        next_target = self.target_model.predict([next_state[0], next_state[1]])
        next_target += ((1 - next_avail_moves) * -1e9)
        next_target = np.amax(next_target, axis = 1)
        
        target = reward + ((1 - done) * (self.gamma * next_target))

        target_f = self.target_model.predict([state[0], state[1]])

        x_indices = np.arange(0, action.shape[0])
        y_indices = np.argmax(action + ((1 - avail_moves) * -1e-9), axis = 1)

        target_f[x_indices, y_indices] = target
        
        hist = self.model.fit([state[0], state[1]], target_f, epochs=1, verbose=0)

        self.train_loss += hist.history['loss'][0]
        self.nsteps += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_loss(self):
        if self.nsteps != 0:
            return self.train_loss / self.nsteps
        return "Training hasn't started"

    def evaluate(self, env, eval_games = 50):
        nwins = 0

        for i in range(eval_games):
            state, _ = env.reset(player = PLAYER, move_prob = np.random.random())
            done = False
            while not done:
                action = self.act(state, env, train = False)

                state, _, reward, done, _ = env.step(PLAYER, action)
                if done:
                    nwins += int(reward == PLAYER)
        
        win_percentage = nwins / eval_games 
        print(" - Win percentage: {}".format(win_percentage))
        return win_percentage

    def load(self, name):
        self.model.load_weights(name)

    def save(self, win_percent, step):
        self.model.save_weights(
            os.path.join(self.model_path, 'model_{}_{}.h5'.format(step, win_percent)))

def populate_rb():
    env = Quoridor()
    board_shape, tile_shape = env.state_shape()
    action_shape = env.action_shape()

    agent = DQNAgent(board_shape, tile_shape, action_shape)
    done = False
    batch_size = 512

    # Fill up the replay buffer
    done = True
    
    print("Populating replay buffer...")
    for i in tqdm(range(agent.memory.maxlen)):
        if done:
            agent.reset()
            state, valid_moves = env.reset(player = PLAYER,
                                  move_prob = [np.random.random(), np.random.random()])
            done = False
        
        action = agent.act(state, env)
        
        next_state, next_valid_moves, reward, done, _ = env.step(PLAYER, action)

        agent.memorize(state, valid_moves, action, reward, next_state, next_valid_moves, done)
        state = next_state
        valid_moves = next_valid_moves
    
    with open('rb.pkl', 'wb') as myFile:
        pickle.dump(agent.memory, myFile)


def train(model_path, rb = None, offpolicy = False,
          batch_size = 32, buffer_size = 20000,
          epsilon = 1.0):

    env = Quoridor()
    board_shape, tile_shape = env.state_shape()
    action_shape = env.action_shape()

    agent = DQNAgent(board_shape, tile_shape, action_shape,
                     model_path = model_path,
                     batch_size = batch_size,
                     buffer_size = buffer_size,
                     epsilon = epsilon)
    done = False

    # Fill up the replay buffer
    done = True

    if rb is not None:
        with open(rb, 'rb') as myFile:
            replay_buffer = pickle.load(myFile)
        agent.memory = replay_buffer
        print("Loaded replay buffer!")
    else:
        print("Populating replay buffer...")
        for i in tqdm(range(agent.memory.maxlen)):
            if done:
                agent.reset()
                state, avail_moves = env.reset(player = PLAYER,
                                  move_prob = [np.random.random(), np.random.random()])
                done = False
            
            action = agent.act(state, env)
            
            next_state, next_avail_moves, reward, done, _ = env.step(PLAYER, action)

            agent.memorize(state, avail_moves, action, reward, next_state, next_avail_moves, done)
            state = next_state
            avail_moves = next_avail_moves
    
    if offpolicy:
        train_off_policy(agent, batch_size)
        return
    
    prev_win = 0.0
    for e in range(1, EPISODES + 1):
        agent.reset()
        state, avail_moves = env.reset(player = PLAYER,
                          move_prob = [np.random.random(), np.random.random()])
        done = False

        while not done:
            action = agent.act(state, env)
            
            next_state, next_avail_moves, reward, done, _ = env.step(PLAYER, action)

            agent.memorize(state, avail_moves, action, reward, next_state, next_avail_moves, done)
            state = next_state
            avail_moves = next_avail_moves
            if done:
                print("episode: {}/{}, e: {:.2}, nsteps: {}, loss: {}"
                      .format(e, EPISODES, agent.epsilon, agent.nsteps, agent.get_loss()))

            agent.replay()

        if e % 50 == 0:
            win_percentage = agent.evaluate(env)
            agent.save(win_percentage, e)

def train_off_policy(agent, batch_size):
    for i in range(1, 10000+1):
        agent.replay(batch_size)

        if i % 100 == 0:
            print("Iteration {}: Loss {}".format(i, agent.get_loss()))
            agent.reset()
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type = str, required = True)
    parser.add_argument('--evaluate', action = 'store_true')
    parser.add_argument('--rb', type = str, default = None)
    parser.add_argument('--buffer_size', type = int, default = 10000)
    parser.add_argument('--offpolicy', action = 'store_true')
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--epsilon', default = 1.0, type = float)

    args = parser.parse_args()

    if args.rb is not None:
        assert(os.path.isfile(args.rb))
    
    # Load or start fresh
    model_path = args.model_path
    if not os.path.isfile(model_path):
        assert(not os.path.isdir(model_path))
        os.makedirs(model_path)
    else:
        print("Loading a model: {}!".format(model_path))

    if not args.evaluate:
        train(rb = args.rb, offpolicy = args.offpolicy, model_path = model_path,
              batch_size = args.batch_size,
              buffer_size = args.buffer_size,
              epsilon = args.epsilon)
    else:
        assert(os.path.isfile(model_path))
        evaluate(model_path)
