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
import time
import multiprocessing as mp
import argparse
from tqdm import tqdm
import pickle

EPISODES = 10000
class DQNAgent:
    def __init__(self, board_size, nmoves_size, action_size, 
                 maxlen = 1000000, 
                 episodes = 10000,
                 gamma = 0.95,
                 epsilon = 1.0,
                 epsilon_min = 0.1,
                 epsilon_decay = 0.9999999,
                 use_pooling = True):

        self.board_size = board_size
        self.nmoves_size = nmoves_size
        self.action_size = action_size

        self.memory = deque(maxlen=maxlen)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = NN(board_size, nmoves_size, action_size).get_model()

        self.train_loss = 0
        self.nsteps = 0
        self.target_model = clone_model(self.model)

        self.use_pooling = use_pooling
    
    def reset(self):
        self.train_loss = 0
        self.nsteps = 0
        self.target_model = clone_model(self.model)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, env, train = True):
        if train and np.random.rand() <= self.epsilon:
            return env.sample(1)
        act_values = self.model.predict([np.expand_dims(state[0], axis = 0), 
                                        np.expand_dims(state[1], axis = 0)])[0]
        return act_values
    
    def get_batch(self, batch_size):
        # TODO: Optimize
        minibatch = random.sample(self.memory, batch_size)
        
        batch_dict = {}
        batch_dict['state'] = [[], []]
        batch_dict['action'] = [[]]
        batch_dict['reward'] = [[]]
        batch_dict['next_state'] = [[], []]
        batch_dict['done'] = [[]]

        for batch in minibatch:
            batch_dict['state'][0].append(batch[0][0])
            batch_dict['state'][1].append(batch[0][1])
            
            batch_dict['action'][0].append(batch[1])
            batch_dict['reward'][0].append(batch[2])

            batch_dict['next_state'][0].append(batch[3][0])
            batch_dict['next_state'][1].append(batch[3][1])

            batch_dict['done'][0].append(batch[4])

        for key in batch_dict.keys():
            for i in range(len(batch_dict[key])):
                batch_dict[key][i] = np.stack(batch_dict[key][i], axis = 0)
            
            if len(batch_dict[key]) == 1:
                batch_dict[key] = batch_dict[key][0]

        return batch_dict['state'], batch_dict['action'], \
               batch_dict['reward'], batch_dict['next_state'], batch_dict['done']

    def replay(self, batch_size):
        # Stack em, train as batches
        state, action, reward, next_state, done = self.get_batch(batch_size)
    
        target = (reward + self.gamma *
                  np.amax(self.target_model.predict([next_state[0], next_state[1]]), axis = 1))

        indices = np.where(done)
        target[indices] = reward[indices]

        target_f = self.target_model.predict([state[0], state[1]])

        x_indices = np.arange(0, action.shape[0])
        y_indices = np.argmax(action, axis = 1)

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

    def evaluate(self, env, eval_games = 10):
        env.reset(player = 1)
        nwins = 0

        for i in range(eval_games):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state, env)

                state, reward, done, _ = env.step(1, action)
                if done:
                    nwins += int(reward == 1)
        
        print(" - Win percentage: {}".format(nwins / eval_games))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def populate_rb():
    env = Quoridor()
    board_shape, tile_shape = env.state_shape()
    action_shape = env.action_shape()

    agent = DQNAgent(board_shape, tile_shape, action_shape)
    done = False
    batch_size = 256

    # Fill up the replay buffer
    done = True
    
    print("Populating replay buffer...")
    for i in tqdm(range(agent.memory.maxlen)):
        if done:
            agent.reset()
            state = env.reset(player = 1,
                              move_prob = [np.random.random(), np.random.random()])
            done = False
        
        action = agent.act(state, env)
        
        next_state, reward, done, _ = env.step(1, action)

        agent.memorize(state, action, reward, next_state, done)
        state = next_state
    
    with open('rb.pkl', 'wb') as myFile:
        pickle.dump(agent.memory, myFile)


def train(rb = None, offpolicy = False):
    env = Quoridor()
    board_shape, tile_shape = env.state_shape()
    action_shape = env.action_shape()

    agent = DQNAgent(board_shape, tile_shape, action_shape)
    done = False
    batch_size = 32

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
                state = env.reset(player = 1,
                                  move_prob = [np.random.random(), np.random.random()])
                done = False
            
            action = agent.act(state, env)
            
            next_state, reward, done, _ = env.step(1, action)

            agent.memorize(state, action, reward, next_state, done)
            state = next_state
    
    if offpolicy:
        train_off_policy(agent, batch_size)
        return

    for e in range(1, EPISODES + 1):
        agent.reset()
        state = env.reset(player = 1,
                          move_prob = [np.random.random(), np.random.random()])
        done = False

        while not done:
            action = agent.act(state, env)
            
            next_state, reward, done, _ = env.step(1, action)

            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, e: {:.2}, nsteps: {}, loss: {}"
                      .format(e, EPISODES, agent.epsilon, agent.nsteps, agent.get_loss()))

            agent.replay(batch_size)


        if e % 100 == 0:
            agent.save("./save/tdqn-quoridor-dqn.h5")
            agent.evaluate(env)

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

    agent = DQNAgent(board_shape, tile_shape, action_shape)
    agent.load(model_path)
    
    eval_dir = 'evaluate'
    os.makedirs(eval_dir)

    agent.reset()
    env.reset(player = 1, move_prob = 0.11)

    count = 0

    env.render(eval_dir, count)

    while not env.done:
        state = env.get_state(player = 1)
        action = agent.act(state, env, train = False)

        env.move(player = 1, move = action)
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

    parser.add_argument('--evaluate', action = 'store_true')
    parser.add_argument('--rb', type = str, default = None)
    parser.add_argument('--offpolicy', action = 'store_true')
    parser.add_argument('--model', type = str, default = None)

    args = parser.parse_args()
    
    if args.rb is not None:
        assert(os.path.isfile(args.rb))

    if not args.evaluate:
        train(rb = args.rb, offpolicy = args.offpolicy)
    else:
        assert(os.path.isfile(args.model))
        evaluate(args.model)
