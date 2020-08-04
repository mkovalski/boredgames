#!/usr/bin/env python

from bored_games.envs.quoridor import Quoridor
from bored_games.envs.quoridor.models import QuoridorDQN
from bored_games.algos import DQNAgent
import numpy as np
import torch
import os
import sys
import copy
import argparse
import curses
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, required = True)

args = parser.parse_args()

# Setup environment
env = Quoridor(set_move_prob = True)
action_shape, state_shape = env.action_shape(), env.state_shape()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 # Create a model
model = QuoridorDQN(board_shape = state_shape[0],
                    tile_shape = state_shape[1],
                    output_shape = action_shape).to(device)


model = torch.load(args.model)
#model.load_state_dict(torch.load(args.model))
model.eval()

# Eval time y'all
env.reset(move_opponent = False)

video_path = 'tmp'

env.render(video_path, 0)

curr_player = np.random.randint(1, 3)
done = False
count = 0

while not env.done:
    count += 1
    time.sleep(1)

    if curr_player != 1:
        action = env.sample()
        env.move(curr_player, action)
    else:
        with torch.no_grad():
            state = [torch.from_numpy(np.expand_dims(x, axis = 0)).to(device) for x in env.get_state()]
            action = model(*state).detach().cpu().numpy()

        action = np.squeeze(action, axis = 0)
        env.move(curr_player, action)

    env.render(video_path, count)
    curr_player = (curr_player % 2) + 1

