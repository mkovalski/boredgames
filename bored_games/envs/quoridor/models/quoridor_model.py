#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuoridorDQN(nn.Module):
    def __init__(self, 
                 board_shape,
                 tile_shape,
                 output_shape, 
                 n_conv_layers = 2, 
                 base_conv_filters = [16, 32, 64], 
                 merge_dim = 256):
        '''A model for quoridor env based for DQN

        Args:
            board_shape (tuple): Shape of the quoridor board state
            tile_shape (tuple): Shape of the tiles used for state
            output_shape (tuple): Expected output shape for action size
            n_conv_layers (int): Number of convolutional layers to use
            base_conv_filters (list): Filters to use for each layer,
                length of list must be equal to `n_conv_layers`
            merge_dim (int): Size of merge dimension for board shape
        
        '''
        super(QuoridorDQN, self).__init__()
    
        self.board_shape = board_shape
        self.tile_shape = tile_shape
        self.output_shape = output_shape
        self.n_conv_layers = n_conv_layers
        self.base_conv_filters = base_conv_filters
        self.merge_dim = merge_dim
        
        self.convs = []
        
        prev = 1

        for i in range(self.n_conv_layers):
            self.convs.append(nn.Conv2d(prev, base_conv_filters[i],
                                     kernel_size = 5))
            prev = base_conv_filters[i]
        
        sh = self.__calc_dense_input_shape()
        
        self.linear = nn.Linear(sh, self.merge_dim)
        self.linear2 = nn.Linear(self.merge_dim + self.tile_shape[0], self.merge_dim)
        self.output_layer = nn.Linear(self.merge_dim, np.prod(self.output_shape))

    def __calc_dense_input_shape(self):
        inp = torch.randn(1, *self.board_shape, dtype = torch.float)
        for i in range(self.n_conv_layers):
            out = self.convs[i](inp)
            inp = out
        
        return out.view(1, -1).shape[-1]

    def forward(self, board, tiles):

        bs = board.shape[0]

        inp = board
        for i in range(self.n_conv_layers):
            out = F.relu(self.convs[i](inp))
            inp = out
        
        out = F.relu(self.linear(out.view(bs, -1)))
        out = torch.cat([out, tiles], dim = 1)
        out = F.relu(self.linear2(out))
        out = self.output_layer(out)

        return out

if __name__ == '__main__':
    
    from bored_games.envs.quoridor import Quoridor 

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    env = Quoridor()
    action_shape, state_shape = env.action_shape(), env.state_shape()

    model = QuoridorDQN(board_shape = state_shape[0],
                        tile_shape = state_shape[1],
                        output_shape = action_shape).to(device)
    
    state, _ = env.reset()

    output = model(torch.from_numpy(state[0]),
                   torch.from_numpy(state[1]))
