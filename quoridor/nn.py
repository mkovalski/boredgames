#!/usr/bin/env python

import numpy as np
from keras.layers import Input, Conv2D, Flatten, Add, Dense
from keras.layers.core import RepeatVector
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K
from quoridor import Quoridor

class NN():
    def __init__(self, board_shape, tile_shape, output_shape,
                 n_conv_layers = 2,
                 base_conv_filters = 4,
                 merge_dim = 256,
                 activation = 'relu'):

        board_input = Input(shape=board_shape,)
        tile_input = Input(shape = tile_shape,)

        inp = board_input


        for i in range(1, n_conv_layers + 1):
            filters = base_conv_filters ** i
            output = Conv2D(filters = filters,
                            kernel_size = 5, 
                            strides = 1,
                            padding = 'valid',
                            activation = activation)(inp)
            inp = output

        
        sh = np.prod(output.shape.as_list()[1:])
        output = Flatten()(output)
        
        # Merge dim
        output = Dense(merge_dim, activation = activation)(output)
        
        # Tile input
        tile_output = RepeatVector(merge_dim // tile_shape[0])(tile_input)
        tile_output = Flatten()(tile_output)
        
        # Merge layer
        output = Add()([output, tile_output])
        
        # Two dense layers
        output = Dense(output_shape[0], activation = activation)(output)
            
        # Output
        output = Dense(output_shape[0])(output)
        
        self.model = Model(inputs = [board_input, tile_input],
                           outputs = output)
        self.model.compile(optimizer = RMSprop(learning_rate=0.0001),
                           loss = 'mse',
                           metrics = ['accuracy'])
    
    def get_model(self):
        return self.model

if __name__ == '__main__':
    game = Quoridor()
    board_shape, tile_shape = game.state_shape()
    action_shape = game.action_shape()

    nn = NN(board_shape,
            tile_shape,
            action_shape)
                        
            
        
        
