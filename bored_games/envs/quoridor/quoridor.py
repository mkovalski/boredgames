#!/usr/bin/env python

import numpy as np
import sys
import os
from PIL import Image, ImageDraw
import argparse
from .lib import set_valid_walls

from keras.layers import Input, Conv2D, Flatten, Concatenate, Dense, MaxPooling2D
from keras.layers.core import RepeatVector
from keras.losses import Huber
from keras.models import Model
from keras.optimizers import RMSprop
from tensorflow.keras.losses import Huber
import keras.backend as K

def verbose_print(string):
    if os.environ.get("VERBOSE") == 1:
        print(string)

class Player():
    '''Internal player class used by game'''
    MAX_MOVES = 10
    FACING = [1, 0, 1, 0]
    MOVEINC = [-2, -2, 2, 2]

    def __init__(self, marker, init_loc, winning_row, move_prob = None):
        assert(marker in [1, 2]), "Must be player 1 or 2"

        self.marker = marker
        self.opponent = 1 if marker == 2 else 2
        self._init_loc = init_loc
        self.curr_loc = init_loc
        self.nmoves = self.MAX_MOVES
        self.winning_row = winning_row
        self.move_prob = move_prob

    def reset(self):
        self.curr_loc = np.copy(self._init_loc)
        self.nmoves = self.MAX_MOVES

    def dec_moves(self):
        if self.nmoves > 0:
            self.nmoves -= 1

    def move(self, direction, board, depth = 0):
        '''Left Up Right Down'''

        if depth > 1:
            print("Maximum recursion depth!")
            sys.exit(2)
        
        self.curr_loc[self.FACING[direction]] += self.MOVEINC[direction]
        
        if board[self.curr_loc[0], self.curr_loc[1]] == self.opponent:
            self.move(direction, board, depth = depth + 1)

    def won(self):
        return self.curr_loc[0] == self.winning_row

class Wall():
    def __init__(self, start, mid, end):
        self.start = start
        self.mid = mid
        self.end = end

    def __eq__(self, obj):
        return (self.start == obj.start and self.end == obj.end) or \
            (self.start == obj.end and self.end == obj.start)
    
    def __hash__(self):
        return str(self.start) + str(self.end)

    def __str__(self):
        return "{}, {}".format(self.start, self.end)

class Quoridor():
    DTYPE = np.int8

    OPEN = 0
    NOWALL = -8 #np.iinfo(DTYPE).min
    WALL = 8 #np.iinfo(DTYPE).max
    
    EXPLORED = -2
    PLAYER1 = 1
    PLAYER2 = 2
    PLAYER3 = 3
    PLAYER4 = 4

    EPS = 1e-8

    MOVEDIRS = ['left', 'up', 'down', 'right']

    def __init__(self, player,
                 N = 9,
                 num_players = 2,
                 max_moves = 1000,
                 move_prob = None):
        '''
        Quoridor board game
        
        Args:
            player (int): Player num, either 1 or 2
            N (int): Size of the board
            num_players (int): Number of players, for now support 2 but 4 will be coming soon!
            max_moves (int): Maximum number of moves that can happen before termination
            move_prob (list): When a random move is chosen, probability that the player
                at that index - 1 chooses a move. For example, in a 2 player game and
                move_prob = [0.5, 0.7], player 1 chooses to move (if possible) instead of play tile
                at probability 0.5, player 2 does so at 0.7. Default does not change probability
        '''

        assert(N % 2 == 1)
        assert(num_players == 2), "Only two players for now, support for 4 coming soon!"
        assert(max_moves > 0), "Need to play at least one move"
        assert(1 <= player <= num_players), "Pick a player between 1 and {}".format(num_players)

        self.player = player
        self.N = N
        self.num_players = num_players
        self.max_moves = max_moves

        if isinstance(move_prob, list):
            assert(len(move_prob) == num_players), "Must be size of num players"
        else:
            move_prob = [move_prob] * num_players

        self.board = np.zeros((self.N*2-1, self.N*2-1), dtype = self.DTYPE)
        self.player_map = {}
        
        player1_loc = np.asarray([self.board.shape[0] - 1, self.N - 1])
        player2_loc = np.asarray([0, self.N - 1])
        
        self.player_map[self.PLAYER1] = Player(self.PLAYER1, player1_loc, 0, move_prob = move_prob[0])
        self.player_map[self.PLAYER2] = Player(self.PLAYER2, player2_loc, self.board.shape[0] - 1,
                                               move_prob = move_prob[1])
        self.wall_array, self.expanded_wall_array = self.__create_wall_array__()

        self.valid_walls = None
        self.per_move_valid_walls = None
        self.nmoves = 0
    
    def reset(self, move_opponent = True):
        '''
        Reset the environment

        Args:
            player (None, int): Do we want to change the player for the environment
            move_opponent (bool): Move the opponent with 50% probability during a reset

        '''
        
        self.done = False
        self.winner = None
        self.nmoves = 0

        # Clear board, setup wall spaces
        self.board[:] = self.OPEN
        self.board[1::2, :] = self.NOWALL
        self.board[:, 1::2] = self.NOWALL
        
        self.valid_walls = np.ones(self.max_wall_positions, dtype = np.uint8)
        self.per_move_valid_walls = np.ones(self.max_wall_positions, dtype = np.uint8)
        
        for k in self.player_map.keys():
            self.player_map[k].reset()

            self.board[self.player_map[k].curr_loc[0], 
                       self.player_map[k].curr_loc[1]] = self.player_map[k].marker
        
        # If we are a player and ask to reset, then with 50% prob opponent goes
        if move_opponent and np.random.random() > .5:
            sample = self.sample(self.opponent)
            self.move(self.opponent, sample)

        return self.get_state(), self.get_all_moves(self.player)
    
    # Attributes

    def __str__(self):
        return str(self.board)
    
    # List of properties

    @property
    def opponent(self):
        if self.player == 1:
            return 2
        return 1
    
    @property
    def max_wall_positions(self):
        '''Number of positions a wall could go into'''
        return 2 * ((self.N-1)**2)
    
    @property
    def max_positions(self):
        return self.max_wall_positions + 4

    def action_shape(self):
        return (self.max_positions,)
    
    def state_shape(self):
        return (*self.board.shape, 1), (self.num_players,)
    
    def get_tile_state(self):
        tile_state = np.asarray([self.player_map[1].nmoves,
                      self.player_map[2].nmoves], dtype = np.float32)
        
        return tile_state
    
    def get_state(self):
        board = np.copy(self.board)
        tile_state = self.get_tile_state()
        return board.reshape(*board.shape, 1), tile_state.reshape(*tile_state.shape)

    def __create_wall_array__(self):
        count = 0
        arr = [None] * self.max_wall_positions

        # Start with the vertical walls
        for i in range(0, self.board.shape[0] - 1, 2):
            for j in range(1, self.board.shape[1], 2):
                arr[count] = Wall((i, j), (i+1, j), (i+2, j))
                count += 1

        # End with some horizontal bois
        for i in range(1, self.board.shape[0], 2):
            for j in range(0, self.board.shape[1] - 1, 2):
                arr[count] = Wall((i, j), (i, j+1), (i, j+2))
                count += 1

        assert(count == self.max_wall_positions)
        
        return arr, np.asarray([[x.start, x.mid, x.end] for x in arr])
    
    def get_valid_moves(self, player = None):
        player = player if player else self.player
        opponent = 1 if player == 2 else 2

        i = self.player_map[player].curr_loc[0]
        j = self.player_map[player].curr_loc[1]
        
        # Left, up, right, down
        walls = [[i, j - 1], [i - 1, j], [i, j + 1], [i + 1, j]]
        locs = [[i, j - 2], [i - 2, j], [i, j + 2], [i + 2, j]]
        
        jump_locs = [[0, -2], [-2, 0], [0, 2], [2, 0]]

        ret = np.zeros(len(walls), dtype = np.uint8)

        for i in range(len(locs)):
            loc = locs[i]
            wall = walls[i]
            
            if 0 <= loc[0] < self.board.shape[0] and 0 <= loc[1] < self.board.shape[1] and \
                self.board[wall[0], wall[1]] == self.NOWALL and self.board[loc[0], loc[1]] != self.EXPLORED:
                
                # If we are doing this for a move, check if we can hop a player
                if self.board[tuple(loc)] == opponent:
                    test_loc = np.copy(np.asarray(loc)) + jump_locs[i] 
                    test_wall = np.copy(np.asarray(wall)) + jump_locs[i]
                    
                    if 0 <= test_loc[0] < self.board.shape[0] and 0 <= test_loc[1] < self.board.shape[1] and \
                        self.board[test_wall[0], test_wall[1]] == self.NOWALL:
                        ret[i] = 1

                else:        
                    ret[i] = 1

        return ret

    def finished(self, player, row):
        if player == self.PLAYER1:
            if row == 0:
                return True
            return False
        elif player == self.PLAYER2:
            if row == self.board.shape[0] - 1:
                return True
            return False
        else:
            print("HOW TF WE GET HERE HUH?")
            sys.exit(1)

    def get_reward(self):
        if self.done:
            if self.player == self.winner:
                return 1
            elif self.opponent == self.winner:
                return -1
            else:
                return 0
        return 0
    
    def update_walls(self):
        '''Get a list of spaces in which we are allowed to add a wall'''
        walls = []
        
        self.per_move_valid_walls[:] = 1
        
        # Once for the spaces we can't use
        for idx, wall in enumerate(self.wall_array):
            if self.valid_walls[idx] == 1:
                if self.board[wall.start] == self.WALL or self.board[wall.mid] == self.WALL or self.board[wall.end] == self.WALL:
                    self.valid_walls[idx] = 0
        
        indices = np.where(self.valid_walls == 1)[0]
        
        # Numba functions
        # TODO: C++ rewrite
        set_valid_walls(indices, self.per_move_valid_walls, self.board, 
                        self.expanded_wall_array, self.player_map[1].curr_loc,
                        self.player_map[2].curr_loc)

        self.per_move_valid_walls *= self.valid_walls
        
    def __add_wall__(self, wall):
        if self.board[wall.start] != self.NOWALL or self.board[wall.mid] != self.NOWALL or self.board[wall.end] != self.NOWALL:
            print("YUGE ERROR")
            sys.exit(2)
        self.board[wall.start] = self.WALL
        self.board[wall.mid] = self.WALL
        self.board[wall.end] = self.WALL
        
    def __remove_wall__(self, wall):
        self.board[wall.start[0], wall.start[1]] = self.NOWALL
        self.board[wall.mid[0], wall.mid[1]] = self.NOWALL
        self.board[wall.end[0], wall.end[1]] = self.NOWALL
    
    def get_all_moves(self, player):
        if self.player_map[player].nmoves == 0:
            valid_walls = np.zeros_like(self.valid_walls)
        else:
            valid_walls = self.per_move_valid_walls

        return np.append(valid_walls, self.get_valid_moves(player))

    def move_pawn(self, player, move):
        x, y = self.player_map[player].curr_loc
        self.board[x, y] = 0
        self.player_map[player].move(move, self.board)
        x, y = self.player_map[player].curr_loc
        self.board[x, y] = player
   
    def move(self, player, move):
        avail_moves = self.get_all_moves(player)
        if not self.done:
            move = move.astype(np.float32)
            
            if np.any(avail_moves == 1):
                # Mask available moves as small neg value
                move[np.where(avail_moves == 0)] = float('-inf')

                idx = np.argmax(move)

                if idx >= len(move) - 4:
                    idx = np.argmax(move[-4:])
                    verbose_print("Moving {}".format(self.MOVEDIRS[idx]))
                    self.move_pawn(player, idx)
                else:
                    verbose_print("Place a wall at {}".format(self.wall_array[idx]))
                    self.__add_wall__(self.wall_array[idx])
                    self.valid_walls[idx] = 0
                    self.update_walls()
                    self.player_map[player].dec_moves()
                
            self.nmoves += 1
            if self.nmoves == self.max_moves:
                self.done = True
                self.winner = 0
            
            # Update winner
            if self.player_map[player].won():
                self.done = True
                self.winner = player
        
        return avail_moves

    def sample(self, player = None):
        '''Only sample valid moves
        Can choose to prefer moving over putting wall with some probability
        '''

        player = player if player else self.player

        valid_moves = self.get_all_moves(player)
        
        sample = np.zeros(self.max_positions, dtype = np.uint8)
        
        all_indices = np.where(valid_moves == 1)[0]

        if len(all_indices) == 0:
            return sample
        
        idx = None

        move_prob = self.player_map[player].move_prob

        if move_prob is not None:
            move_indices = all_indices[np.where(all_indices >= self.max_wall_positions)]
            wall_indices = all_indices[np.where(all_indices < self.max_wall_positions)]

            if np.random.random() < move_prob:
                if len(move_indices) != 0:
                    idx = np.random.choice(move_indices)
                else:
                    idx = np.random.choice(wall_indices)
            else:
                if len(wall_indices) != 0:
                    idx = np.random.choice(wall_indices)
                else:
                    idx = np.random.choice(move_indices)
        else:
            idx = np.random.choice(all_indices)

        sample[idx] = 1

        return sample

    def render(self, eval_dir, image_idx):
        image = Image.new(mode='L', size=(450, 450), color=255)
        draw = ImageDraw.Draw(image)

        y_start = 0
        y_end = image.height
        step_size = int(image.width / self.N)
        
        # Flip x and y for this as pillow takes these args the opposite way
        # as a numpy block

        # Horizontal
        for idx, x in enumerate(range(step_size, image.height, step_size)):
            for idy, y in enumerate(range(0, image.width, step_size)):
                
                x_idx = (idx*2) + 1
                y_idx = (idy*2)
                
                width = 1
                if x_idx < self.board.shape[0] and self.board[x_idx, y_idx] == self.WALL:
                    width = 8

                line = ((y, x), (y + step_size, x))
                draw.line(line, fill=128, width = width)
        
        # Vertial lines + board pieces
        for idy, y in enumerate(range(step_size, image.width, step_size)):
            for idx, x in enumerate(range(0, image.height, step_size)):
                y_idx = (idy*2) + 1
                x_idx = (idx*2)

                width = 1
                if y_idx < self.board.shape[1] and self.board[x_idx, y_idx] == self.WALL:
                    width = 8
                
                line = ((y, x), (y, x + step_size))
                draw.line(line, fill=128, width = width)
        
        p1_x, p1_y = np.where(self.board == 1)
        p1_x = int(p1_x) // 2
        p1_y = int(p1_y) // 2
        
        loc = (step_size * p1_y, step_size * p1_x,
               step_size * p1_y + step_size,
               step_size * p1_x + step_size)
        draw.ellipse(loc, fill = 'blue', outline = 'blue')
        
        p2_x, p2_y = np.where(self.board == 2)
        p2_x = int(p2_x) // 2
        p2_y = int(p2_y) // 2
        
        loc = (step_size * p2_y, step_size * p2_x,
               step_size * p2_y + step_size,
               step_size * p2_x + step_size)
        draw.ellipse(loc, fill = 'red', outline = 'red')

        image.save(os.path.join(eval_dir, 'image_{}.png'.format(image_idx)), 'png')

    def step(self, action, opponent_move = True):
        avail_moves = self.move(self.player, action)
        if opponent_move:
            self.move(self.opponent, self.sample(self.opponent))
        
        curr_valid_moves = self.get_all_moves(self.player)

        return self.get_state(), curr_valid_moves, self.get_reward(), self.done, 'nada'

    def create_model(self, n_conv_layers = 3,
                     base_conv_filters = 16,
                     merge_dim = 256,
                     activation = 'relu',
                     algo = 'dqn'):

        board_shape, tile_shape = self.state_shape()

        board_input = Input(shape=board_shape,)
        tile_input = Input(shape = tile_shape,)

        inp = board_input

        for i in range(1, n_conv_layers + 1):
            filters = base_conv_filters * i
            output = Conv2D(filters = filters,
                            kernel_size = 3,
                            strides = 1,
                            padding = 'same',
                            activation = activation)(inp)
            output = MaxPooling2D()(output)

            inp = output

        sh = np.prod(output.shape.as_list()[1:])
        output = Flatten()(output)

        # Merge dim
        output = Dense(merge_dim, activation = activation)(output)

        # Tile input with a dense layer
        tile_output = Dense(merge_dim, activation = activation)(tile_input)

        # Merge layer
        output = Concatenate()([output, tile_output])

        # Two dense layers
        output = Dense(self.max_positions, activation = activation)(output)

        # Output
        output = Dense(self.max_positions)(output)

        model = Model(inputs = [board_input, tile_input],
                      outputs = output)

        model.compile(optimizer = RMSprop(learning_rate=0.0001),
                      loss = Huber(),
                      metrics = ['accuracy'])

        return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type = int, default = 9)
    parser.add_argument('--move_prob', type = float, default = None)
    args = parser.parse_args()

    game = Quoridor(N = args.N, )

    for i in range(100):
        player = 2
        game.reset(player)

        while not game.done:
            sample = game.sample(player)
            game.move(player, sample)
            player = max((player + 1) % 3, 1)
            
        print(game.board)