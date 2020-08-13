#!/usr/bin/env python

import numpy as np
import sys
import os
from PIL import Image, ImageDraw
import argparse
from bored_games.envs.quoridor.lib import set_valid_walls
from termcolor import colored, cprint

def verbose_print(string):
    if os.environ.get('VERBOSE', '0') == '1':
        print(string)

class Player():
    '''Internal player class used by game'''
    MAXMOVES = 10
    FACING = [1, 0, 1, 0]
    MOVEINC = [-2, -2, 2, 2]
    OPPONENT = 2

    def __init__(self, init_loc):
        self._init_loc = init_loc
        self.curr_loc = init_loc
        self.nmoves = self.MAXMOVES

    def reset(self):
        self.curr_loc = np.copy(self._init_loc)
        self.nmoves = self.MAXMOVES

    def dec_moves(self):
        if self.nmoves > 0:
            self.nmoves -= 1

    def move(self, direction, board, depth = 0):
        '''Left Up Right Down'''

        if depth > 1:
            print("Maximum recursion depth!")
            sys.exit(2)
        
        self.curr_loc[self.FACING[direction]] += self.MOVEINC[direction]
        
        # Hop over opponent
        if board[self.curr_loc[0], self.curr_loc[1]] == self.OPPONENT:
            self.move(direction, board, depth = depth + 1)

    def opponent_view(self, N):
        '''Which position does my opponent see me at'''
        pt = (N * 2) - 2
        return pt - (self.curr_loc)

    def won(self):
        if self.curr_loc[0] == 0:
            return True
        return False

class Wall():
    def __init__(self, start, mid, end):
        '''Wall to play on quoridor board'''
        self.initial_values = (start, mid, end)
        self.start = start
        self.mid = mid
        self.end = end

    def reset(self):
        self.start, self.mid, self.end = self.initial_values

    def __eq__(self, obj):
        return (self.start == obj.start and self.end == obj.end) or \
            (self.start == obj.end and self.end == obj.start)
    
    def __hash__(self):
        return str(self.start) + str(self.end)

    def __str__(self):
        return "{}, {}".format(self.start, self.end)
    
    def rot180(self, N):
        '''Rotate ordering of a wall 180 degress'''

        pt = 2 * (N - 1)

        self.start, self.end = \
            (pt - self.end[0], pt - self.end[1]), (pt - self.start[0], pt - self.start[1])

        self.mid = (pt - self.mid[0], pt - self.mid[1])

class Quoridor():
    
    OPEN = 0
    NOWALL = -8
    WALL = 8
    
    DTYPE = np.int8

    PLAYER1 = 1
    PLAYER2 = 2

    NUMPLAYERS = 2

    MOVEDIRS = ['left', 'up', 'down', 'right']

    # For ascii rendering
    COLORS = {PLAYER1: 'blue',
              PLAYER2: 'magenta',
              OPEN: 'grey',
              WALL: 'red',
              NOWALL: 'green'}
    
    def __init__(self, 
                 N = 9,
                 max_moves = 1000,
                 set_move_prob = False,
                 normalize = True):
        '''
        Quoridor board game
        
        Args:
            N (int): Size of the board, must be an odd number
            max_moves (int): Maximum number of moves that can happen before considered a draw
            set_move_prob (bool): Allow environment to set a probability of moving pawn vs
                placing a wall. Discrepency is pretty large (IE 128 walls to play vs 4 moves)
                so try to balance exploration
            normalize (bool): Use scaling of -1 to 1 for states

        '''

        assert(N % 2 == 1)
        assert(max_moves > 0), "Need to play at least one move"
        assert(isinstance(normalize, bool))

        self.N = N
        self.max_moves = max_moves
        self.set_move_prob = set_move_prob
        self.normalize = normalize
        
        self.move_prob = None
        
        # Internal states
        self.board = np.zeros((self.N*2-1, self.N*2-1), dtype = self.DTYPE)
        self.tile_state = np.zeros(self.NUMPLAYERS, dtype = np.float32)
        
        # Initial player location relative to player
        player_loc = np.asarray([self.board.shape[0] - 1, self.N - 1])
        self.player_array = [Player(player_loc), Player(player_loc)]
        
        # Array of wall positions
        self.wall_array, self.expanded_wall_array = self.__create_wall_array__()
        self.valid_walls = np.ones_like(self.wall_array, dtype = np.uint8)
        self.per_move_valid_walls = np.ones_like(self.wall_array, dtype = np.uint8)

        self.nmoves = 0
        self.winner = None
    
    def reset(self, move_opponent = True):
        '''
        Reset the environment

        Args:
            move_opponent (bool): During a reset, if set to True, makes the opponent
                move first with probability of 50%

        '''
        
        self.done = False
        self.winner = np.asarray([0] * self.NUMPLAYERS)
        self.nmoves = 0

        # Clear board, setup wall spaces
        self.board[:] = self.OPEN
        self.board[1::2, :] = self.NOWALL
        self.board[:, 1::2] = self.NOWALL
        
        # Reset walls
        # Valid walls is for all walls to be played
        # Per move is those allowed per turn
        self.valid_walls[:] = 1
        self.per_move_valid_walls[:] = 1

        for wall in self.wall_array: 
            wall.reset()
        
        [x.reset() for x in self.player_array]
        self.board[self.player_array[0].curr_loc[0], self.player_array[0].curr_loc[1]] = 1
        opponent_view = self.player_array[1].opponent_view(self.N)
        self.board[opponent_view[0], opponent_view[1]] = 2

        self.tile_state[0], self.tile_state[1] = self.player_array[0].MAXMOVES, self.player_array[1].MAXMOVES

        # Set the probability of a move vs a wall set
        if self.set_move_prob:
            self.move_prob = np.random.random(self.NUMPLAYERS)
        
        # Return the base state, neither player moved
        return self.get_state(), self.get_valid_actions()
    
    # Attributes
    def __str__(self):
        return str(self.board)
    
    # List of properties

    @property
    def max_wall_positions(self):
        '''Number of positions a wall could go into'''
        return 2 * ((self.N-1)**2)
    
    @property
    def max_positions(self):
        return self.max_wall_positions + 5

    def action_shape(self):
        return (self.max_positions,)
    
    def state_shape(self):
        return (1, *self.board.shape), (self.num_players,)
    
    def get_tile_state(self):
        tile_state = np.copy(self.tile_state)
        if self.normalize:
            tile_state /= self.player_array[0].MAXMOVES
            tile_state *= 2
            tile_state -= 1
        
        return tile_state
    
    def get_state(self):
        '''Get the full state of the environment, includes board and number of tiles remaining'''
        board = np.copy(self.board)
        
        if self.normalize:
            board = 2 * ((board - self.NOWALL) / (self.WALL - self.NOWALL)) - 1

        tile_state = self.get_tile_state()

        return board.reshape(1, *board.shape).astype(np.float32), \
            tile_state.reshape(*tile_state.shape).astype(np.float32)

    def __create_wall_array__(self):
        '''Create an array of walls with all valid positions that is easily reversible'''
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

        #assert(count == self.max_wall_positions)
        # TODO: Keep asserts arround to remember for writing test
        
        return arr, np.asarray([[x.start, x.mid, x.end] for x in arr])
    
    def get_valid_moves(self):
        '''Determine which direction a player is able to move'''

        opponent = 2
        
        i, j = self.player_array[0].curr_loc

        # Left, up, right, down
        walls = [[i, j - 1], [i - 1, j], [i, j + 1], [i + 1, j]]
        locs = [[i, j - 2], [i - 2, j], [i, j + 2], [i + 2, j]]
        
        jump_locs = [[0, -2], [-2, 0], [0, 2], [2, 0]]

        ret = np.zeros(len(walls), dtype = np.uint8)

        for idx in range(len(locs)):
            loc = locs[idx]
            wall = walls[idx]
            
            # If we can't move
            if loc[0] < 0 or loc[0] >= self.board.shape[0] or loc[1] < 0 or loc[1] >= self.board.shape[1] or \
                self.board[wall[0], wall[1]] == self.WALL:
                continue
            
            # If we are doing this for a move, check if we can hop a player
            if self.board[tuple(loc)] == opponent:
                
                test_loc = np.copy(np.asarray(loc)) + jump_locs[idx] 
                test_wall = np.copy(np.asarray(wall)) + jump_locs[idx]
                
                if test_loc[0] < 0 or test_loc[0] >= self.board.shape[0] or test_loc[1] < 0 or test_loc[1] >= self.board.shape[1] or \
                    self.board[test_wall[0], test_wall[1]] == self.WALL:
                        continue

                ret[idx] = 1

            else:        
                ret[idx] = 1

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
        # TODO
        if self.winner[0] == 0:
            return 0

        if self.winner[0] == 1:
            return 1
        return -1

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
                        self.expanded_wall_array, self.player_array[0].curr_loc,
                        self.player_array[1].curr_loc)

        self.per_move_valid_walls &= self.valid_walls
        
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
    
    def get_valid_walls(self):
        if self.player_array[0].nmoves == 0:
            valid_walls = np.zeros_like(self.wall_array)
        else:
            valid_walls = self.per_move_valid_walls

        return valid_walls
        
    def get_valid_actions(self):
        actions = np.concatenate([self.get_valid_walls(),
                                  self.get_valid_moves(),
                                  [0]]).astype(np.uint8)
        
        # TODO: Speed up case where no actions are available
        if not np.any(actions[0:-1]):
            actions[-1] = 1

        return actions

    def move_pawn(self, move):
        '''Moves the player, arg checking done outside function'''
        # Open up a space
        x, y = self.player_array[0].curr_loc
        self.board[x, y] = self.OPEN

        # Move to new space
        self.player_array[0].move(move, self.board)
        x, y = self.player_array[0].curr_loc
        self.board[x, y] = self.PLAYER1
    
    def move(self, move):
        avail_moves = self.get_valid_actions()
        if not self.done:
            if avail_moves[-1] != 1:
                move = move.astype(np.float32)
                # Mask available moves as small neg value
                move[np.where(avail_moves == 0)] = float('-inf')

                idx = np.argmax(move)
                
                # Wall addition
                if idx < self.max_wall_positions:
                    verbose_print("Place a wall at {}".format(self.wall_array[idx]))
                    self.__add_wall__(self.wall_array[idx])
                    self.wall_array[idx].valid = False
                    self.update_walls()
                    self.player_array[0].dec_moves()
                else:
                    idx = np.argmax(move[self.max_wall_positions:-1])
                    verbose_print("Moving {}".format(self.MOVEDIRS[idx]))
                    self.move_pawn(idx)
                
            self.nmoves += 1
            if self.nmoves == self.max_moves:
                self.done = True
                self.winner[0] = 0
            
            # Update winner
            if self.player_array[0].won():
                self.done = True
                self.winner[0] = 1
                self.winner[1] = -1
        
        return avail_moves

    def sample(self):
        '''Only sample valid moves
        Can choose to prefer moving over putting wall with some probability
        '''

        valid_walls = self.get_valid_walls()
        valid_actions = self.get_valid_moves()
        
        valid_moves = np.concatenate([valid_walls, valid_actions])
        
        sample = np.zeros(self.max_positions, dtype = np.uint8)
        
        all_indices = np.where(valid_moves == 1)[0]

        if len(all_indices) == 0:
            # Do nothing move
            sample[-1] = 1
            return sample
        
        idx = None
        if self.set_move_prob:
            wall_indices = np.where(valid_walls == 1)[0]
            move_indices = np.where(valid_actions == 1)[0]

            # Move instead of selecting a wall
            if np.random.random() < self.move_prob[0]:
                if len(move_indices) != 0:
                    idx = np.random.choice(move_indices) + len(valid_walls)
                else:
                    idx = np.random.choice(wall_indices)
            else:
                if len(wall_indices) != 0:
                    idx = np.random.choice(wall_indices)
                else:
                    idx = np.random.choice(move_indices) + len(valid_walls)
        else:
            # Choice out of all available indices
            idx = np.random.choice(all_indices)

        sample[idx] = 1

        return sample

    def render_ascii(self, flip = False):
        '''Small ascii rendering of current board state'''
        
        if flip:
            board = self.swap_board_view()
        else:
            board = self.board

        for i in range(board.shape[0]):
            line = ''
            if i % 2 == 0:
                line += '  '
            for j in range(board.shape[1]):
                color = self.COLORS[board[i,j]]
                if board[i,j] in set([self.WALL, self.NOWALL]):
                    if i % 2 == 0:
                        board_char = '  |  '
                    else:
                        board_char = '-----'
                        if j % 2 == 1:
                            board_char = '-'
                            color = 'grey'
                    line += colored(board_char, color)
                else:
                    line += colored(board[i,j], color)
            print(line)
        
        print('\n\n')

    def render(self, eval_dir, image_idx):
        '''Using pillow'''
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
    
    def swap_board_view(self):
        curr_i, curr_j = self.player_array[0].curr_loc
        dest_i, dest_j = self.player_array[1].opponent_view(self.N)

        # For now
        assert(self.board[curr_i, curr_j] == 1)
        assert(self.board[dest_i, dest_j] == 2)
        
        board = np.copy(self.board)
        board[curr_i, curr_j], board[dest_i, dest_j] = \
            board[dest_i, dest_j], board[curr_i, curr_j]

        # Rotate the board
        return np.rot90(np.rot90(board))

    def __swap(self):
        '''Update the interface of the board.
            - Rotate the board
            - Swap tile state
            - Reverse walls

        '''
        # Swap the players
        curr_i, curr_j = self.player_array[0].curr_loc
        dest_i, dest_j = self.player_array[1].opponent_view(self.N)

        # For now
        assert(self.board[curr_i, curr_j] == 1)
        assert(self.board[dest_i, dest_j] == 2)

        self.board[curr_i, curr_j], self.board[dest_i, dest_j] = \
            self.board[dest_i, dest_j], self.board[curr_i, curr_j]

        # Rotate the board
        self.board = np.rot90(np.rot90(self.board))

        # Tile state
        self.tile_state = self.tile_state[::-1]
        
        # Rotate walls
        self.__rotate_wall_array(self.valid_walls)
        self.__rotate_wall_array(self.per_move_valid_walls)

        # Insert limp bizkit line
        self.winner = np.roll(self.winner, -1)
        self.player_array = np.roll(self.player_array, -1)
        if self.set_move_prob:
            self.move_prob = np.roll(self.move_prob, -1)
   
    def __rotate_wall_array(self, wall_array):
        # Swap the wall array
        nitems = len(wall_array) // 2
        wall_array[0:nitems] = wall_array[nitems-1::-1]
        wall_array[nitems:] = wall_array[len(wall_array):nitems-1:-1]
        
    def step(self, action):
        self.move(action)
        
        # Update the board
        self.__swap()
        
        return self.get_state(), self.get_valid_actions(), self.get_reward(), self.done, 'nada'

if __name__ == '__main__':
    from tqdm import tqdm
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type = int, default = 9,
        help = "Shape of the board")
    args = parser.parse_args()

    game = Quoridor(N = args.N, set_move_prob = True)
    flip = True

    for i in tqdm(range(1)):
        # Player 1 goes first
        state, valid_actions = game.reset()
        game.render_ascii()

        while not game.done:
            sample = game.sample()
            game.step(sample)
            game.render_ascii(flip)

            if flip:
                flip = False
            else:
                flip = True

        game.render_ascii(flip)
