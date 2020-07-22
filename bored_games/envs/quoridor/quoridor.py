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
    MAX_MOVES = 10
    FACING = [1, 0, 1, 0]
    MOVEINC = [-2, -2, 2, 2]

    def __init__(self, marker, init_loc, winning_row):
        assert(marker in [1, 2]), "Must be player 1 or 2"

        self.marker = marker
        self.opponent = 1 if marker == 2 else 2
        self._init_loc = init_loc
        self.curr_loc = init_loc
        self.nmoves = self.MAX_MOVES
        self.winning_row = winning_row

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
        '''Wall to play on quoridor board'''
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

    OPEN = 0
    NOWALL = -8
    WALL = 8
    
    EXPLORED = -2 # For DFS

    DTYPE = np.int8

    PLAYER1 = 1
    PLAYER2 = 2
    PLAYER3 = 3
    PLAYER4 = 4

    MOVEDIRS = ['left', 'up', 'down', 'right']

    # For ascii rendering
    COLORS = {PLAYER1: 'blue',
              PLAYER2: 'magenta',
              PLAYER3: 'cyan',
              PLAYER4: 'white',
              OPEN: 'grey',
              WALL: 'red',
              NOWALL: 'green'}
    
    def __init__(self, player = 1,
                 N = 9,
                 num_players = 2,
                 max_moves = 1000,
                 set_move_prob = False):
        '''
        Quoridor board game
        
        Args:
            player (int): Player num, either 1 or 2
            N (int): Size of the board, must be an odd number
            num_players (int): Number of players, currently support for 2
            max_moves (int): Maximum number of moves that can happen before considered a draw
            set_move_prob (bool): Allow environment to set a probability of moving pawn vs
                placing a wall. Discrepency is pretty large (IE 128 walls to play vs 4 moves)
                so try to balance exploration

        '''

        assert(N % 2 == 1)
        assert(num_players == 2), "Only two players for now, support for 4 coming soon!"
        assert(max_moves > 0), "Need to play at least one move"
        assert(1 <= player <= num_players), "Pick a player between 1 and {}".format(num_players)

        self.player = player
        self.N = N
        self.num_players = num_players
        self.max_moves = max_moves
        self.set_move_prob = set_move_prob
        self.move_prob = None
        
        self.board = np.zeros((self.N*2-1, self.N*2-1), dtype = self.DTYPE)
        self.player_map = {}
        
        player1_loc = np.asarray([self.board.shape[0] - 1, self.N - 1])
        player2_loc = np.asarray([0, self.N - 1])
        
        self.player_map[self.PLAYER1] = Player(self.PLAYER1, player1_loc, 0)
        self.player_map[self.PLAYER2] = Player(self.PLAYER2, player2_loc, self.board.shape[0] - 1)
        self.wall_array, self.expanded_wall_array = self.__create_wall_array__()

        self.valid_walls = None
        self.per_move_valid_walls = None
        self.nmoves = 0
    
    def reset(self, move_opponent = True, move_priority = None):
        '''
        Reset the environment

        Args:
            move_opponent (bool): During a reset, if set to True, makes the opponent
                move first with probability of 50%

        '''
        
        self.done = False
        self.winner = None
        self.nmoves = 0
        self.move_priority = move_priority

        # Clear board, setup wall spaces
        self.board[:] = self.OPEN
        self.board[1::2, :] = self.NOWALL
        self.board[:, 1::2] = self.NOWALL
        
        # Reset walls
        self.valid_walls = np.ones(self.max_wall_positions, dtype = np.uint8)
        self.per_move_valid_walls = np.ones(self.max_wall_positions, dtype = np.uint8)
        
        # Reset players, mark board with players
        for k in self.player_map.keys():
            self.player_map[k].reset()
            self.board[self.player_map[k].curr_loc[0], 
                       self.player_map[k].curr_loc[1]] = self.player_map[k].marker
        
        if self.set_move_prob:
            self.move_prob = np.random.random(self.num_players)

        # If we are a player and ask to reset, then with 50% prob opponent goes
        if move_opponent and np.random.random() > .5:
            sample = self.sample(self.opponent)
            self.move(self.opponent, sample)

        return self.get_state(), self.get_valid_actions(self.player)
    
    # Attributes
    def __str__(self):
        return str(self.board)
    
    # List of properties

    @property
    def opponent(self):
        return (self.player % self.num_players) + 1
    
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
        tile_state = np.asarray([self.player_map[1].nmoves,
                      self.player_map[2].nmoves], dtype = np.float32)
        
        # Normalize -1 to 1
        tile_state /= self.player_map[1].MAX_MOVES
        tile_state *= 2
        tile_state -= 1
        
        return tile_state
    
    def get_state(self):
        board = np.copy(self.board)
        # Normalize -1 to 1
        board = 2 * ((board - self.NOWALL) / (self.WALL - self.NOWALL)) - 1

        tile_state = self.get_tile_state()
        return board.reshape(1, *board.shape).astype(np.float32), \
            tile_state.reshape(*tile_state.shape).astype(np.float32)

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
    
    def get_valid_walls(self, player):
        if self.player_map[player].nmoves == 0:
            valid_walls = np.zeros_like(self.valid_walls)
        else:
            valid_walls = self.per_move_valid_walls

        return valid_walls
        
    def get_valid_actions(self, player):
        actions = np.append(self.get_valid_walls(player), self.get_valid_moves(player))
        actions = np.append(actions, 0)

        if not np.any(actions[0:-1]):
            actions[-1] = 1
        return actions

    def move_pawn(self, player, move):
        x, y = self.player_map[player].curr_loc
        self.board[x, y] = 0
        self.player_map[player].move(move, self.board)
        x, y = self.player_map[player].curr_loc
        self.board[x, y] = player
   
    def move(self, player, move):
        avail_moves = self.get_valid_actions(player)
        if not self.done:
            move = move.astype(np.float32)
            
            if avail_moves[-1] != 1:
                # Mask available moves as small neg value
                move[np.where(avail_moves == 0)] = float('-inf')

                idx = np.argmax(move)

                if idx < self.max_wall_positions:
                    verbose_print("Place a wall at {}".format(self.wall_array[idx]))
                    self.__add_wall__(self.wall_array[idx])
                    self.valid_walls[idx] = 0
                    self.update_walls()
                    self.player_map[player].dec_moves()
                else:
                    idx = np.argmax(move[self.max_wall_positions:-1])
                    verbose_print("Moving {}".format(self.MOVEDIRS[idx]))
                    self.move_pawn(player, idx)
            else:
                verbose_print("No moves to make!")

                
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

        valid_walls = self.get_valid_walls(player)
        valid_actions = self.get_valid_moves(player)
        
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
            if np.random.random() < self.move_prob[player-1]:
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

    def render_ascii(self):
        lines = ''

        for i in range(self.board.shape[0]):
            line = ''
            if i % 2 == 0:
                line += '  '
            for j in range(self.board.shape[1]):
                color = self.COLORS[self.board[i,j]]
                if self.board[i,j] in set([self.WALL, self.NOWALL]):
                    if i % 2 == 0:
                        board_char = '  |  '
                    else:
                        board_char = '-----'
                        if j % 2 == 1:
                            board_char = '-'
                            color = 'grey'
                    line += colored(board_char, color)
                else:
                    line += colored(self.board[i,j], color)

            print(line + '\n')

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
        self.move(self.player, action)
        if opponent_move:
            self.move(self.opponent, self.sample(self.opponent))
        
        curr_valid_moves = self.get_valid_actions(self.player)

        return self.get_state(), curr_valid_moves, self.get_reward(), self.done, 'nada'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type = int, default = 9,
        help = "Shape of the board")
    args = parser.parse_args()

    player = 1
    game = Quoridor(player = player, N = args.N, )
    
    # Player 1 goes first
    game.reset(move_opponent = False)

    while not game.done:
        sample = game.sample(player)
        game.move(player, sample)
        player = (player % 2) + 1
       
    game.render_ascii()
