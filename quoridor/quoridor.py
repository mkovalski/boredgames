#!/usr/bin/env python

import numpy as np
from copy import copy
import sys
import os
from PIL import Image, ImageDraw
import argparse
import time
import multiprocessing as mp

# TODO: Speed up by writing in C++
# DFS for each move is the bottleneck

def verbose_print(string):
    if os.environ.get("VERBOSE") == 1:
        print(string)

class Player():
    '''Internal player class used by game'''
    MAX_MOVES = 10

    def __init__(self, marker, init_loc, winning_row):
        assert(marker in [1, 2]), "Must be player 1 or 2"
        self.marker = marker
        self.opponent = 1 if marker == 2 else 2
        self._init_loc = init_loc
        self.curr_loc = init_loc
        self.nmoves = self.MAX_MOVES
        self.winning_row = winning_row
        self.move_prob = None

    def reset(self, move_prob = None):
        self.curr_loc = copy(self._init_loc)
        self.nmoves = self.MAX_MOVES
        self.move_prob = move_prob
        if self.move_prob is not None and self.move_prob < 0.1:
            self.move_prob = None

    def dec_moves(self):
        if self.nmoves > 0:
            self.nmoves -= 1

    def move(self, direction, board, depth = 0):
        '''L U R D'''
        assert(0<=direction<4)
        if depth > 1:
            print("Maximum recursion depth!")
            sys.exit(2)

        if direction == 0:
            self.curr_loc[1] -= 2
        elif direction == 1:
            self.curr_loc[0] -= 2
        elif direction == 2:
            self.curr_loc[1] += 2
        elif direction == 3:
            self.curr_loc[0] += 2
        else:
            print("How did we get here?")
            sys.exit(2)
        
        if board[self.curr_loc[0], self.curr_loc[1]] == self.opponent:
            #print("A JUMP MOVE!")

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
    '''Let's play some quoridor'''
    
    DTYPE = np.int8

    OPEN = 0
    NOWALL = -9 #np.iinfo(DTYPE).min
    WALL = 9 #np.iinfo(DTYPE).max
    
    EXPLORED = -2
    PLAYER1 = 1
    PLAYER2 = 2

    EPS = 1e-8

    MOVEDIRS = ['left', 'up', 'down', 'right']

    def __init__(self, N = 9):
        assert(N % 2 == 1)
        self.N = N

        self.board = np.zeros((self.N*2-1, self.N*2-1), dtype = self.DTYPE)
        self.player_map = {}
        
        player1_loc = [self.board.shape[0] - 1, self.N - 1]
        player2_loc = [0, self.N - 1]
        
        self.player_map[self.PLAYER1] = Player(self.PLAYER1, player1_loc, 0)
        self.player_map[self.PLAYER2] = Player(self.PLAYER2, player2_loc, self.board.shape[0] - 1)

        self.wall_array = self.create_wall_array()

        self.valid_walls = None
        self.per_move_valid_walls = None

        # Renderings

        self.reset()
    
    def reset(self, player = None, move_prob = None):
        self.done = False
        self.winner = None

        if not isinstance(move_prob, list):
            move_prob = [move_prob]

        # Clear board, setup wall spaces
        self.board[:] = self.OPEN
        self.board[1::2, :] = self.NOWALL
        self.board[:, 1::2] = self.NOWALL
        
        self.valid_walls = np.ones(self.max_wall_positions, dtype = np.uint8)
        self.per_move_valid_walls = np.ones(self.max_wall_positions, dtype = np.uint8)
        
        for k in self.player_map.keys():
            player = self.player_map[k]
            player.reset(move_prob[k-1] if len(move_prob) > 1 else move_prob[0])
            self.board[player.curr_loc[0], player.curr_loc[1]] = player.marker
        
        # If we are a player and ask to reset, then with 50% prob opponent goes
        if player is not None and np.random.random() > .5:
            opponent = 1 if player == 2 else 2
            sample = self.sample(opponent)
            self.move(opponent, sample)
            
        return self.get_state(1)

    def action_shape(self):
        return (self.max_positions,)

    def state_shape(self):
        return (*self.board.shape, 1), (2,)
    
    def get_tile_state(self, player):
        opponent = 1 if player == 2 else 1

        tile_state = np.asarray([self.player_map[player].nmoves,
                      self.player_map[opponent].nmoves], dtype = np.float32)
        
        tile_state /= float(self.player_map[player].MAX_MOVES)

        return tile_state
        
    def get_state(self, player, normalize = True):
        '''Get the state for the player'''
        
        board = np.copy(self.board.reshape(self.board.shape[0], self.board.shape[1], 1))
        board = (board - np.iinfo(self.DTYPE).min) / (np.iinfo(self.DTYPE).max - np.iinfo(self.DTYPE).min)
        
        tile_state = self.get_tile_state(player)

        return board, tile_state

    def create_wall_array(self):
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
        return arr
    
    @property
    def max_wall_positions(self):
        '''Number of positions a wall could go into'''
        return 2 * ((self.N-1)**2)
    
    @property
    def max_positions(self):
        return self.max_wall_positions + 4

    def __str__(self):
        return str(self.board)

    def can_complete(self, player):
        self.complete_cache = {}
        board = np.copy(self.board)

        return self.dfs(self.player_map[player].curr_loc[0],
                        self.player_map[player].curr_loc[1],
                        player, board)
    
    def dfs(self, i, j, player, board):
        if (i, j) in self.complete_cache:
            return self.complete_cache[(i, j)]

        if self.finished(player, i):
            self.complete_cache[(i, j)] = True
            return True
        
        prev = board[i, j]
        board[i, j] = self.EXPLORED
        
        for x, y in self.get_neighbors(i, j, board = board)[0]:
            if self.dfs(x, y, player, board):
                board[i, j] = prev
                self.complete_cache[(i, j)] = True
                return True
        
        board[i, j] = prev
        self.complete_cache[(i, j)] = False
        return False
    
    def get_neighbors(self, i, j, board, player = None):
        # TODO: Player hops and trap setting
        # Left, up, right, down

        walls = [[i, j - 1], [i - 1, j], [i, j + 1], [i + 1, j]]
        locs = [[i, j - 2], [i - 2, j], [i, j + 2], [i + 2, j]]
        
        jump_locs = [[0, -2], [-2, 0], [0, 2], [2, 0]]

        ret = np.zeros(len(walls), dtype = np.uint8)
        ret_locs = []

        for i in range(len(locs)):
            loc = locs[i]
            wall = walls[i]
            
            if 0 <= loc[0] < self.board.shape[0] and 0 <= loc[1] < self.board.shape[1] and \
                self.board[wall[0], wall[1]] == self.NOWALL and board[loc[0], loc[1]] != self.EXPLORED:
                
                # If we are doing this for a move, check if we can hop a player
                if player != None:
                    opponent = 1 if player == 2 else 2

                    if self.board[tuple(loc)] == opponent:
                        test_loc = np.copy(np.asarray(loc)) + jump_locs[i] 
                        test_wall = np.copy(np.asarray(wall)) + jump_locs[i]
                        
                        if 0 <= test_loc[0] < self.board.shape[0] and 0 <= test_loc[1] < self.board.shape[1] and \
                            self.board[test_wall[0], test_wall[1]] == self.NOWALL:
                            ret[i] = 1
                            ret_locs.append(loc)

                    else:
                        ret[i] = 1
                        ret_locs.append(loc)
                else:        
                    ret[i] = 1
                    ret_locs.append(loc)

        return ret_locs, ret

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

    def get_reward(self, player):
        if self.done:
            return 2 * int(player == self.winner) - 1
        return 0
    
    def update_walls(self):
        '''Get a list of spaces in which we are allowed to add a wall'''
        walls = []
        
        self.per_move_valid_walls[:] = 1
        
        # TODO: Optimize
        # Once for the 
        for idx, wall in enumerate(self.wall_array):
            if self.valid_walls[idx] == 1:
                if self.board[wall.start] == self.WALL or self.board[wall.mid] == self.WALL or self.board[wall.end] == self.WALL:
                    self.valid_walls[idx] = 0
        
        # Multiprocessing
        indices = np.where(self.valid_walls == 1)[0]
        with mp.Pool(8) as pool:
            res = pool.map(self.is_valid_wall, list(indices))

        self.per_move_valid_walls[indices] = res
        
        '''
        for idx, wall in enumerate(self.wall_array):
            if self.valid_walls[idx] == 1:
                tmpWall = self.wall_array[idx]
                self.__add_wall__(tmpWall)
                if not all(self.can_complete(k) for k in self.player_map.keys()):
                    self.per_move_valid_walls[idx] = 0
                self.__remove_wall__(tmpWall)
        '''
        self.per_move_valid_walls *= self.valid_walls
    
    def is_valid_wall(self, idx):
        res = 1
        tmpWall = self.wall_array[idx]
        self.__add_wall__(tmpWall)
        if not all(self.can_complete(k) for k in self.player_map.keys()):
            res = 0
            #self.per_move_valid_walls[idx] = 0
        self.__remove_wall__(tmpWall)
        return res

    def __add_wall__(self, wall):
        if self.board[wall.start] != self.NOWALL or self.board[wall.end] != self.NOWALL:
            print("YUGE ERROR")
            sys.exit(2)
        self.board[wall.start] = self.WALL
        self.board[wall.mid] = self.WALL
        self.board[wall.end] = self.WALL
        
    def __remove_wall__(self, wall):
        self.board[wall.start[0], wall.start[1]] = self.NOWALL
        self.board[wall.mid[0], wall.mid[1]] = self.NOWALL
        self.board[wall.end[0], wall.end[1]] = self.NOWALL
    
    def get_valid_moves(self, player):
        return self.get_neighbors(self.player_map[player].curr_loc[0],
                                  self.player_map[player].curr_loc[1],
                                  player = player,
                                  board = self.board)[-1]

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
        if not self.done:
            move = move.astype(np.float32)
            move += self.EPS
            avail_moves = self.get_all_moves(player)
            
            if not np.any(avail_moves == 1):
                #print("You're trapped buddy")
                return

            # Mask available moves as 0, add epsilon, renormalize
            move *= avail_moves
            idx = np.argmax(move)

            if idx >= len(move) - 4:
                idx = np.argmax(move[-4:])
                verbose_print("Moving {}".format(self.MOVEDIRS[idx]))
                self.move_pawn(player, idx)
            else:
                verbose_print("Place a wall at {}".format(self.wall_array[idx]))
                self.__add_wall__(self.wall_array[idx])
                self.valid_walls[idx] = 0
                if self.player_map[player].nmoves != 0:
                    self.update_walls()
                self.player_map[player].dec_moves()
            
            # Update winner
            if self.player_map[player].won():
                self.done = True
                self.winner = player
                #print("Player {} wins!".format(self.winner))
        else:
            pass
            #print("Game complete! Please reset")
    
    def move_opponent(self, opponent, direction = None, wall = None):
        if not direction and not wall:
            print("didn't do anything...")
            return

        sample = np.zeros(self.max_positions, dtype = np.uint8) 
        if direction:
            idx = self.MOVEDIRS.find(direction.lower())
            idx += self.max_wall_positions
        else:
            idx = None
            wall = Wall(wall[0], wall[1])
            for i, w in enumerate(self.wall_array):
                if wall == w:
                    idx = i
                    break
            if idx == None:
                print("What happened?")
                return

        sample[idx] = 1

        self.move(opponent, sample)

    def sample(self, player):
        '''Only sample valid moves
        Can choose to prefer moving over putting wall with some probability
        '''
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

    def step(self, player, action):
        opponent = 1 if player == 2 else 2

        self.move(player, action)
        self.move(opponent, self.sample(opponent))

        return self.get_state(player), self.get_reward(player), self.done, 'nada'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type = int, default = 9)
    parser.add_argument('--move_prob', type = float, default = None)
    args = parser.parse_args()

    game = Quoridor(N = args.N, )
    game.reset()
    
    player = 1

    while not game.done:
        sample = game.sample(player)
        game.move(player, sample)
        # Ew
        player += 1
        if player >= 3:
            player = 1
        
    print(game.board)
