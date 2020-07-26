#!/usr/bin/env python
import numpy as np
from numba import jit, njit, prange
import time

@njit(parallel = True)
def set_valid_walls(indices, valid_walls, board, wall_array, player1_loc, player2_loc):
    for i in prange(len(indices)):
        idx = indices[i]
        new_board = np.copy(board)
        cache = np.ones(board.shape, dtype = np.int8) * -1

        add_wall(wall_array[idx], new_board)
        
        if dfs(player1_loc[0], player1_loc[1], new_board, 0, cache) == 1:
            cache[:] = -1
            if dfs(player2_loc[0], player2_loc[1], new_board, board.shape[0]-1, cache) == 0:
                valid_walls[idx] = 0
        else:
            valid_walls[idx] = 0

        remove_wall(wall_array[idx], new_board)

@njit
def add_wall(wall_array, board):
    board[wall_array[0, 0], wall_array[0, 1]] = 8
    board[wall_array[1, 0], wall_array[1, 1]] = 8
    board[wall_array[2, 0], wall_array[2, 1]] = 8

@njit
def remove_wall(wall_array, board):
    board[wall_array[0, 0], wall_array[0, 1]] = -8
    board[wall_array[1, 0], wall_array[1, 1]] = -8
    board[wall_array[2, 0], wall_array[2, 1]] = -8

@njit
def dfs(i, j, board, row, cache):
    if cache[i, j] != -1:
        return cache[i, j]
    
    if i == row:
        cache[i, j] = 1
        return 1

    prev = board[i, j]
    board[i, j] = -2

    for x, y in get_neighbors(i, j, board):
        if dfs(x, y, board, row, cache):
            board[i, j] = prev
            cache[i, j] = 1
            return 1

    board[i, j] = prev
    cache[i, j] = 0
    return 0

@njit
def get_neighbors(i, j, board):
    walls = [[i, j - 1], [i - 1, j], [i, j + 1], [i + 1, j]]
    locs = [[i, j - 2], [i - 2, j], [i, j + 2], [i + 2, j]]

    jump_locs = [[0, -2], [-2, 0], [0, 2], [2, 0]]

    ret = np.zeros(len(walls), dtype = np.uint8)
    ret_locs = []

    for idx in range(len(locs)):
        loc = locs[idx]
        wall = walls[idx]

        if loc[0] < 0 or loc[0] >= board.shape[0] or loc[1] < 0 or loc[1] >= board.shape[1] or \
            board[wall[0], wall[1]] != -8 or board[loc[0], loc[1]] == -2:
                continue

        ret_locs.append(loc)

    return ret_locs
