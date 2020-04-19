#include <map>
#include <vector>
#include <set>
#include "player.h"

Player::Player(int marker, std::pair<int, int> init_loc, int winning_row) {
    this->marker = marker;
    this->init_loc = init_loc;
    this->winning_row = winning_row;

    curr_loc = init_loc;
    nmoves = MAX_MOVES;
    if (marker == 1)
        opponent = 2;
    else
        opponent = 1;

    move_prob = 0.0f;
}

void Player::reset(float move_prob) {
    curr_loc = init_loc;
    nmoves = MAX_MOVES;
    move_prob = move_prob;
}

void Player::dec_moves() {
    if (nmoves > 0)
        nmoves--;
}

void Player::move(int direction, int board[N][N]) {
    if (direction == 0)
        curr_loc.second -= 2;
    else if (direction == 1)
        curr_loc.first -= 2;
    else if (direction == 2)
        curr_loc.second += 2;
    else if (direction == 3)
        curr_loc.first += 2;

    if (board[curr_loc.first][curr_loc.second] == opponent)
        move(direction, board);
}
