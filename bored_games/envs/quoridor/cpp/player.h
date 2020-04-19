#include <vector>
#include <map>
#include <set>

#ifndef PLAYER
#define PLAYER

class Player {
    static const int MAX_MOVES = 10;
    public:
        Player(int, std::pair<int, int>, int);
        void reset(float);
        void dec_moves();
        void move(int, int[N][N]);

    private:
        int marker;
        int opponent;
        std::pair<int, int> init_loc;
        std::pair<int, int> curr_loc;
        int nmoves;
        int winning_row;
        float move_prob;
};

#endif // PLAYER
