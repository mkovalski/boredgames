#include <vector>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include "math.h"
#include "player.h"
#include "wall.h"

#ifndef QUORIDOR
#define QUORIDOR

const int N = 9;
constexpr int MAXWALLPOSITIONS = 2 * pow(N-1, 2);
constexpr int MAXPOSITIONS = 2 * pow(N-1, 2) + 4;
const int OPEN = 0;
const int NOWALL = -8;
const int WALL = 8;
const int EXPLORED = -2
const int PLAYER1 = 1;
const int PLAYER2 = 2;
const float EPS = 1e-8;
const string MOVEDIRS[4] = {"left", "up", "down", "right"};

class Quoridor {
    public:
        Quoridor();
        void reset(int);
        std::tuple<int> action_shape();
        std::tuple<std::tuple<int, int, int>, std::tuple<int>> state_shape();
        void get_tile_state(int, std::array<int, 2>&);
        std::pair<int[N][N], int[2]> get_state();
        

    private:
        // Values
        int[N][N] board;
        Wall[MAXWALLPOSITIONS] wall_array;
        Player[2] player_map;
        bool[MAXWALLPOSITIONS] valid_walls;
        bool[MAXWALLPOSITIONS] per_move_valid_walls;
        bool done;
        int winner;
        std::map<std::pair<int, int>, bool> complete_cache;
        int[MAXWALLPOSITIONS] create_wall_array();
        
        // Funcs
        bool can_complete(int);
        bool dfs(int, int, int, int[N][N]);
        int[4] get_neighbors(int, int, int[N][N], int);
        bool finished(int, int);
        int get_reward(int);
        bool update_walls();
        bool is_valid_wall(int);
        
        void add_wall(Wall);
        void remove_wall(Wall);

        vector<std::pair<int, int>> get_valid_moves(int);
        int[MAXPOSITIONS] get_all_moves(int);
        void move_pawn(int, int);
        void move(int, int[MAXPOSITIONS]);
        int[MAXPOSITIONS] sample(int);

};

#endif // QUORIDOR
