#include <vector>
#include <map>
#include <set>

#ifndef WALL
#define WALL

class Wall {
    public:
        Wall(std::pair<int, int>, std::pair<int, int>, std::pair<int, int>);
        bool operator==(const Wall& rhs) { 
            return ((start == rhs.start and end == rhs.end) || (start == rhs.end and end == rhs.start));
        }

    private:
        std::pair<int, int> start;
        std::pair<int, int> mid;
        std::pair<int, int> end;
};

#endif // WALL
