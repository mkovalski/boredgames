# boredgames

![Quoridor Example](https://github.com/mkovalski/boredgames/blob/master/images/quoridor_example.gif)

RL environments + algorithms for board games.


## Environments

The environments are similar to OpenAI Gym environments and can be used similarly, but they differ in two ways:
    - The reset and step take a "player" argument to signify which player you are moving as.
        - The environments have a property called 'n_players' which will let you know the maximum number of players for each game
        - When possible, the boards are either transposed or player values are changed in the state
    - Values returned from the "step" function / replay buffer
        - A list of valid moves that the agent can make from the state
        - A list of valid moves for the next state (if applicable to the algorithm)

The environments also provide the neural network structure for specific algorithms, as some network structures
may work better in some games than others, the states / number of states may be of different sizes, etc.
