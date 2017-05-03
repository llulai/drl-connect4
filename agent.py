import numpy as np
import random
from environment import get_valid_moves, make_move, get_winner
from collections import deque
from model import create_model
import operator


def parse_state(state):
    return np.array(state).reshape((1, 42))


class Agent:
    """it is the most basic class for an agent
       it performs random actions"""

    def __init__(self, tiles=(None, None)):
        """
        :param tiles: Tile to be used by the agent when playing the game
        """
        self._tile = tiles[0]
        self._opponent = tiles[1]
        self.learns = False

    def get_action(self, state):
        """
        :param state: 6x7 list representing the current state of the game
        :return int: Index of the column to put the piece (always checks for valid moves)
        """
        return random.choice(get_valid_moves(state))

    def get_tile(self):
        return self._tile

    def set_tiles(self, tiles=(None, None)):
        self._tile = tiles[0]
        self._opponent = tiles[1]


class IntelligentAgent(Agent):
    """it is a basic intelligent agent that follows this strategy
       1- if there is a move that makes it win, take it
       2- if the opponent has a winning move, block it
       3- take random action"""

    def __init__(self, tiles=(None, None)):
        """
        :param tiles: Tile to be used by the agent when playing the game
        """
        # call parent init
        Agent.__init__(self, tiles=tiles)

    def get_action(self, state):
        """
        :param state : (list) 6x7 list representing the current state of the game
        :return int: Index of the column to put the piece (always checks for valid moves)
        """

        # get all possible moves
        possible_actions = get_valid_moves(state)

        # check if it has a winning move
        for action in possible_actions:
            simulated_state = make_move(state, action, tile=self._tile)
            # if the simulated next state is a winning game
            if get_winner(simulated_state) == self._tile:
                # take the action
                return action

        # check if the opponent has a winning move
        for action in possible_actions:
            simulated_state = make_move(state, action, tile=self._opponent)
            # if the simulated state is a loosing game
            if get_winner(simulated_state) == self._opponent:
                # block that move
                return action

        # otherwise take random action
        return Agent.get_action(self, state)


class LearningAgent(Agent):
    def __init__(self,
                 tiles=(None, None),
                 gamma=0.75,
                 memory=100,
                 model=create_model(),
                 batch_size=32,
                 exploration_rate=0,
                 search_width=1,
                 search_depth=3):

        Agent.__init__(self, tiles)

        self.learns = True
        self.Q = deque([], memory)
        self.gamma = gamma
        self.model = model
        self.batch_size = batch_size
        self.exploration_rate = exploration_rate
        self.search_width = search_width
        self.search_depth = search_depth

    def memorize(self, game):
        self.Q.append(game)

    def learn(self):
        sample_size = min(len(self.Q), self.batch_size)

        # select the games to learn
        games = random.sample(self.Q, sample_size)

        # initialize input and target variables to train
        states = []
        values = []

        for turns in games:
            for i in range(1, len(turns)):
                st0 = parse_state(turns[i-1])

                if i == len(turns) - 1:
                    # if the agent won, reward is 1, if lose -1 and if draw = 0
                    p = get_winner(turns[i]) * self._tile

                else:
                    st1 = parse_state(turns[i])
                    pt1 = self.model.predict(st1)[0][0]
                    p = self.gamma * pt1

                states.append(st0[0])
                values.append(p)

        states = np.array(states)
        values = np.array(values)

        self.model.train_on_batch(states, values)

    def get_action(self, state):
        # take random action with probability given by exploration rate
        if random.random() < self.exploration_rate:
            action = Agent.get_action(self, state)
            return action

        max_value = - float('inf')
        max_action = None

        # iterate over the valid actions
        for action in get_valid_moves(state):
            # simulate the action and get the value for that state
            new_state = make_move(state, action, self._tile)
            parsed_new_state = parse_state(new_state)
            value_new_state = self.model.predict(parsed_new_state)[0][0]

            # check if the current action has the highest value
            if value_new_state > max_value:
                max_value = value_new_state
                max_action = action

        # return the action with the highest value
        return max_action

    def get_value_action(self, state, look=0):
        if look == 0:
            return {0: 0}
        valid_moves = get_valid_moves(state)
        values = []

        for move in range(len(valid_moves)):
            new_state = make_move(state, valid_moves[move], self._tile)
            parsed_new_state = parse_state(new_state)
            values[valid_moves[move]] = self.model.predict(parsed_new_state)[0][0]

            max(values.items(), key=operator.itemgetter(1))[0]

            if get_winner(new_state) == self._tile:
                values[valid_moves[move]] = 1
                break
            else:
                opponent_moves = get_valid_moves(new_state)
                for opponent_move in range(len(opponent_moves)):
                    last_state = make_move(new_state, opponent_moves[opponent_move], self._opponent)
                    if get_winner(last_state) == self._opponent:
                        values[valid_moves[move]] = -1
                        break
                    else:
                        results = self.get_value_action(last_state, look=look - 1).values()
                        values[valid_moves[move]] += sum(results) / 49.

        return values


class SearchAgent(Agent):
    def __init__(self, depth=1, tiles=(None, None)):
        self.depth = depth
        Agent.__init__(self, tiles=tiles)

    def get_action(self, state):
        values = self.get_value_action(state, self.depth)
        try:
            with open('training_examples.txt', 'a') as f:
                f.write(str(state) + ';')
                f.write(str(values) + '\n')
        except IOError:
            with open('training_examples.txt', 'w') as f:
                f.write(str(values) + '\n')
        return max(values.items(), key=operator.itemgetter(1))[0]

    def get_value_action(self, state, look=1):

        if look == 0:
            return {0: 0}

        valid_moves = get_valid_moves(state)
        values = {}

        for move in range(len(valid_moves)):
            values[valid_moves[move]] = 0
            new_state = make_move(state, valid_moves[move], self._tile)
            if get_winner(new_state) == self._tile:
                values[valid_moves[move]] = 1
                break
            else:
                opponent_moves = get_valid_moves(new_state)
                for opponent_move in range(len(opponent_moves)):
                    last_state = make_move(new_state, opponent_moves[opponent_move], self._opponent)
                    if get_winner(last_state) == self._opponent:
                        values[valid_moves[move]] = -1
                        break
                    else:
                        results = self.get_value_action(last_state, look=look - 1).values()
                        values[valid_moves[move]] += sum(results) / 49.

        return values
