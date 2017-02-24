import numpy as np
import random
from environment import get_valid_moves, make_move, get_winner
from collections import deque
from model import create_model


def parse_state(state):
    return np.array(state).reshape((1, 6, 7, 1))


class Agent:
    """it is the most basic class for an agent
       it performs random actions"""
    def __init__(self, tile=None):
        """
        :param tile (int): Tile to be used by the agent when playing the game
        """
        self.tile = tile
        self.learns = False

    def get_action(self, state):
        """
        :param state: 6x7 list representing the current state of the game
        :return int: Index of the column to put the piece (always checks for valid moves)
        """
        return random.choice(get_valid_moves(state))
    
    def get_tile(self):
        return self.tile

    def set_tile(self, tile):
        self.tile = tile


class IntelligentAgent(Agent):
    """it is a basic intelligent agent that follows this strategy
       1- if there is a move that makes it win, take it
       2- if the opponent has a winning move, block it
       3- take random action"""

    def __init__(self, tile=None, opponent_tile=None):
        """

        :param tile (int): Tile to be used by the agent when playing the game
        :param opponent_tile: Tile used by the opponent agent when playing the game
        """
        # call parent init
        Agent.__init__(self, tile=tile)

        # set the opponent tile
        if opponent_tile:
            self.opponent_tile = opponent_tile

    def get_action(self, state):
        """
        :param state : (list) 6x7 list representing the current state of the game
        :return int: Index of the column to put the piece (always checks for valid moves)
        """

        # get all possible moves
        possible_actions = get_valid_moves(state)

        # check if it has a winning move
        for action in possible_actions:
            simulated_state = make_move(state, action, tile=self.tile)
            # if the simulated next state is a winning game
            if get_winner(simulated_state) == self.tile:
                # take the action
                return action

        # check if the opponent has a winning move
        for action in possible_actions:
            simulated_state = make_move(state, action, tile=self.opponent_tile)
            # if the simulated state is a loosing game
            if get_winner(simulated_state) == self.opponent_tile:
                # block that move
                return action

        # otherwise take random action
        return Agent.get_action(self, state)


class LearningAgent(Agent):
    def __init__(self,
                 tile=None,
                 alpha=0.5,
                 gamma=0.9,
                 memory=100,
                 model=create_model(),
                 batch_size=32,
                 exploration_rate=0.25):

        Agent.__init__(self, tile)
        self.learns = True

        self.Q = deque([], memory)
        self.alpha = alpha
        self.gamma = gamma
        self.model = model
        self.batch_size = batch_size
        self.exploration_rate = exploration_rate

    def memorize(self, game):
        self.Q.append(game)

    def learn(self):
        games = random.sample(self.Q, self.batch_size)
        inputs = []
        y = []
        for game in games:
            for turn in game:
                x = parse_state(turn['st0'])
                p = self.model.predict(x).reshape(7)

                try:
                    x1 = parse_state(turn['st1'])
                    p1 = self.model.predict(x1)[0]

                    reward = p1.max()
                except:
                    reward = turn['r']

                p[turn['a']] = reward

                inputs.append(x[0])
                y.append(p)

        inputs = np.array(inputs)
        y = np.array(y)

        self.model.train_on_batch(inputs, y)

    def get_action(self, state):
        if random.random() < self.exploration_rate:
            return Agent.get_action(self, state)

        parsed_state = parse_state(state)
        action = self.model.predict(parsed_state).argmax()
        valid_actions = get_valid_moves(state)

        if action in valid_actions:
            return action
        else:
            return Agent.get_action(self, state)
