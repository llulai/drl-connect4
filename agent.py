import numpy as np
import random
from environment import get_valid_moves, make_move, get_winner


def parse_state(state):
    return ''.join(map(str, list(np.array(state).flatten())))


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
        tiles = [self.tile, self.opponent_tile]

        # check if it has a winning move
        for action in possible_actions:
            simulated_state = make_move(state, action, tile=self.tile)
            # if the simulated next state is a winning game
            if get_winner(simulated_state, tiles) == self.tile:
                # take the action
                return action

        # check if the opponent has a winning move
        for action in possible_actions:
            simulated_state = make_move(state, action, tile=self.opponent_tile)
            # if the simulated state is a loosing game
            if get_winner(simulated_state, tiles) == self.opponent_tile:
                # block that move
                return action

        # otherwise take random action
        return Agent.get_action(self, state)


class LearningAgent(Agent):
    def __init__(self, tile=None):
        Agent.__init__(self, tile)
        self.Q = {}
        self.alpha = 0.5
        self.learns = True

    def learn(self, turns):

        for i in range(1, len(turns)):

            previous_state = turns[i-1]
            current_state = turns[i]

            self.add_state(previous_state)
            self.add_state(current_state)

            parsed_previous_state = parse_state(previous_state)
            parsed_current_state = parse_state(current_state)

            old_q = self.Q[parsed_previous_state]

            current_value = self.Q[parsed_current_state]

            self.Q[parsed_previous_state] = old_q + self.alpha * (current_value - old_q)

    def get_action(self, state):
        valid_moves = get_valid_moves(state)

        if random.random() < 0.1:
            return Agent.get_action(self, state)

        self.add_state(state)

        max_move = None
        max_value = None
        for move in valid_moves:
            simulated_state = make_move(state, move, self.tile)
            self.add_state(simulated_state)
            parsed_simulated_state = parse_state(simulated_state)
            state_value = self.Q[parsed_simulated_state]
            if state_value > max_value:
                max_value = state_value
                max_move = move

        return max_move

    def add_state(self, state):
        parsed_state = parse_state(state)
        if parsed_state not in self.Q.keys():
            winner = get_winner(state, [1, -1])
            if winner == 1:
                self.Q[parsed_state] = 1
            elif winner == 0:
                self.Q[parsed_state] = 0.5
            elif winner == -1:
                self.Q[parsed_state] = 0
