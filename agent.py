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
    def __init__(self, tile=None, alpha=0.5, gamma=0.9):
        Agent.__init__(self, tile)
        self.Q = {}
        self.alpha = alpha
        self.learns = True
        self.added_states = 0
        self.gamma = gamma

    def learn(self, turns):
        parsed_states = []
        for turn in turns:
            parsed_state = parse_state(turn)
            self.add_state(turn, parsed_state)
            parsed_states.append(parsed_state)

        number_of_turns = len(parsed_states)

        for i in reversed(range(1, number_of_turns)):
            old_q = self.Q[parsed_states[i - 1]]
            current_value = self.Q[parsed_states[i]]
            self.Q[parsed_states[i - 1]] = old_q + self.alpha * (
            self.gamma ** (number_of_turns - i) * current_value - old_q)

    def get_action(self, state):
        valid_moves = get_valid_moves(state)

        self.add_state(state)

        max_move = None
        max_value = None
        for move in valid_moves:
            simulated_state = make_move(state, move, self.tile)
            parsed_simulated_state = parse_state(simulated_state)

            self.add_state(simulated_state, parsed_simulated_state)
            state_value = self.Q[parsed_simulated_state]
            if state_value > max_value:
                max_value = state_value
                max_move = move

        if max_value > 0:
            return max_move
        else:
            if random.random() < 0.1:
                return Agent.get_action(self, state)
            else:
                return max_move

    def add_state(self, state, parsed_state=None):
        if not parsed_state:
            parsed_state = parse_state(state)

        if parsed_state not in self.Q.keys():
            winner = get_winner(state)
            if winner == 1:
                self.Q[parsed_state] = 1
            elif winner == 0:
                self.Q[parsed_state] = 0
            elif winner == -1:
                self.Q[parsed_state] = -1
            self.added_states += 1
