import numpy as np
import random
from environment import get_valid_moves, make_move, get_winner
from neural_network import NeuralNetwork


def parse_state(state):
    return list(np.array(state).flatten())


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
    def __init__(self, tile=None, opponent_tile=None, alpha=0.5, gamma=0.9):
        Agent.__init__(self, tile)
        self.opponent_tile = opponent_tile
        self.learns = True
        self.added_states = 0
        self.gamma = gamma
        self.model = NeuralNetwork(42, 128, 1, alpha)
        self.learning = True

    def learn(self, turns):
        self.model.e_input_to_hidden = 0
        self.model.e_hidden_to_output = 0
        parsed_states = []

        for turn in turns:
            parsed_states.append(parse_state(turn))

        for i in range(1, len(parsed_states)):
            winner = get_winner(turns[i])
            if winner * self.tile == 1:
                reward = 1
            else:
                reward = 0

            self.model.train(parsed_states[i-1], parsed_states[i], reward)

    def get_action(self, state):
        valid_moves_1 = get_valid_moves(state)

        max_move = None
        max_value = None
        for move in valid_moves_1:
            simulated_state = make_move(state, move, self.tile)
            valid_moves_2 = get_valid_moves(simulated_state)

            if valid_moves_2:
                for move2 in valid_moves_2:
                    simulated_state_2 = make_move(state, move2, self.opponent_tile)
                    parsed_simulated_state_2 = parse_state(simulated_state_2)
                    state_value = self.model.predict(parsed_simulated_state_2)

                    if state_value > max_value:
                        max_value = state_value
                        max_move = move
            else:
                parsed_simulated_state = parse_state(simulated_state)
                state_value = self.model.predict(parsed_simulated_state)

                if state_value > max_value:
                    max_value = state_value
                    max_move = move

        if random.random() < 0.1 and self.learning:
            return Agent.get_action(self, state)
        else:
            return max_move
