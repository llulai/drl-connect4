from environment import get_valid_moves, make_move, get_winner, get_not_valid_moves
import random

from collections import deque

from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import Adam

import numpy as np


class Agent:
    """it is the most basic class for an agent
       it performs random actions"""
    def __init__(self, tile=None):
        self.tile = tile

    def get_action(self, state):
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
        # call parent init
        super(IntelligentAgent, self).__init__(tile=tile)

        # set the opponent tile
        if opponent_tile:
            self.opponent_tile = opponent_tile

    def get_action(self, state):
        """returns the optimal action for the given state"""

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
        return super(IntelligentAgent, self).get_action(state)


class SuperAgent(Agent):
    def __init__(self, tile, model=None, memory=100, batch_size=5):
        super(SuperAgent, self).__init__(tile=tile)

        # initialize memory of the agent
        self.D = deque([], memory)

        # set the sample size for the experience replay
        self.batch_size = batch_size

        # if a model is given, use that one
        if model:
            self.model = model

        # otherwise use default model
        else:
            self.model = self.get_model()

    def get_action(self, state):
        # parse the given state
        parsed_state = self.parse_state(state)

        # get the priority for each move
        predicted_moves = self.model.predict(parsed_state).argsort()

        # get a list of valid moves
        valid_moves = get_valid_moves(state)

        # greedy exploration
        if random.random() < 0.1:
            return random.choice(valid_moves)

        # take the best option available
        for i in range(7):
            # get the index of the tempted move
            tempted_move = np.where(predicted_moves[0] == i)[0][0]
            # if the top priority move is among valid moves
            if tempted_move in valid_moves:
                # take this move
                return tempted_move

    def parse_state(self, state):
        # flatten the array
        flat_state = list(np.reshape(state, -1))

        # add the multiplication among each variable
        for i in range(42):
            for j in range(42):
                if i != j:
                    product = flat_state[i] * flat_state[j]
                    flat_state.append(product)

        # return the new array
        return np.array([flat_state])

    def remembers(self, turns):
        # iterate over each turn in the game
        for i in range(len(turns)):
            # parse the current state
            parsed_st = self.parse_state(turns[i]['st'])
            turns[i]['pst'] = parsed_st

            # if it is not terminal state
            if turns[i]['st_1']:
                # parse next state
                parsed_st_1 = self.parse_state(turns[i]['st_1'])
                turns[i]['pst_1'] = parsed_st_1
            # if it is terminal state
            else:
                # don't parse it
                turns[i]['pst_1'] = None

        # append the game to the memory
        self.D.append(turns)

    def get_model(self):
        print("Now we build the model")
        model = Sequential()
        model.add(Dense(1024, input_shape=(1764,)))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(7))

        adam = Adam(lr=0.01)
        model.compile(loss='mse', optimizer=adam)
        print("We finish building the model")
        return model

    def train(self):
        # get random sample from memory
        games = random.sample(self.D, self.batch_size)

        # initialize inputs and targets lists
        all_inputs = []
        all_targets = []

        # iterate every game
        for game in games:

            # get how many turns there were in the game
            turns = len(game)

            # initialize inputs and targets lists for the game
            inputs = np.zeros((turns, 1764))  # turns, flat_grid
            targets = np.zeros((turns, 7))  # turns, actions

            # parse each turn
            for i, turn in enumerate(game):
                state_t = turn['pst']      # parsed state
                action_t = turn['action']  # action index
                reward_t = turn['reward']  # observed reward
                state_t1 = turn['pst_1']   # next state (None if terminal state)
                raw_state = turn['st']     # not parsed state (used for punish not valid actions

                inputs[i:i + 1] = state_t  # put the state in the inputs array

                # get the predicted reward for each action in the current state
                targets[i] = self.model.predict(state_t)

                # punish not valid actions
                not_valid_moves = get_not_valid_moves(raw_state)

                for not_valid_action in not_valid_moves:
                    targets[i, not_valid_action] = -10

                # if it is a terminal state
                if state_t1 == None:
                    # set the reward for taking action_t as the observed value
                    targets[i, action_t] = reward_t

                # if it is not a terminal state
                else:
                    # get the predicted reward for each action for the next state
                    q_sa = self.model.predict(state_t1)
                    # set the reward as the observed reward plus the max predicted reward of the next state
                    targets[i, action_t] = reward_t + 0.9 * np.max(q_sa)

            # add input and targets of the current game
            all_inputs.extend(inputs)
            all_targets.extend(targets)

        # cast inputs and targets as numpy arrays
        all_inputs = np.array(all_inputs)
        all_targets = np.array(all_targets)

        # perform gradient step
        self.model.train_on_batch(all_inputs, all_targets)