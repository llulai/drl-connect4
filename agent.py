from environment import get_valid_moves, make_move, get_winner
import random

from collections import deque

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers.core import Activation, Flatten, Dense
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
       2- if the oponent has a winning move, block it
       3- take random action"""

    def __init__(self, tile=None, oponent_tile=None):
        self.tile = tile
        if oponent_tile:
            self.oponent_tile = oponent_tile

    def get_action(self, state):
        # get all possible moves
        possible_actions = get_valid_moves(state)
        tiles = [self.tile, self.oponent_tile]
        # check if it has a winning move
        for action in possible_actions:
            simulated_state = make_move(state, action, tile=self.tile)
            if get_winner(simulated_state, tiles) == self.tile:
                return action

        # check if the oponent has a winning move
        for action in possible_actions:
            simulated_state = make_move(state, action, tile=self.oponent_tile)
            if get_winner(simulated_state, tiles) == self.oponent_tile:
                return action

        # otherwise take random action
        return super(IntelligentAgent, self).get_action(state)
    
class SuperAgent(Agent):
    def __init__(self, tile, model=None, memory=10000):
        self.tile = tile
        self.D = deque([], memory)
        self.BATCH = 500
        
        if model:
            self.model = model
        else:
            self.model = self.get_model()

    def get_action(self, state):
        parsed_state = self.parse_state(state)
        predicted_move = self.model.predict(parsed_state).argmax()
        valid_moves = get_valid_moves(state)
        
        if predicted_move in valid_moves:
            return predicted_move
        else:
            return super(SuperAgent, self).get_action(state)
    
    def parse_state(self, state):
        try:
            return np.array([np.reshape(state, (6,7,1)).transpose(2,0,1)])
        except:
            print(state)
        
    def remembers(self, turns):
        for turn in turns:
            self.D.append(turn)
    
    def get_model(self):
        print("Now we build the model")
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, subsample=(4,4), border_mode='same',input_shape=(1, 6, 7)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(7))

        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        print("We finish building the model")
        return model
    
    def train(self):
        minibatch = random.sample(self.D, self.BATCH)

        inputs = np.zeros((self.BATCH, 1, 6, 7))   #32, 1, 6, 7
        targets = np.zeros((inputs.shape[0], 7))                         #32, 7

        #Now we do the experience replay
        for i in range(0, len(minibatch)):
            state_t = minibatch[i]['st']
            action_t = minibatch[i]['action']   #This is action index
            reward_t = minibatch[i]['reward']
            state_t1 = minibatch[i]['st_1']
            # if terminated, only equals reward

            inputs[i:i + 1] = self.parse_state(state_t)    #I saved down s_t

            targets[i] = self.model.predict(self.parse_state(state_t))  # Hitting each buttom probability

            if not state_t1:
                targets[i, action_t] = reward_t
            else:
                q_sa = self.model.predict(self.parse_state(state_t1))
                targets[i, action_t] = reward_t + 0.9 * np.max(q_sa)

            self.model.train_on_batch(inputs, targets)


