from environment import get_valid_moves, make_move, get_winner, get_not_valid_moves
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
    def __init__(self, tile, model=None, memory=100, batch=5):
        self.tile = tile
        self.D = deque([], memory)
        self.BATCH = batch
        
        if model:
            self.model = model
        else:
            self.model = self.get_model()

    def get_action(self, state):
        parsed_state = self.parse_state(state)
        predicted_moves = self.model.predict(parsed_state).argsort()
        valid_moves = get_valid_moves(state)
        
        if random.random() < 0.1:
            return random.choice(valid_moves)
        
        for i in range(7):
            tempted_move = np.where(predicted_moves[0]==i)[0][0]
            if tempted_move in valid_moves:
                return tempted_move
        
        return super(SuperAgent, self).get_action(state)
    
    def parse_state(self, state):
        flat_state = np.reshape(state, -1)

        for i in range(42):
            for j in range(42):
                if  i != j:
                    mult = flat_state[i] * flat_state[j]
                    flat_state.append(mult)

        return flat_state
        
    def remembers(self, turns):
        for i in range(len(turns)):
            parsed_st = self.parse_state(turns[i]['st'])
            if turns[i]['st_1']:
                parsed_st_1 = self.parse_state(turns[i]['st_1'])
                turns[i]['pst_1'] = parsed_st_1
            else:
                turns[i]['pst_1'] = None
            
            turns[i]['pst'] = parsed_st
            
            
        self.D.append(turns)
    
    def get_model(self):
        print("Now we build the model")
        model = Sequential()
        model.add(Dense(1024, input_shape=(1764, )))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(7))

        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        print("We finish building the model")
        return model
    
    def train(self):
        games = random.sample(self.D, self.BATCH)
        
        all_inputs = []
        all_targets = []
        
        for game in games:
            #minibatch = game[0]
            
            turns = len(game)

            inputs = np.zeros((turns, 1, 6, 7))         #turns, pixels, rows, columns
            targets = np.zeros((turns, 7))    #turns, actions

            #Now we do the experience replay
            for i, turn in enumerate(game):
                state_t = turn['pst']
                action_t = turn['action']   #This is action index
                reward_t = turn['reward']
                state_t1 = turn['pst_1']
                
                raw_state = turn['st']
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = self.model.predict(state_t)  # Hitting each buttom probability
                
                #punish not valid actions
                not_valid_moves = get_not_valid_moves(raw_state)
                
                for not_valid_action in not_valid_moves:
                    targets[i, not_valid_action] = -10

                if state_t1 == None:
                    targets[i, action_t] = reward_t
                    
                else:
                    q_sa = self.model.predict(state_t1)
                    targets[i, action_t] = reward_t + 0.9 * np.max(q_sa)
            
            all_inputs.extend(inputs)
            all_targets.extend(targets)
        
        all_inputs = np.array(all_inputs)
        all_targets = np.array(all_targets)
        
        self.model.train_on_batch(all_inputs, all_targets)
        
        return (all_inputs, all_targets)


