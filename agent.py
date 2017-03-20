import numpy as np
import random
from environment import get_valid_moves, make_move, get_winner
from collections import deque
from model import create_model, actor_model, critic_model
import operator
from keras.models import load_model
from keras import backend as k
import tensorflow as tf


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
                 model=None,
                 batch_size=32,
                 exploration_rate=0,
                 file_name='models/agent.h5'):

        Agent.__init__(self, tiles)

        self.learns = True
        self.Q = deque([], memory)
        self.gamma = gamma

        self.batch_size = batch_size
        self.exploration_rate = exploration_rate
        self.file_name = file_name

        if model:
            self.model = model
        else:
            try:
                self.model = load_model(self.file_name)
                print('read model ' + self.file_name)
            except:
                self.model = create_model()
                print('created new model: ' + self.file_name)

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

    def save(self):
        self.model.save(self.file_name)


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
            return {0:0}
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


class ActorCriticAgent(Agent):
    def __init__(self,
                 tiles=(None, None),
                 actor=actor_model(),
                 critic=critic_model(),
                 alpha=.5,
                 beta=.5,
                 gamma=.9,
                 exploration_rate=.3):

        # call parent constructor
        Agent.__init__(self, tiles=tiles)

        # create actor model
        self.actor = actor
        # create critic model
        self.critic = critic

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.learns = True
        self.er = exploration_rate

    def get_action(self, state):
        if random.random() < self.er:
            return Agent.get_action(self, state)

        parsed_state = parse_state(state)
        predicted_actions = list(self.actor.predict(parsed_state)[0].argsort())

        valid_actions = get_valid_moves(state)
        for i in range(7):
            index = predicted_actions.index(i)
            if index in valid_actions:
                return index

    def learn(self, game):
        parsed_game = self.__parse_game(game)
        i = 1

        for turn in parsed_game:

            parsed_st0 = parse_state(turn['st0'])
            p0 = self.critic.predict(parsed_st0)[0][0]

            try:
                parsed_st1 = parse_state(turn['st1'])
                p1 = self.critic.predict(parsed_st1)[0][0]
                r = 0
            except KeyError:
                p1 = 0
                r = get_winner(parsed_game[-1]['f_st'])

            delta = r + self.gamma * p1 - p0

            # get the gradients
            weights = self.critic.trainable_weights
            output = self.critic.output

            critic_grad_func = k.gradients(output, weights)

            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            critic_grad = sess.run(critic_grad_func, feed_dict={self.critic.input:parsed_st0})
            # end get the gradients

            # update weights
            w = np.array(self.critic.get_weights())
            w += self.beta * delta * np.array(critic_grad)
            self.critic.set_weights(w)

            # get the gradients
            weights = self.actor.trainable_weights
            output = self.actor.output

            actor_grad_func = k.gradients(output, weights)

            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            actor_grad = sess.run(actor_grad_func, feed_dict={self.actor.input: parsed_st0})
            # end get gradients

            # update weights
            theta = self.actor.get_weights()
            theta += self.alpha * i * delta * np.array(actor_grad)
            self.actor.set_weights(theta)

            i *= self.gamma

    def __parse_game(self, game):
        parsed_game = []
        for i, turn in enumerate(game):
            if turn['p'] == self._tile:
                parsed_turn = {
                    'st0': turn['st'],
                    'a': turn['a']
                }

                try:
                    parsed_turn['st1'] = game[int(i+2)]['st']
                except IndexError:
                    parsed_turn['f_st'] = game[-1]['f_st']
                except KeyError:
                    parsed_turn['f_st'] = game[-1]['f_st']

                parsed_game.append(parsed_turn)

        return parsed_game

    def save(self):
        self.actor.save('models/actor.h5')
        self.critic.save('models/critic.h5')

