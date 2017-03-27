import numpy as np
import random
from environment import get_valid_moves, make_move, get_winner
from model import ActorModel, CriticModel
import operator
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


class SearchAgent(Agent):
    def __init__(self, tiles=(None, None), depth=1):
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
                 alpha=.001,
                 beta=.001,
                 gamma=.9,
                 exploration_rate=.3):

        # call parent constructor
        Agent.__init__(self, tiles=tiles)
        self.learns = True

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        # create actor model
        self.actor = ActorModel(session=self.sess)

        # create critic model
        self.critic = CriticModel(session=self.sess)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.er = exploration_rate

    def get_action(self, state):
        if random.random() < self.er:
            return Agent.get_action(self, state)

        parsed_state = parse_state(state)
        predicted_actions = list(self.actor.predict(parsed_state)[0].argsort())
        #print(predicted_actions)

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
                #print(r)

            delta = r + self.gamma * p1 - p0
            #print('p0:', p0)
            #print('p1:', p1)
            #print('delta:', delta)

            # get the gradients
            critic_grad = self.critic.get_gradient(parsed_st0)

            # update weights
            w = np.array(self.critic.get_weights())
            assert w.shape == critic_grad.shape
            w += np.dot(np.multiply(self.beta, delta), critic_grad)
            self.critic.set_weights(w)

            # get the gradients
            actor_grad = self.actor.get_gradient(parsed_st0)

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

