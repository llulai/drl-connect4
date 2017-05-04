from agent import Agent, LearningAgent
from itertools import cycle
from environment import get_initial_state, game_over, make_move, get_winner
from random import randrange
import tensorflow as tf


def simulate(agent=LearningAgent(),
             sparring=LearningAgent(),
             opponent=Agent(),
             iterations=10,
             log=True,
             backup=False,
             print_every=10):
    
    # set tiles for agent and sparring
    agent.set_tiles((1, -1))
    sparring.set_tiles((-1, 1))
    opponent.set_tiles((-1, 1))

    # create an iterator for alternate players
    players = cycle([agent, sparring])

    # initialize list to return
    results = []

    # run n simulations
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        agent.sess = sess

        if sparring.learns:
            sparring.sess = sess
        for iteration in range(1, iterations + 1):

            # play one game
            current_game = play_game(players)

            # train players
            train(agent, current_game)
            train(sparring, current_game)

            if log and iteration % print_every == 0:
                # initialize stats variables
                old_er = agent.exploration_rate
                agent.exploration_rate = 0
                won = 0
                total_reward = 0
                test_players = cycle([agent, opponent])

                # play 100 games
                for i in range(100):
                    test_game = play_game(test_players)
                    reward = get_winner(test_game[-1])

                    total_reward += reward

                    if reward > 0:
                        won += 1

                agent.exploration_rate = old_er

                print('won ' + str(won) + ' out of 100 games')
                print('reward: ' + str(total_reward))

            if backup:
                pass
                #if agent.learns:
                #    agent.model.save('models/agent.h5')

                #if sparring.learns:
                #    sparring.model.save('models/sparring.h5')

    return agent, results


def train(agent, current_game):
    # train the agent
    if agent.learns:
        agent.memorize(current_game)
        agent.learn()


def play_game(players):
    current_player = get_random_player(players)
    state = get_initial_state()
    current_game = []

    while not game_over(state):
        # initial state for this turn to string
        initial_state = state

        # change current player
        current_player = next(players)

        # ask the agent to give an action
        action = current_player.get_action(state)

        # perform the action and update the state of the game
        state = make_move(state, action, current_player.get_tile())

        # add the current turn to the list
        current_game.append(initial_state)

    # add last state to the game
    current_game.append(state)

    return current_game


def get_random_player(players):
    # randomize the first agent to play
    current_player = next(players)
    for _ in range(randrange(1, 3)):
        current_player = next(players)

    return current_player
