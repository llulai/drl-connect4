from agent import Agent, LearningAgent
from itertools import cycle
from environment import get_initial_state, game_over, make_move, get_winner
from random import randrange
import tensorflow as tf


def simulate(agent=LearningAgent(),
             sparring=LearningAgent(),
             opponents=None,
             iterations=10,
             log=True,
             backup=False,
             print_every=10,
             n_test=100):
    
    # set tiles for agent and sparring
    agent.set_tiles((1, -1))
    sparring.set_tiles((-1, 1))
    for opponent in opponents:
        opponent.set_tiles((-1, 1))

    # create an iterator for alternate players
    players = cycle([agent, sparring])

    # initialize list to return
    results = []

    # create the variable to save the model
    saver = tf.train.Saver()

    # run n simulations
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # add the session to the agent so it can optimize
        agent.sess = sess

        # read if there is a saved model
        try:
            saver.restore(sess, 'models/agent.ckpt')
        except:
            pass

        # add the session to the sparring if it is a learning agent
        if sparring.learns:
            sparring.sess = sess

        for iteration in range(1, iterations + 1):

            # play one game
            current_game = play_game(players)

            result = {'iteration': iteration,
                      'sparring': sparring.name,
                      'current_opponent': sparring.name,
                      'game': _extract_games(current_game),
                      'reward': get_winner(current_game[-1]['state']),
                      'kind': 'training',
                      'fist_player': current_game[0]['player']}

            results.append(result)

            # train players
            train(agent, current_game)
            train(sparring, current_game)

            if iteration % print_every == 0:
                # initialize stats variables
                old_er = agent.exploration_rate
                agent.exploration_rate = 0

                for opponent in opponents:
                    won = 0
                    total_reward = 0
                    test_players = cycle([agent, opponent])

                    started_played = 0
                    started_won = 0

                    # play test games
                    for i in range(n_test):
                        game = play_game(test_players)
                        first_player = game[0]['player']
                        parsed_game = parse_game(game, agent.get_tile())
                        reward = get_winner(parsed_game[-1])

                        result = {'iteration': iteration,
                                  'sparring': sparring.name,
                                  'current_opponent': opponent.name,
                                  'game': _extract_games(game),
                                  'reward': reward,
                                  'kind': 'testing',
                                  'fist_player': game[0]['player']}

                        results.append(result)

                        total_reward += reward

                        if reward > 0:
                            won += 1

                        if first_player == 1:
                            started_played += 1
                            if reward > 0:
                                started_won += 1
                    if log:

                        print('won {} out of {} games against {} after {} iterations'.format(won,
                                                                                             n_test,
                                                                                             opponent.name,
                                                                                             iteration))
                        print('won {}% of the started games'.format(int(100. * started_won / started_played)))
                        print('started {} games'.format(started_played))
                        print('reward: ' + str(total_reward))

                # reset to the old exploration rate
                agent.exploration_rate = old_er

            if backup:
                saver.save(sess, 'models/agent.ckpt')

    return results


def _extract_games(game):
    return [turn['state'] for turn in game]


def train(agent, current_game):
    # train the agent
    parsed_game = parse_game(current_game, agent.get_tile())
    if agent.learns:
        agent.memorize(parsed_game)
        agent.learn()


def parse_game(game, player):
    parsed_game = []
    for turn in game:
        if turn['player'] == player:
            parsed_game.append(turn['state'])

    parsed_game.append(game[-1]['state'])

    return parsed_game


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
        current_game.append({'state': initial_state, 'player': current_player.get_tile()})

    # add last state to the game
    current_game.append({'state': state, 'player': current_player.get_tile()})

    return current_game


def get_random_player(players):
    # randomize the first agent to play
    current_player = next(players)
    for _ in range(randrange(1, 3)):
        current_player = next(players)

    return current_player


def test_games(agent, opponent, games=100):
    players = cycle([agent, opponent])

    won_games = 0
    total_reward = 0

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if agent.learns:
            agent.sess = sess

            load_model(sess, saver)

        for i in range(games):
            game = parse_game(play_game(players), agent.get_tile())
            winner = get_winner(game[-1])

            if winner == 1:
                won_games += 1

            total_reward += winner

        print('won {} out of {} games'.format(won_games, games))
        print('reward: {}'.format(total_reward))


# TODO: check if file exists first
def load_model(session, saver):
    try:
        saver.restore(session, 'models/agent.ckpt')
    except:
        pass
