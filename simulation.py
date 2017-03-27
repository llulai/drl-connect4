from itertools import cycle
from environment import get_initial_state, game_over, make_move, get_winner
from random import randrange


def simulate(agent=None,
             sparring=None,
             opponent=None,
             iterations=10,
             log=True,
             backup=False,
             print_every=10,
             learn_every=100):
    
    # set tiles for agent and sparring
    agent.set_tiles((1, -1))
    sparring.set_tiles((-1, 1))
    opponent.set_tiles((-1, 1))

    # create an iterator for alternate players
    players = cycle([agent, sparring])

    # initialize list to return
    results = []

    # run n simulations
    for iteration in range(1, iterations + 1):
        # play one game
        current_game = play_game(players)

        if agent.learns:
            agent.learn(current_game)
        if sparring.learns:
            sparring.memorize(current_game)

        #if iteration % learn_every == 0:
        #    # train players
        #    if agent.learns:
        #        agent.learn()
        #    if sparring.learns:
        #        sparring.learn()

        if log and iteration % print_every == 0:
            # initialize stats variables
            old_er = agent.er
            agent.er = 0
            won = 0
            draw = 0
            lost= 0
            total_reward = 0
            test_players = cycle([agent, opponent])

            # play 100 games
            for i in range(100):
                test_game = play_game(test_players)
                reward = get_winner(test_game[-1]['f_st'])
                #print(reward)

                total_reward += reward

                if reward > 0:
                    won += 1
                elif reward < 0:
                    lost += 1
                else:
                    draw += 1

            agent.er = old_er

            print('won {0}; draw: {1}; lost: {2}'.format(won, draw, lost))

        if backup and iteration % print_every == 0:
            if agent.learns:
                agent.save()

            if sparring.learns:
                sparring.save()

    return agent, results


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

        turn = {'st': initial_state, 'p': current_player.get_tile(), 'a': action}

        # add the current turn to the list
        current_game.append(turn)
        if current_player.get_tile() == 1:
            pass

    # add last state to the game
    current_game.append({'f_st': state, 'p':0})

    return current_game


def get_random_player(players):
    # randomize the first agent to play
    current_player = next(players)
    for _ in range(randrange(1, 3)):
        current_player = next(players)

    return current_player
