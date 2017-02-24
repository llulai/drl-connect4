from agent import Agent
from itertools import cycle
from environment import get_initial_state, game_over, make_move, get_winner
from random import randrange
import pickle


def simulate(agents=[None, None], iterations=10, tiles=[1, -1], log=True, backup=False, print_every=10):
    
    # if the agents are not passed
    # create dumb agents
    for i in range(len(agents)):
        if not agents[i]:
            agents[i] = Agent(tiles[i])
        else:
            agents[i].set_tile(tiles[i])

    # create an iterator for alternate players
    players = cycle(agents)

    # initialize list to return
    results = []
    won = 0
    games_started = 0
    total_reward = 0

    # run n simulations
    for iteration in range(1, iterations + 1):
        # print('iteration ' + str(iteration) + ' of ' + str(iterations))

        # get an empty board
        state = get_initial_state()

        # initialize the list for this game
        current_game = []

        # randomize the first agent to play
        for _ in range(randrange(1, 3)):
            current_player = next(players)
        
        # play until the game is over

        played_first =  current_player.tile == 1

        while not game_over(state):

            # initial state for this turn to string
            initial_state = state
            
            # change current player
            current_player = next(players)

            # ask the agent to give an action
            action = current_player.get_action(state)

            # perform the action and update the state of the game
            state = make_move(state, action, current_player.tile)
            
            # if the current player is agent 1
            # add the current turn to the list
            if current_player.tile == 1:
                turn = {'a': action, 'st0': initial_state, 'st1': state}
                current_game.append(turn)

        current_game = parse_game(current_game, state, 0.9)

        # add the last game to the results list
        results.append(current_game)

        for agent in agents:
            if agent.learns:
                agent.memorize(current_game)
                if iteration > agent.batch_size:
                    agent.learn()

        if log:
            reward = get_winner(state)
            total_reward += reward
            if reward > 0:
                won += 1
            #games_started += 1 if played_first else 0

            if iteration % print_every == 0:
                print('won ' + str(won) + ' out of ' + str(print_every) + ' games')
                print('reward: ' + str(total_reward))

                total_reward = 0
                won = 0
             #   games_started = 0

            if backup and iteration % print_every == 0:
                for agent in agents:
                    if agent.learns:
                        agent.model.save(str(agent.tile) + 'model.h5')

    return agents[0], results


def save_q(agent):
    pickle_file = 'learning_agent.pickle'

    dataset = {'learningagent': agent.Q}

    with open(pickle_file, 'wb') as picklefile:
        pickle.dump(dataset, picklefile, pickle.HIGHEST_PROTOCOL)
    picklefile.close()


def parse_game(current_game, last_state, gamma):
    # clean results for game
    clean_game = []

    turns = len(current_game)
    reward = get_winner(last_state)

    for i, turn in enumerate(current_game):
        turn['r'] = gamma ** (turns - i - 1) * reward

        clean_game.append(turn)

    return clean_game
