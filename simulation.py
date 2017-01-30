from agent import Agent
from itertools import cycle
from environment import get_initial_state, game_over, make_move, get_winner
from random import randrange


def simulate(agents=[None, None], iterations=10, tiles=[1, -1], gamma=0.9, log=True, learn_after=5, train=False, backup=False):
    
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
    total_reward = 0

    # run n simulations
    for iteration in range(1, iterations):
        print('iteration ' + str(iteration) + ' of ' + str(iterations))

        # get an empty board
        state = get_initial_state()

        # initialize the list for this game
        current_game = []

        # randomize the first agent to play
        for _ in range(randrange(2)):
            current_player = next(players)
        
        # play until the game is over
        while not game_over(state, tiles):

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
            if current_player.get_tile() == 1:
                step = (initial_state, action)
                current_game.append(step)

        # parse game
        clean_game = parse_game(current_game, state, gamma, tiles)
        # add the last game to the results list
        results.append(clean_game)

        for agent in agents:
            if agent.learns:
                agent.remembers(clean_game)

        if train and iteration > learn_after:

            for agent in agents:
                if agent.learns:
                    agent.train()


        if log:
            reward = get_winner(state, tiles)
            total_reward += reward
            if reward > 0:
                won += 1

            if iteration % 10 == 0:
                print('won ' + str(won) + ' out of 10 games')
                print('reward: ' + str(total_reward))

                total_reward = 0
                won = 0

            if backup:
                for agent in agents:
                    if agent.learns:
                        agent.model.save('backup_model')


    return results


def parse_game(current_game, last_state, gamma, tiles):
    # clean results for game
    clean_game = []

    turns = len(current_game)
    reward = get_winner(last_state, tiles)

    for i, step in enumerate(current_game):

        clean_step = {
            'st': step[0],
            'action': step[1],
            'reward': gamma ** (turns - i - 1) * reward
        }

        try:
            clean_step['st_1'] = current_game[i + 1][0]
        except IndexError:
            clean_step['st_1'] = None

        clean_game.append(clean_step)

    return clean_game
