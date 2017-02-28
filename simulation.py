from agent import Agent, LearningAgent
from itertools import cycle
from environment import get_initial_state, game_over, make_move, get_winner
from random import randrange


def simulate(agent=None, opponent=None, iterations=10, log=True, backup=False, print_every=10, start=0):
    
    # if the agents are not passed
    # create dumb agents
    if not agent:
        agent = LearningAgent(tile=1)
    else:
        agent.set_tile(1)

    if not opponent:
        opponent = Agent(-1)
    # create an iterator for alternate players
    players = cycle([agent, opponent])

    # initialize list to return
    results = []
    won = 0
    total_reward = 0

    # run n simulations
    for iteration in range(start + 1, start + iterations + 1):
        # get an empty board
        state = get_initial_state()

        # initialize the list for this game
        current_game = []

        # randomize the first agent to play
        for _ in range(randrange(1, 3)):
            current_player = next(players)
        
        # play until the game is over
        while not game_over(state):

            # initial state for this turn to string
            initial_state = state
            
            # change current player
            current_player = next(players)

            # ask the agent to give an action
            action = current_player.get_action(state)

            # perform the action and update the state of the game
            state = make_move(state, action, current_player.tile)

            # add the current turn to the list
            current_game.append(initial_state)

        # add last state to the game
        current_game.append(state)

        # add the last game to the results list
        results.append(current_game)

        agent.memorize(current_game)
        if iteration  - start > agent.batch_size:
            agent.learn()

        if log:
            reward = get_winner(state)
            total_reward += reward
            if reward > 0:
                won += 1

            if iteration % print_every == 0:
                print('won ' + str(won) + ' out of ' + str(print_every) + ' games')
                print('reward: ' + str(total_reward))

                total_reward = 0
                won = 0

            if backup and iteration % print_every == 0:
                agent.model.save('models/model_' + str(iteration) + '.h5')

    return agent, results
