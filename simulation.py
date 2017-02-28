from agent import Agent, IntelligentAgent, LearningAgent
from itertools import cycle
from environment import get_initial_state, game_over, make_move, get_winner
from random import randrange
import pickle


def simulate(agent=LearningAgent(), sparring=IntelligentAgent(), opponent=Agent(), iterations=10, log=True, backup=False, print_every=10):
    
    # if the agents are not passed
    # create dumb agents
    agent.set_tile(1)
    sparring.set_tile(-1)

    # create an iterator for alternate players
    players = cycle([agent, sparring])

    # initialize list to return
    won = 0
    total_reward = 0

    # run n simulations
    for iteration in range(1, iterations + 1):

        # get an empty board
        state = get_initial_state()

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
            
            # if the current player is agent 1
            # add the current turn to the list
            if current_player.learns:
                r = get_winner(state)
                turn = {'a': action, 'st0': initial_state, 'st1': state, 'r': r}
                current_player.memorize(turn)
                current_player.learn()


        #TODO learn over epochs against itself and then test against ia

        if log:
            reward = get_winner(state)
            total_reward += reward
            if reward > 0:
                won += 1
            # games_started += 1 if played_first else 0

            if iteration % print_every == 0:
                print('won ' + str(won) + ' out of ' + str(print_every) + ' games')
                print('reward: ' + str(total_reward))

                total_reward = 0
                won = 0

            if backup and iteration % print_every == 0:
                if agent.learns:
                    agent.model.save('models/model_deep_q_learning.h5')

    return agent
