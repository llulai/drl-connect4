from environment import Agent
from itertools import cycle
import random

def simulate(agents=[Agent, Agent], sample=1000, tiles=[1,2]):
    
    agents[0].set_tile(tiles[0])
    agents[1].set_tile(tiles[1])

    for _ in range(sample):
        state = get_initial_state()
        agents = cycle(random.shuffle(agents))

        while not game_over(state):
            current_player = agents.next()
            action = current_player.get_action(state)
            state = make_move(state, action, current_player.tile)
