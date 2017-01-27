from environment import get_valid_moves


class Agent:
    """it is the most basic class for an agent
       it performs random actions"""
    def __init__(self):
        pass

    def get_action(self, state):
        return random.choice(get_valid_moves(state))


class IntelligentAgent(Agent):
    """it is a basic intelligent agent that follows this strategy
       1- if there is a move that makes it win, take it
       2- if the oponent has a winning move, block it
       3- take random action"""

    def __init__(self, tile=None, oponent_tile=None):
        self.tile = tile
        if oponent_tile:
            self.oponent_tile = oponent_tile

    def get_action(self, state):
        # get all possible moves
        possible_actions = get_valid_moves(state))
        
        # check if it has a winning move
        for action in possible_actions:
            simulated_state = make_move(state, action, tile=self.tile)
            if get_winner(simulated_state) == self.tile:
                return action

        # check if the oponent has a winning move
        for action in possible_actions:
            simulated_state = make_move(state, action, tile=self.oponent_tile)
            if get_winner(simulated_state) == self.oponent_tile:
                return action

        # otherwise take random action
        return super(IntelligentAgent, self).get_action(state)


