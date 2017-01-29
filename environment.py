from copy import deepcopy

def get_valid_moves(state):
    """it returns a list with the indexes of the colums
       where it is possible to make a move
       @state: 2D list (default 6 rows and 7 colums)"""
    
    valid_moves = [] 
    # check every cell in the top row
    for i, cell in enumerate(state[0]):
        if cell == 0:
            valid_moves.append(i)

    return valid_moves

def get_winner(state, tiles):
    height = len(state)
    width = len(state[0])
    
    for tile in tiles:
        # check horizontal
        for x in range(width - 3):
            for y in range(height):
               if state[y][x] == state[y][x+1] == state[y][x+2] == state[y][x+3] == tile:
                    return tile

        # check vertical
        for x in range(width):
            for y in range(height - 3):
                if state[y][x] == state[y+1][x] == state[y+2][x] == state[y+3][x] == tile:
                    return tile

        # check / diagonal
        for x in range(width - 3):
            for y in range(height - 3):
                if state[y][x] == state[y+1][x+1] == state[y+2][x+2] == state[y+3][x+3] == tile:
                    return tile

        # check \ diagonal
        for x in range(3, width - 3):
            for y in range(height - 3):
                if state[y][x] == state[y+1][x-1] == state[y+2][x-2] == state[y+3][x-3] == tile:
                    return tile
    
    return 0

def make_move(state, action, tile):
    new_state = deepcopy(state)
    if new_state[0][action] == 0:
        for i in reversed(range(len(new_state))):
            if new_state[i][action] == 0:
                new_state[i][action] = tile
                return new_state

def game_over(state, tiles):
    return get_winner(state, tiles) or not get_valid_moves(state)

def get_initial_state():
    return [[0 for _ in range(7)] for __ in range(6)]


class Environment:
    def __init__(self):
        self.state = get_initial_state()

    def start(self):
        self.state = get_initial_state()

    def make_move(self, action, tile):
        # check if it is valid first
        self.state = make_move(self.state, action, tile)

    def over(self, tiles):
        return game_over(self.state, tiles)
