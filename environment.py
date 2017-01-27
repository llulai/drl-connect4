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

def get_winner(state):
    pass

def make_move(state, action, tile):
    if state[0][action] == 0:
        for i in reverse(range(len(state))):
            if state[i][action] == 0:
                state[i][action] = tile
                return state

def game_over(state):
    return False

def get_initial_state():
    return False

