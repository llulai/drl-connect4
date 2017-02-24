from copy import deepcopy


def get_valid_moves(state):
    """
    it returns a list with the indexes of the columns
       where it is possible to make a move
    :param state: 2D list (default 6 rows and 7 columns)
    :return: (list) with indexes where it is possible to put a tile
    """
    
    valid_moves = [] 
    # check every cell in the top row
    for i, cell in enumerate(state[0]):
        if cell == 0:
            valid_moves.append(i)

    return valid_moves


def get_not_valid_moves(state):
    """
    it returns a list with the indexes of the columns
       where it is not possible to make a move
    :param state: 2D list (default 6 rows and 7 columns)
    :return: (list) with indexes where it is not possible to put a tile
    """
    
    not_valid_moves = [] 
    # check every cell in the top row
    for i, cell in enumerate(state[0]):
        if cell != 0:
            not_valid_moves.append(i)

    return not_valid_moves


def get_winner(state):
    """
    :param state: 2D list (default 6 rows and 7 columns)
    :return: returns the tile of the winner or 0 in case there is no winner
    """
    height = len(state)
    width = len(state[0])

    # check horizontal
    for x in range(width - 3):
        for y in range(height):
            if state[y][x] == state[y][x+1] == state[y][x+2] == state[y][x+3] != 0:
                return state[y][x]

    # check vertical
    for x in range(width):
        for y in range(height - 3):
            if state[y][x] == state[y+1][x] == state[y+2][x] == state[y+3][x] != 0:
                return state[y][x]

    # check / diagonal
    for x in range(width - 3):
        for y in range(height - 3):
            if state[y][x] == state[y+1][x+1] == state[y+2][x+2] == state[y+3][x+3] != 0:
                return state[y][x]

    # check \ diagonal
    for x in range(3, width - 3):
        for y in range(height - 3):
            if state[y][x] == state[y+1][x-1] == state[y+2][x-2] == state[y+3][x-3] != 0:
                return state[y][x]
    
    return 0


def make_move(state, action, tile):
    """
    :param state: 2D list (default 6 rows and 7 columns)
    :param action: (int) index of the column where to put the tile
    :param tile:  (int) tile to put
    :return: state after taking the action
    """
    new_state = deepcopy(state)
    if new_state[0][action] == 0:
        for i in reversed(range(len(new_state))):
            if new_state[i][action] == 0:
                new_state[i][action] = tile
                return new_state


def game_over(state):
    """
    checks whether there are any valid moves left or if there is a winner
    :param state: 2D list (default 6 rows and 7 columns)
    :return: True if the game is over, False otherwise
    """
    return get_winner(state) or not get_valid_moves(state)


def get_initial_state():
    """
    :return: a 6x7 list filled with 0
    """
    return [[0 for _ in range(7)] for __ in range(6)]
