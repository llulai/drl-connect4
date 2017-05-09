from PIL import Image
from PIL import ImageDraw
import pandas as pd
import matplotlib.pyplot as plt


def draw_board(board):
    def draw_tile(i, j, color='white'):
        draw.ellipse([(2 + j * 9, 2 + i * 9), ((j + 1) * 9, (i + 1) * 9)], fill=color)

    colors = {-1: 'red', 0: 'white', 1: 'yellow'}
    image = Image.new('RGB', (65, 56), 'blue')
    draw = ImageDraw.Draw(image)

    for i in range(len(board)):
        for j in range(len(board[i])):
            color = colors[board[i][j]]
            draw_tile(i, j, color)

    return image


def draw_game(game, path='logs/game.gif'):
    images = []
    for board in game:
        images.append(draw_board(board))

    images[0].save(path, save_all=True, append_images=images, duration=500)


def get_reward(data, sparring='search 0 agent', opponent='search 0 agent', kind='testing'):
    ts0 = data[data['sparring'] == sparring]
    ts0 = ts0[ts0['current_opponent'] == opponent]
    ts0 = ts0[ts0['kind'] == kind]
    return ts0.groupby('iteration')['reward'].sum()


def get_won(data, sparring='search 0 agent', opponent='search 0 agent', kind='testing'):
    ts0 = data[data['sparring'] == sparring]
    ts0 = ts0[ts0['current_opponent'] == opponent]
    ts0 = ts0[ts0['kind'] == kind]
    ts0 = ts0[ts0['reward'] == 1]
    ts0 = ts0.rename(columns={'reward': 'won games'})
    return ts0.groupby('iteration')['won games'].count()


def get_stats(data, sparring='search 0 agent', opponent='search 0 agent', kind='testing'):
    won = get_won(data, sparring=sparring, opponent=opponent, kind=kind)
    reward = get_reward(data, sparring=sparring, opponent=opponent, kind=kind)
    return pd.concat([won, reward], axis=1).fillna(0)


def plot_stats(data, sparring='search 0 agent', opponent='search 0 agent', kind='testing'):
    title = 'sparring: {} \nopponent: {}'.format(sparring, opponent)
    get_stats(data, sparring=sparring, opponent=opponent).plot(title=title)
    plt.show()
