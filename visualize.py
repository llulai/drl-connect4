from PIL import Image
from PIL import ImageDraw


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
