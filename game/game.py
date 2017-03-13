import pygame
from itertools import cycle
from text_writer import TextWriter
from environment import get_initial_state, make_move, game_over
from agent import  IntelligentAgent

from menu import Menu

class Game:
    def __init__(self):
        self.SCREEN = pygame.display.set_mode((700, 600))
        self.game_over = False
        self.text_writer = TextWriter('font/corpus.png', self.SCREEN)
        self.menu = Menu()

        self.COLORS = {
            'black': (0,0,0),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0 , 255, 0),
            'blue': (0, 0, 255),
        }

        self. clock = pygame.time.Clock()
        self.frame = 0
        self.current_loop = 'menu'

    def print_menu(self):
        self.SCREEN.fill(self.COLORS['white'])
        self.text_writer.font_size = 3
        self.text_writer.write('start game', (100, 100))
        self.text_writer.write('dificulty easy', (100, 150))
        self.text_writer.write('exit', (100, 200))
        if self.menu.active_arrow:
            row = 100 + self.menu.current_option * 50
            self.text_writer.write('>', (85, row))

    def start(self):
        pygame.init()
        pygame.display.set_caption('conectar cuatro')

        while not self.game_over:
            if self.current_loop == 'menu':
                self.menu_loop()
            elif self.current_loop == 'game':
                self.game_loop()

        pygame.quit()
        quit()

    def __start_game(self):
        self.board = get_initial_state()

        self.players = cycle(['human', 'computer'])
        self.current_player = next(self.players)
        self.agent = IntelligentAgent((-1, 1))

    def game_loop(self):
        if self.current_player == 'computer':
            action = self.agent.get_action(self.board)
            self.board = make_move(self.board, action, -1)
            self.current_player = next(self.players)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True

            if not game_over(self.board):

                if event.type == pygame.MOUSEBUTTONUP:
                    col = self.get_col(event)

                    if self.current_player == 'human':
                        self.board = make_move(self.board, col, 1)
                        self.current_player = next(self.players)

        self.print_board()

    def get_col(self, event):
        x, y =  pygame.mouse.get_pos()
        return  int(x / 100)

    def print_board(self):
        self.SCREEN.fill(self.COLORS['white'])
        for row in range(len(self.board)):
            for col in range(len(self.board[row])):
                self.print_tile(row, col)

        pygame.draw.rect(self.SCREEN, self.COLORS['black'], (200, 300, 100, 100))

        if game_over(self.board):
            print('game over')

        pygame.display.update()
        self.clock.tick(24)

    def print_tile(self, row, col):
        tile = self.board[row][col]
        if tile:
            color = self.COLORS['red'] if tile == -1 else self.COLORS['blue']
            pygame.draw.circle(self.SCREEN, color, (50 + col * 100, 50 + row * 100), 45)



    def menu_loop(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.menu.select_next_option()
                elif event.key == pygame.K_UP:
                    self.menu.select_previous_option()

                if event.key == pygame.K_RETURN:

                    selected_option = self.menu.options[self.menu.current_option]

                    if selected_option == 'start_game':
                        self.__start_game()
                        self.current_loop = 'game'
                    elif selected_option == 'exit':
                        self.game_over = True

        if self.frame % 12 == 0:
            self.menu.active_arrow = not self.menu.active_arrow
            self.frame = 0

        self.print_menu()
        pygame.display.update()
        self.clock.tick(24)
        self.frame +=1


def main():
    game = Game()
    game.start()


if __name__ == '__main__':
    main()

