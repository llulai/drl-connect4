import pygame


class SpriteSheet(object):
    def __init__(self, filename):
        try:
            self.sheet = pygame.image.load(filename).convert()
        except pygame.error, message:
            print 'Unable to load spritesheet image:', filename
            raise SystemExit, message

    # Load a specific image from a specific rectangle
    def image_at(self, rectangle, colorkey=None):
        "Loads image from x,y,x+offset,y+offset"
        rect = pygame.Rect(rectangle)
        image = pygame.Surface(rect.size).convert()
        image.blit(self.sheet, (0, 0), rect)
        if colorkey is not None:
            if colorkey is -1:
                colorkey = image.get_at((0,0))
            image.set_colorkey(colorkey, pygame.RLEACCEL)
        return image

    # Load a whole bunch of images and return them as a list
    def images_at(self, rects, colorkey=None):
        "Loads multiple images, supply a list of coordinates"
        return [self.image_at(rect, colorkey) for rect in rects]

    # Load a whole strip of images
    def load_strip(self, rect, image_count, colorkey=None):
        "Loads a strip of images and returns them as a list"
        tups = [(rect[0]+rect[2]*x, rect[1], rect[2], rect[3])
                for x in range(image_count)]
        return self.images_at(tups, colorkey)


class TextWriter(object):
    def __init__(self, filename, SCREEN):
        self.corpus = pygame.image.load(filename).convert()
        self.SCREEN = SCREEN
        self.abc = self.__get_font__()
        self.font_size = self.__get_font_size()

    def __get_font__(self):
        abc = {
            'a': self.__get_letter_at((0, 0, 5, 7)),
            'b': self.__get_letter_at((5, 0, 5, 7)),
            'c': self.__get_letter_at((10, 0, 5, 7)),
            'd': self.__get_letter_at((15, 0, 5, 7)),
            'e': self.__get_letter_at((20, 0, 4, 7)),
            'f': self.__get_letter_at((24, 0, 4, 7)),
            'g': self.__get_letter_at((28, 0, 5, 7)),
            'h': self.__get_letter_at((33, 0, 5, 7)),
            'i': self.__get_letter_at((38, 0, 2, 7)),
            'j': self.__get_letter_at((40, 0, 5, 7)),
            'k': self.__get_letter_at((45, 0, 5, 7)),
            'l': self.__get_letter_at((50, 0, 4, 7)),
            'm': self.__get_letter_at((54, 0, 6, 7)),
            'n': self.__get_letter_at((60, 0, 5, 7)),
            'o': self.__get_letter_at((65, 0, 5, 7)),
            'p': self.__get_letter_at((70, 0, 5, 7)),
            'q': self.__get_letter_at((75, 0, 5, 7)),
            'r': self.__get_letter_at((80, 0, 5, 7)),
            's': self.__get_letter_at((85, 0, 5, 7)),
            't': self.__get_letter_at((90, 0, 6, 7)),
            'u': self.__get_letter_at((96, 0, 5, 7)),
            'v': self.__get_letter_at((101, 0, 6, 7)),
            'w': self.__get_letter_at((107, 0, 6, 7)),
            'x': self.__get_letter_at((113, 0, 5, 7)),
            'y': self.__get_letter_at((118, 0, 5, 7)),
            'z': self.__get_letter_at((123, 0, 5, 7)),
            '1': self.__get_letter_at((128, 0, 4, 7)),
            '2': self.__get_letter_at((132, 0, 5, 7)),
            '3': self.__get_letter_at((137, 0, 5, 7)),
            '4': self.__get_letter_at((142, 0, 5, 7)),
            '5': self.__get_letter_at((147, 0, 5, 7)),
            '6': self.__get_letter_at((152, 0, 5, 7)),
            '7': self.__get_letter_at((157, 0, 5, 7)),
            '8': self.__get_letter_at((162, 0, 5, 7)),
            '9': self.__get_letter_at((167, 0, 5, 7)),
            '0': self.__get_letter_at((172, 0, 5, 7)),
        }

        return abc

    def __get_font_size(self):
        sizes = {
            'a': {'width': 5, 'height': 7},
            'b': {'width': 5, 'height': 7},
            'c': {'width': 5, 'height': 7},
            'd': {'width': 5, 'height': 7},
            'e': {'width': 4, 'height': 7},
            'f': {'width': 4, 'height': 7},
            'g': {'width': 5, 'height': 7},
            'h': {'width': 5, 'height': 7},
            'i': {'width': 2, 'height': 7},
            'j': {'width': 5, 'height': 7},
            'k': {'width': 5, 'height': 7},
            'l': {'width': 4, 'height': 7},
            'm': {'width': 6, 'height': 7},
            'n': {'width': 5, 'height': 7},
            'o': {'width': 5, 'height': 7},
            'p': {'width': 5, 'height': 7},
            'q': {'width': 5, 'height': 7},
            'r': {'width': 5, 'height': 7},
            's': {'width': 5, 'height': 7},
            't': {'width': 6, 'height': 7},
            'u': {'width': 5, 'height': 7},
            'v': {'width': 6, 'height': 7},
            'w': {'width': 6, 'height': 7},
            'x': {'width': 5, 'height': 7},
            'y': {'width': 5, 'height': 7},
            'z': {'width': 5, 'height': 7},
            '1': {'width': 4, 'height': 7},
            '2': {'width': 5, 'height': 7},
            '3': {'width': 5, 'height': 7},
            '4': {'width': 5, 'height': 7},
            '5': {'width': 5, 'height': 7},
            '6': {'width': 5, 'height': 7},
            '7': {'width': 5, 'height': 7},
            '8': {'width': 5, 'height': 7},
            '9': {'width': 5, 'height': 7},
            '0': {'width': 5, 'height': 7},
        }

        return sizes

    def __get_letter_at(self, rectangle):
        rect = pygame.Rect(rectangle)
        image = pygame.Surface(rect.size).convert()
        image.blit(self.corpus, (0, 0), rect)
        return image

    def write(self, pos=(0, 0), text=''):
        current_pos = pos

        for letter in text:
            self.SCREEN.blit(self.abc[letter], current_pos)
            current_pos = self.update_pos(current_pos, letter)
            print(current_pos)

    def update_pos(self, current_pos, letter):
        letter_width = self.font_size[letter]['width']
        current_pos = (current_pos[0] + letter_width, current_pos[1])
        return current_pos



class Game:
    def __init__(self):
        self.SCREEN = pygame.display.set_mode((700, 600))
        self.game_over = False
        self.text_writer = TextWriter('font/abecedarium2.png', self.SCREEN)

        self.COLORS = {
            'black': (0,0,0),
            'white': (255,255,255)
        }

        self. clock = pygame.time.Clock()

    def write_text(self, text):
        self.screen.blit(self.font.render(text, True, (255, 0, 0)), (200, 100))

    def start(self):
        print('starting game...')
        pygame.init()
        pygame.display.set_caption('conectar cuatro')
        self.SCREEN.fill(self.COLORS['white'])

        self.text_writer.write((100, 100), 'a')
        self.text_writer.write((100, 110), '6')
        self.text_writer.write((100, 120), '7')
        self.text_writer.write((100, 130), '8')
        self.text_writer.write((100, 140), '9')
        self.text_writer.write((100, 150), '0')

        while not self.game_over:
            self.menu_loop()

        self.finish_game()

    def menu_loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True

            print(event)

        pygame.display.update()
        self.clock.tick(60)

    def finish_game(self):
        pygame.quit()
        quit()


def main():
    game = Game()
    game.start()


if __name__ == '__main__':
    main()

