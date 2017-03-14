import pygame


class TextWriter(object):
    def __init__(self, filename, screen):
        self.font_size = 1
        self.sheet = pygame.image.load(filename).convert()
        self.screen = screen
        self.abc = self.__get_font__()

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
            '?': self.__get_letter_at((177, 0, 4, 7)),
            ':': self.__get_letter_at((182, 0, 2, 7)),
            ' ': self.__get_letter_at((184, 0, 3, 7)),
            '>': self.__get_letter_at((186, 0, 4, 7)),
        }

        return abc

    def __get_letter_at(self, rectangle):
        rect = pygame.Rect(rectangle)
        image = pygame.Surface(rect.size).convert()
        image.blit(self.sheet, (0, 0), rect)
        return image

    def write(self, text='', pos=(0, 0)):
        current_pos = pos

        for letter in text:
            img = self.abc[letter]

            w, h = img.get_size()
            img = pygame.transform.scale(img, (w * self.font_size, h * self.font_size))

            self.screen.blit(img, current_pos)
            current_pos = self.update_pos(current_pos, letter)

    def update_pos(self, current_pos, letter):
        w, h = self.abc[letter].get_size()
        w *= self.font_size
        h *= self.font_size
        current_pos = (current_pos[0] + w, current_pos[1])
        return current_pos

    def set(self, options):
        pass
        #try:
        #    self.font_size = options['font_size']
