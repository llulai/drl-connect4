class Menu:
    def __init__(self):
        self.options = ['start_game', 'dificulty', 'exit']
        self.current_option = 0
        self.active_arrow = True

    def select_next_option(self):
        options = len(self.options)
        if self.current_option + 1 == options:
            self.current_option = 0
        else:
            self.current_option += 1

    def select_previous_option(self):
        options = len(self.options)
        if self.current_option == 0:
            self.current_option = options -1
        else:
            self.current_option -= 1
