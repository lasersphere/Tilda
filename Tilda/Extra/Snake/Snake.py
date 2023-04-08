"""
Created on 

@author: simkaufm

Module Description:

Game name: PainterFiled
Author: Igor Vasylchenko

As this module and its sub-modules use PyQt5 they are distributed and
may be used under the terms of the GNU General Public License version 3.0.
See http://www.gnu.org/copyleft/gpl.html
"""

import sys

from PyQt5.QtCore import (QTimer, Qt)
from PyQt5.QtWidgets import (QApplication, QGraphicsScene,
                             QMainWindow)

# Generated gui module
from Tilda.Extra.Snake.SnakeGui import Ui_Field as Ui_MainWindow

# Custom classes and functions
from Tilda.Extra.Snake.SnakeFunctions import \
    (add_bonus, create_bonuses, create_squares, create_snake, print_main, print_results, print_rules)


'''Exceptions hadling block. Needed to track errors during
operation.'''

'''sys._excepthook = sys.excepthook
def exception_hook(exctype, value, traceback):
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)
sys.excepthook = exception_hook'''


class Snake(QMainWindow, Ui_MainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # Important property that deletes all widgets (including QTimer)
        # on close.
        # self.setAttribute(Qt.WA_DeleteOnClose)

        self.setFocusPolicy(Qt.StrongFocus)

        # Variables
        self.end = False
        self.timer_interval = 350
        self.key = Qt.Key_Up
        self.snake = create_snake()
        self.scene = QGraphicsScene()
        self.squares_xy, squares = create_squares(9, 9)

        self.bonuses = create_bonuses(self.snake, self.squares_xy)

        '''Timer is used to repeat self.turn_loop and detect if key was changed.
        It is stopped by default and starts after self.start() is called.'''
        self.timer = QTimer(self)

        self.draw_field(self.bonuses, squares, self.snake)
        self.QGraph_field.setScene(self.scene)
        self.print_text(print_main())

        # Connecting signals to slots
        self.QBut_main.clicked.connect(lambda: self.print_text(print_main()))
        self.QBut_reset.clicked.connect(self.reset)
        self.QBut_rules.clicked.connect(lambda: self.print_text(print_rules()))
        self.QBut_start.clicked.connect(self.start)

        self.timer.timeout.connect(self.turn_loop)

    def draw_field(self, bonuses, squares, snake):
        for square in squares:
            self.scene.addItem(square)
        for snakepart in snake.values():
            self.scene.addItem(snakepart)
        for bonus in bonuses.values():
            self.scene.addItem(bonus)

    def keyPressEvent(self, event):
        key_map = {Qt.Key_Enter:  self.start,  Qt.Key_Return: self.start,
                   Qt.Key_Escape: self.close,  Qt.Key_P:      self.pause,
                   Qt.Key_R:      self.reset}
        key = event.key()

        # Snake head movement
        if key in {Qt.Key_Left, Qt.Key_Right,
                   Qt.Key_Up,   Qt.Key_Down}:
            self.key = key

        # Start, pause, reset or exit game
        elif key in key_map:
            key_map[key]()

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()

    def print_text(self, source):
        self.QText_status.setHtml(source)

    def reset(self):
        self.timer.stop()

        self.end = False
        self.key = Qt.Key_Up
        self.snake = create_snake()
        self.squares_xy, squares = create_squares(9, 9)

        self.bonuses = create_bonuses(self.snake, self.squares_xy)

        self.scene.clear()
        self.draw_field(self.bonuses, squares, self.snake)
        self.print_text(print_main())

    def start(self):
        if not self.end and not self.timer.isActive():
            self.timer.start(self.timer_interval)

    def turn_loop(self):
        isCollided, self.key = self.snake.collides(self.key)
        head = self.snake[0]
        if isCollided:
            self.timer.stop()
            self.end = True
            self.print_text(print_results(self.snake))
        else:
            empty_xy = self.snake.move(self.key)
            eaten_bonus = self.snake.eats_bonus(self.bonuses)
            if eaten_bonus is not None:
                self.scene.removeItem(eaten_bonus)
                new_tail = self.snake.grow(empty_xy)
                new_bonus = add_bonus(self.bonuses, self.snake, self.squares_xy)
                self.scene.addItem(new_tail)
                self.scene.addItem(new_bonus)

    def focusOutEvent(self, *args, **kwargs):
        self.pause()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Snake()
    window.show()
    sys.exit(app.exec_())
