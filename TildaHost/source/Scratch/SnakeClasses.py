"""
Created on 

@author: simkaufm

Module Description:
"""

from PyQt5.QtCore    import (QRectF, Qt)
from PyQt5.QtGui     import (QBrush, QColor, QImage)
from PyQt5.QtWidgets import QGraphicsItem

class BonusQGraphics(QGraphicsItem):
    def __init__(self, xy, icon='graphics/bonus.png'):
        QGraphicsItem.__init__(self)
        self.icon = icon
        self.xy = xy

    def boundingRect(self):
        x, y = self.xy
        return QRectF(x*30, y*30, 28, 28)

    def paint(self, painter, option, widget):
        x, y = self.xy
        target = QRectF(x*30, y*30, 28, 28)
        source = QRectF(0, 0, 28, 28)
        painter.drawImage(target, QImage(self.icon), source)

class SnakeDict(dict):
    def __init__(self):
        dict.__init__(self)

    def collides(self, key):
        '''Prevents snake head from moving backwards (collision of head
        with previous body block) and detects its collision with bounds or
        snake body. Returns True and key=None if collision. Otherwise
        returns False and key.'''

        head, tail = self[0], self[len(self)-1]
        isCollided = False

        opposite_keys = {Qt.Key_Up: Qt.Key_Down,
                         Qt.Key_Down: Qt.Key_Up,
                         Qt.Key_Left: Qt.Key_Right,
                         Qt.Key_Right:Qt.Key_Left
                         }
        # Preventing moving backwards (inside itself)
        x, y, angle = head.prepareGoto(key)
        if (x, y) == self[1].xy:
            key = opposite_keys[key]
            x, y, angle = head.prepareGoto(key)

        # Collision with bounds
        bounds_set = {(x<0), (x>8), (y<0), (y>8)}
        if True in bounds_set:
            isCollided, key = True, None
            return isCollided, key

        # Collision with body, tail will move next turn so it is not counted
        for snakepart in self.values():
            if snakepart not in (head, tail):
                if (x, y) == snakepart.xy:
                    isCollided, key = True, None
                    break

        # No collision
        return isCollided, key

    def eats_bonus(self, bonuses):
        head = self[0]
        if head.xy in bonuses.keys(): # Head eats bonus
            eaten_bonus = bonuses.pop(head.xy)
        else:
            eaten_bonus = None
        return eaten_bonus

    def grow(self, empty_xy):
        new_tail = SnakepartQGraphics(empty_xy)
        self[len(self)] = new_tail
        return new_tail

    def move(self, key):
        '''Returns last coordinate at which new snakepart can be added.'''

        head = self[0]
        new_xy = head.goto(key)
        body = sorted(self.items())[1:]

        for ID, snakepart in body:
            old_xy = snakepart.xy
            new_xy = snakepart.gotoxy(new_xy)
        return old_xy

class SnakepartQGraphics(QGraphicsItem):
    def __init__(self, xy, icon=None):
        QGraphicsItem.__init__(self)
        self.icon = icon
        self.xy = xy

    def boundingRect(self):
        x, y = self.xy
        return QRectF(x*30, y*30, 28, 28)

    def goto(self, key):
        '''Based on prepareGoto() moves snakepart to calculated x, y and
        rotates it. Returns initial x, y.'''

        prev_xy = self.xy[:]
        x, y, angle = self.prepareGoto(key)
        self.prepareGeometryChange()
        self.xy = x, y
        self.setTransformOriginPoint(self.boundingRect().center())
        self.setRotation(angle)
        return prev_xy

    def gotoxy(self, xy, angle=0):
        '''Moves to designated x, y. Returns initial x, y.'''

        prev_xy = self.xy[:]
        self.prepareGeometryChange()
        self.xy = xy
        return prev_xy

    def paint(self, painter, option, widget):
        x, y = self.xy
        if self.icon is None:
            colour = QBrush(QColor('green'))
            painter.setBrush(colour)
            painter.drawRect(x*30, y*30, 28, 28)
        target = QRectF(x*30, y*30, 28, 28)
        source = QRectF(0, 0, 28, 28)
        painter.drawImage(target, QImage(self.icon), source)

    def prepareGoto(self, key):
        '''Calculates x, y and rotation (for head)'''

        x, y = self.xy
        moves = {Qt.Key_Up: (x, y-1, 0), Qt.Key_Down: (x, y+1, 180),
                 Qt.Key_Left: (x-1, y, -90), Qt.Key_Right:(x+1, y, 90)}

        x, y, angle = moves[key]
        return x, y, angle

# Arbitrary class. Is used just to quickly draw rectangles.
class SquareQGrapics(QGraphicsItem):
    def __init__(self, xy=(-1,-1), colour='cyan'):
        QGraphicsItem.__init__(self)
        self.colour = colour
        self.xy = xy

    def boundingRect(self):
        x, y = self.xy
        return QRectF(x*30, y*30, 28, 28)

    def paint(self, painter, option, widget):
        x, y = self.xy
        colour = QBrush(QColor(self.colour))
        painter.setBrush(colour)
        painter.drawRect(x*30, y*30, 28, 28)