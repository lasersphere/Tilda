"""
Created on 

@author: simkaufm

Module Description:
"""

import itertools
import random
import os

# Custom classes
from Scratch.SnakeClasses import (BonusQGraphics, SnakeDict,
                                  SnakepartQGraphics, SquareQGrapics)


def add_bonus(bonuses, snake, squares):
    # Create set of empty squares coordinates
    free_squares = squares.copy()
    obstacles = itertools.chain(snake.values(), bonuses.values())
    for obstacle in obstacles:
        if obstacle.xy in free_squares:
            free_squares.remove(obstacle.xy)

    # After set is comleted randomly place bonus
    free_squares = tuple(free_squares)
    bonus_icon = os.path.join(os.path.dirname(__file__), 'snake_bonus.jpg')
    new_bonus = BonusQGraphics(xy=random.choice(free_squares), icon=bonus_icon)
    bonuses[new_bonus.xy] = new_bonus
    return new_bonus


def create_bonuses(snake, squares):
    bonuses = {}
    for call in range(2):
        add_bonus(bonuses, snake, squares)
    return bonuses


def create_squares(cols, rows):
    squares = {}
    for x in range(cols):
        for y in range(rows):
            squares[(x, y)] = SquareQGrapics((x, y))
    squares_xy, squares = set(squares.keys()), set(squares.values())
    return squares_xy, squares


def create_snake():
    snake = SnakeDict()
    x, y = (0, 3)
    head_icon = os.path.join(os.path.dirname(__file__), 'snake_head.jpg')
    snake[0] = SnakepartQGraphics((x, y), icon=head_icon)  # head
    for n in range(1, 3):
        y += 1
        snake[n] = SnakepartQGraphics((x, y))
    return snake


def print_main():
    text = ('''<p>Welcome to Snake game!</p>

            <p>To start or unpause game press &quot;Start&quot; or
            &quot;Enter&quot; key. Use arrow keys to move snake.</p>

            <p>To pause game press &quot;P&quot; key.</p>

            <p>Click &quot;Main menu&quot; to read this message.</p>

            <p>Click &quot;Rules&quot; to read them.</p>

            <p>Press &quot;Reset&quot; or &quot;R&quot; key to reset game
            field (arrow keys will be frozen).</p>''')
    return text


def print_results(snake):
    score = len(snake) - 3
    text = ('''<p>You end up with score:</p>
            <p>{}</p>'''.format(score))
    return text


def print_rules():
    text = ('''<p>Player gains score when snake eats apple.</p>

            <p>When this happens snake aslo grows by one tile.</p>

            <p>Game ends when snake runs in the border or itself.</p>''')
    return text
