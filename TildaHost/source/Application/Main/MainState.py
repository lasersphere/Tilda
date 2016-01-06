"""
Created on 

@author: simkaufm

Module Description:
Module for defining all Main states.
Names and values must be unique.
https://docs.python.org/3/library/enum.html
"""
from enum import Enum, unique

@unique
class MainState(Enum):
    init = 1
    idle = 2
    error = 3

    starting_simple_counter = 4
    simple_counter_running = 5
    stop_simple_counter = 6

    init_power_supplies = 7
    setting_power_supply = 8
    reading_power_supply = 9
    set_output_power_sup = 10

    preparing_scan = 11
    load_track = 12
    scanning = 13
