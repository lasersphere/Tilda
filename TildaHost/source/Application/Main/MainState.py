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
    setting_switch_box = 24
    measure_offset_voltage = 23
    load_track = 12
    scanning = 13
    saving = 14

    preparing_tilda_passiv = 15
    tilda_passiv_running = 16
    closing_tilda_passiv = 17

    init_dmm = 18
    config_dmm = 19
    reading_dmm = 20  # might not be used
    deinit_dmm = 21
    request_dmm_config_pars = 22

