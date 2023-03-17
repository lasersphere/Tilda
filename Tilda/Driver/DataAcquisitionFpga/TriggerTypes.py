"""
Created on 

@author: simkaufm

Module Description: Module for storage of the trigger type enum
Names and values must be unique.
Values must represent the state number in labview.
https://docs.python.org/3/library/enum.html
"""

from enum import Enum, unique


@unique
class TriggerTypes(Enum):
    no_trigger = 0
    single_hit_delay = 1
    single_hit = 2
    software = 3
    sweep = 4
