"""
Created on

@author: fsommer

Module Description: Module for storage of the scan device type enum
Names and values must be unique.
Values must represent the state number in labview.
https://docs.python.org/3/library/enum.html
"""

from enum import Enum, unique


@unique
class ScanDeviceTypes(Enum):
    DAC = 0
    Triton = 1
