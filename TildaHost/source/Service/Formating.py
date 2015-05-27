'''
Created on 21.01.2015

@author: skaufmann
'''

import numpy as np


def split32bData(int32bData):
    """
    seperate header, headerIndex and payload from each other
    :param int32bData:
    :return: tuple, (firstHeader, secondHeader, headerIndex, payload)
    """
    headerLength = 8
    firstHeader = int32bData >> (32 - int(headerLength/2))
    secondHeader = int32bData >> (32 - headerLength) & ((2 ** 4) - 1)
    headerIndex = (int32bData >> (32 - headerLength - 1)) & 1
    payload = int32bData & ((2 ** 23) - 1)
    return (firstHeader, secondHeader, headerIndex, payload)

