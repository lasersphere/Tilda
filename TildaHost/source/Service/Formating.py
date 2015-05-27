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

def findVoltage(voltage, voltArray):
    """
    find the index of voltage in voltArray. If not existant, create.
    :return: int, index
    """
    '''payload is 23-Bits, Bits 2 to 20 is the DAC register'''
    voltage = (voltage >> 2) & ((2 ** 18) - 1)
    index = np.where(voltArray == voltage)
    if len(index[0]) == 0:
        index = np.where(voltArray == 0)[0][0]
    else:
        index = index[0]
    np.put(voltArray, index, voltage)
    return (index, voltArray)

def mcsSum(element, actVoltInd, sumArray):
    """
    Add new Scaler event on previous acquired ones. Treat each scaler seperatly.
    :return: np.array, sum
    """
    timestamp = element['payload']
    activePmts = (element['firstHeader'] << 4) + element['secondHeader'] #glue header back together
    for i in range(8):
        if activePmts & (2 ** i):
            sumArray[actVoltInd, timestamp, i] += 1
    return sumArray