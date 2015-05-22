'''
Created on 21.01.2015

@author: skaufmann
'''

import numpy as np

class Formatter():
    def split32bData(self, int32bData):
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

    def integerSplitHeaderInfo(self, int32bData):
        """
        Turns 32-Bit incoming Data from the Target to Host fifo from the Fpga into integers.
        :param int32bData:
        :return: tuple, with 3 Elements.
        tuple[0] = bool, if True it is MCS data.
        tuple[1] = list, either list of active pmts for mcs data or
                    tuple, int, first part of header and second part of header
        tuple[2] = int, 23 Bit timestamp for mcs Data or 23 Bit other Information program relevant.
        """
        headerLength = 8
        headerIndex = (int32bData & (2 ** 23)) != 0
        value = int32bData & ((2 ** 23) - 1)
        header = int32bData >> (32 - headerLength)
        if not headerIndex:
            #mcs data
            timestamp = value
            activepmts = [i for i in range(headerLength) if (header & (2 ** i)) > 0]
            return (headerIndex, activepmts, timestamp)
        else:
            firstHeader = header >> int(headerLength/2)
            secondHeader = header & ((2 ** 4) - 1)
            return (headerIndex, (firstHeader, secondHeader), value)


# string = b'10011101100010000000000010000000'
# exampleData = int(string, 2)
# print(len(bin(exampleData)[2:]))
# print(Formatter().integerSplitHeaderInfo(exampleData))
# print(Formatter().split32bData(exampleData)['payload'])

# print(timeit.timeit(lambda: Formatter().binaryDataToInt(exampleData), number=1000000))
# print(timeit.timeit(lambda: Formatter().integerSplitHeaderInfo(exampleData), number=1000000))
