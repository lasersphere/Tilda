'''
Created on 21.01.2015

@author: skaufmann
'''

import numpy as np

class Formatter():
    def split32bData(self, int32bData):
        """
        seperate header, headerindex and payload from each other
        :param int32bData:
        :return: tuple, (firstheader, secondheader, headerindex, payload)
        """
        headerlength = 8
        firstheader = int32bData >> (32 - int(headerlength/2))
        secondheader = int32bData >> (32 - headerlength) & ((2 ** 4) - 1)
        headerindex = (int32bData >> (32 - headerlength - 1)) & 1
        payload = int32bData & ((2 ** 23) - 1)
        return (firstheader, secondheader, headerindex, payload)

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
        headerlength = 8
        headerindex = (int32bData & (2 ** 23)) == 0
        value = int32bData & ((2 ** 23) - 1)
        header = int32bData >> (32 - headerlength)
        if headerindex:
            #mcs data
            timestamp = value
            activepmts = [i for i in range(headerlength) if (header & (2 ** i)) > 0]
            return (headerindex, activepmts, timestamp)
        else:
            progheader = header >> int(headerlength/2)
            secondheader = header & ((2 ** 4) - 1)
            return (headerindex, (progheader, secondheader), value)


# string = b'10011101100010000000000010000000'
# exampleData = int(string, 2)
# print(len(bin(exampleData)[2:]))
# print(Formatter().integerSplitHeaderInfo(exampleData))
# print(Formatter().split32bData(exampleData)['payload'])

# print(timeit.timeit(lambda: Formatter().binaryDataToInt(exampleData), number=1000000))
# print(timeit.timeit(lambda: Formatter().integerSplitHeaderInfo(exampleData), number=1000000))
