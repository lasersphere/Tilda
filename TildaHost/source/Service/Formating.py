'''
Created on 21.01.2015

@author: skaufmann
'''

import timeit

string = b'10011101100000000000000000000000'
exampleData = int(string, 2)
print(len(bin(exampleData)[2:]))

class Formatter():
    def integerSplitHeaderInfo(self, int32bData):
        """

        :param int32bData:
        :return:
        """
        headerlength = 8
        headerindex = (int32bData & (2 ** 23)) > 0
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
            return (headerindex, [progheader, secondheader], value)


print(bin(2 ** 33))
print(Formatter().integerSplitHeaderInfo(exampleData))
# print(timeit.timeit(lambda: Formatter().binaryDataToInt(exampleData), number=1000000))
# print(timeit.timeit(lambda: Formatter().integerSplitHeaderInfo(exampleData), number=1000000))
