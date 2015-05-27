"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import numpy as np


from polliPipe.node import Node
from Service.Formating import split32bData


class NSplit32bData(Node):
    def __init__(self):
        """
        Constructor
        """
        super(NSplit32bData, self).__init__()
        self.type = "Split32bData"

        self.buf = np.zeros((1,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'), ('headerIndex', 'u1'), ('payload', 'u4')])

    def processData(self, data, pipeData):
        """
        convert rawData to list of tuples of the 4 informations:
        [(firstHeader, secondHeader, headerIndex, payload)]
        """
        self.buf = np.resize(self.buf, (len(data),))
        for i,j in enumerate(data):
            result = split32bData(j)
            self.buf[i] = result
        return self.buf

    def clear(self):
        self.buf = np.zeros((1,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'), ('headerIndex', 'u1'), ('payload', 'u4')])

class NSumBunches(Node):
    def __init__(self):
        """
        Constructor
        """
        super(NSumBunches, self).__init__()
        self.type = "SumBunches"

        self.buf = np.zeros((1,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'), ('headerIndex', 'u1'), ('payload', 'u4')])

    def processData(self, data, pipeData):
        """
        sum up all events for each scaler and check for new voltage
        """
        self.buf = data
        curVoltInd = pipeData['curVoltInd']
        print('current Volt index is: ' + str(curVoltInd))


    def clear(self):
        self.buf = np.zeros((1,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'), ('headerIndex', 'u1'), ('payload', 'u4')])