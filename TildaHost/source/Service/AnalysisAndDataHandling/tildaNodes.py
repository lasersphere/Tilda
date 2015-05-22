"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import numpy as np


from polliPipe.node import Node
from Service.Formating import Formatter



class NrawFormatToReadable(Node):
    def __init__(self):
        """
        Constructor
        """
        super(NrawFormatToReadable, self).__init__()
        self.type = "rawFormatToReadable"
        self.form = Formatter()

        self.buf = []

    def processData(self, data, pipeData):
        """
        convert rawData to a readable form
        """
        self.buf = [self.form.integerSplitHeaderInfo(j) for i,j in enumerate(data)]
        return self.buf

    def clear(self):
        self.buf = []

class NSplit32bData(Node):
    def __init__(self):
        """
        Constructor
        """
        super(NSplit32bData, self).__init__()
        self.type = "Split32bData"
        self.form = Formatter()

        self.buf = np.zeros((1,), dtype=[('firstheader', 'u1'), ('secondheader', 'u1'), ('headerIndex', 'u1'), ('payload', 'u4')])

    def processData(self, data, pipeData):
        """
        convert rawData to a readable form
        """
        self.buf = np.resize(self.buf, (len(data),))
        for i,j in enumerate(data):
            result =  self.form.split32bData(j)
            self.buf[i] = result
        return self.buf

    def clear(self):
        self.buf = np.zeros((1,), dtype=[('firstheader', 'u1'), ('secondheader', 'u1'), ('headerIndex', 'u1'), ('payload', 'u4')])


class NSumBunches(Node):
    def __init__(self):
        """
        Constructor
        """
        super(NSumBunches, self).__init__()
        self.type = "SumBunches"

        self.buf = []

    def processData(self, data, pipeData):
        """
        sum up all events for each scaler and check for new voltage
        """
        pipeData['curVoltInd']
        return self.buf

    def clear(self):
        self.buf = []