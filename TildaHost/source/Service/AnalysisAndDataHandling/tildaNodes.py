"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import numpy as np


from polliPipe.node import Node
import Service.Formating as form
from Service.ProgramConfigs import programs as progConfigs


class NSplit32bData(Node):
    def __init__(self):
        """
        convert rawData to list of tuples of the 4 informations:
        [(firstHeader, secondHeader, headerIndex, payload)]
        """
        super(NSplit32bData, self).__init__()
        self.type = "Split32bData"

        self.buf = np.zeros((1,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'), ('headerIndex', 'u1'), ('payload', 'u4')])

    def processData(self, data, pipeData):
        self.buf = np.resize(self.buf, (len(data),))
        for i,j in enumerate(data):
            result = form.split32bData(j)
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
        for i,j in enumerate(self.buf):
            if j['headerIndex'] == 1:
                """
                not MCS data
                """
                if j['firstHeader'] == progConfigs['errorHandler']:
                    print('fpga sends error code: ' + str(j['payload']))
                elif j['firstHeader'] == progConfigs['simpleCounter']:
                    pass
                elif j['firstHeader'] == progConfigs['continuousSequencer']:
                    pass
                elif j['firstHeader'] == progConfigs['dac']:
                    index, voltArray = form.findVoltage(j['payload'], pipeData['voltArray'])
                    pipeData.update(curVoltInd=index, voltArray=voltArray)
            elif j['headerIndex'] == 0:
                """
                MCS Data
                """
                pipeData.update(scalerArray=form.mcsSum(j, pipeData['curVoltInd'], pipeData['scalerArray']))
                print(np.max(pipeData['scalerArray']))

    def clear(self):
        self.buf = np.zeros((1,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'), ('headerIndex', 'u1'), ('payload', 'u4')])