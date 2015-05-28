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
        sort all incoming events into scalerArray and voltArray.
        All Scaler events will be summed up seperatly for each scaler.
        Will work for all incoming data. Host has to choose the shape of ScalerArray when selecting
        timeresolved/not timeresolved.
        """
        super(NSumBunches, self).__init__()
        self.type = "SumBunches"

        self.buf = np.zeros((1,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'), ('headerIndex', 'u1'), ('payload', 'u4')])

    def processData(self, data, pipeData):
        self.buf = data
        for i,j in enumerate(self.buf):
            if j['headerIndex'] == 1: #not MCS/TRS data
                if j['firstHeader'] == progConfigs['errorHandler']:
                    print('fpga sends error code: ' + str(j['payload']))
                elif j['firstHeader'] == progConfigs['simpleCounter']:
                    pass
                elif j['firstHeader'] == progConfigs['continuousSequencer']:
                    pass
                elif j['firstHeader'] == progConfigs['dac']:
                    pipeData['nOfTotalSteps'] += 1
                    index, voltArray = form.findVoltage(j['payload'], pipeData['voltArray'])
                    pipeData.update(curVoltInd=index, voltArray=voltArray)
            elif j['headerIndex'] == 0: #MCS/TRS Data
                pipeData.update(scalerArray=form.mcsSum(j, pipeData['curVoltInd'], pipeData['scalerArray']))

    def clear(self):
        self.buf = np.zeros((1,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'), ('headerIndex', 'u1'), ('payload', 'u4')])

class NSaveSum(Node):
    def __init__(self):
        """
        save the summed up data
        """
        super(NSaveSum, self).__init__()
        self.type = "SaveSum"

        self.buf = []

    def processData(self, data, pipeData):
        pass #dont know yet when this can be called...

    def clear(self):
        self.buf = []
