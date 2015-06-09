"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import numpy as np


from polliPipe.node import Node
import Service.Formating as form



class NSplit32bData(Node):
    def __init__(self):
        """
        convert rawData to list of tuples, see output
        input: list of rawData
        output: [(firstHeader, secondHeader, headerIndex, payload)]
        """
        super(NSplit32bData, self).__init__()
        self.type = "Split32bData"


    def processData(self, data, pipeData):
        buf = np.zeros((len(data),), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'), ('headerIndex', 'u1'), ('payload', 'u4')])
        for i,j in enumerate(data):
            result = form.split32bData(j)
            buf[i] = result
        return buf

class NSumBunchesTRS(Node):
    def __init__(self, pipeData):
        """
        sort all incoming events into scalerArray and voltArray.
        All Scaler events will be summed up seperatly for each scaler.
        Is specially build for the TRS data.

        Input: List of tuples: [(firstHeader, secondHeader, headerIndex, payload)]
        output: tuple of npArrays, (voltArray, timeArray, scalerArray)
        """
        super(NSumBunchesTRS, self).__init__()
        self.type = "SumBunchesTRS"

        self.curVoltIndex = 0
        self.voltArray = np.zeros(pipeData['nOfSteps'], dtype=np.uint32)
        self.timeArray = np.arange(pipeData['delayticks']*10,
                      (pipeData['delayticks']*10 + pipeData['nOfBins']*10),
                      10, dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['nOfSteps'], pipeData['nOfBins'], 8), dtype=np.uint32)

    def processData(self, data, pipeData):
        for i,j in enumerate(data):
            if j['headerIndex'] == 1: #not MCS/TRS data
                if j['firstHeader'] == pipeData['progConfigs']['errorHandler']: #error send from fpga
                    print('fpga sends error code: ' + str(j['payload']))
                elif j['firstHeader'] == pipeData['progConfigs']['dac']: #its a voltag step than
                    pipeData['nOfTotalSteps'] += 1
                    self.curVoltIndex, self.voltArray = form.findVoltage(j['payload'], self.voltArray)
            elif j['headerIndex'] == 0: #MCS/TRS Data
                self.scalerArray = form.trsSum(j, self.curVoltIndex, self.scalerArray)
        return (self.voltArray, self.timeArray, self.scalerArray)


    def clear(self):
        self.curVoltIndex = 0
        self.voltArray = np.zeros(1, dtype=np.uint32)
        self.timeArray = np.arange(0, 1, 1, dtype=np.uint32)
        self.scalerArray = np.zeros((1, 1, 8), dtype=np.uint32)

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
