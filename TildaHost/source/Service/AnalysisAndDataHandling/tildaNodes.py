"""

Created on '20.05.2015'

@author:'simkaufm'

"""

import numpy as np


from polliPipe.node import Node
import Service.Formating as form
import Service.FolderAndFileHandling as filhandl
import Service.ProgramConfigs as progConfigsDict



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


class NAccumulateRawData(Node):
    def __init__(self):
        """
        Node to store incoming Raw Data in one Array.
        input: list of rawData
        output: list of rawData
        """
        super(NAccumulateRawData, self).__init__()
        self.type = "AccumulateRawData"

        self.buf = np.zeros(0, dtype=np.uint32)

    def processData(self, data, pipeData):
        self.buf = np.append(self.buf, data)
        return data

    def saveData(self, incomingData, pipeData):
        dataToSave = self.buf
        self.clear(pipeData) #force clear() because data is stored on disc now
        return dataToSave

    def clear(self, pipeData):
        self.buf = np.zeros(0, dtype=np.uint32)

class NSaveRawData(Node):
    def __init__(self):
        """
        Node to save Raw Data using pickle
        input: list of rawData
        output: list of rawData
        """
        super(NSaveRawData, self).__init__()
        self.type = "SaveRawData"

    def saveData(self, incomingData, pipeData):
        filhandl.savePickle(incomingData, pipeData)
        return None


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
        self.voltArray = np.zeros(pipeData['activeTrackPar']['nOfSteps'], dtype=np.uint32)
        self.timeArray = np.arange(pipeData['activeTrackPar']['delayticks']*10,
                      (pipeData['activeTrackPar']['delayticks']*10 + pipeData['activeTrackPar']['nOfBins']*10),
                      10, dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'], pipeData['activeTrackPar']['nOfBins'], len(pipeData['activeTrackPar']['activePmtList'])), dtype=np.uint32)

    def processData(self, data, pipeData):
        for i,j in enumerate(data):
            if j['headerIndex'] == 1: #not MCS/TRS data
                if j['firstHeader'] == progConfigsDict.programs['errorHandler']: #error send from fpga
                    print('fpga sends error code: ' + str(j['payload']))
                elif j['firstHeader'] == progConfigsDict.programs['dac']: #its a voltag step than
                    pipeData['activeTrackPar']['nOfCompletedSteps'] += 1
                    self.curVoltIndex, self.voltArray = form.findVoltage(j['payload'], self.voltArray)
            elif j['headerIndex'] == 0: #MCS/TRS Data
                self.scalerArray = form.trsSum(j, self.curVoltIndex, self.scalerArray, pipeData['activeTrackPar']['activePmtList'])
        return (self.voltArray, self.timeArray, self.scalerArray)

    def saveData(self, incomingData, pipeData):
        """
        if saving is requested, return the needed Arrays
        """
        return (self.voltArray, self.timeArray, self.scalerArray)

    def clear(self, pipeData):
        self.curVoltIndex = 0
        self.voltArray = np.zeros(pipeData['activeTrackPar']['nOfSteps'], dtype=np.uint32)
        self.timeArray = np.arange(pipeData['activeTrackPar']['delayticks']*10,
                      (pipeData['activeTrackPar']['delayticks']*10 + pipeData['activeTrackPar']['nOfBins']*10),
                      10, dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'], pipeData['activeTrackPar']['nOfBins'], len(pipeData['activeTrackPar']['activePmtList'])), dtype=np.uint32)

class NSaveSum(Node):
    def __init__(self, pipeData):
        """
        save the summed up data
        incoming must always be a tuple of form:
        (self.voltArray, self.timeArray, self.scalerArray)
        """
        super(NSaveSum, self).__init__()
        self.type = "SaveSum"

    def saveData(self, incomingData, pipeData):
        pipeInternals = pipeData['pipeInternals']
        file = pipeInternals['activeXmlFilePath']
        rootEle = filhandl.loadXml(file)
        form.xmlWriteToTrack(rootEle, pipeInternals['activeTrackNumber'], 'scalerArray')

        pass

    def clear(self, pipeData):
        pass
