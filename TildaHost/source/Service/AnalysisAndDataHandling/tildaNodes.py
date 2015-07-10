"""

Created on '20.05.2015'

@author:'simkaufm'

"""

from polliPipe.node import Node
import Service.Formating as form
import Service.FolderAndFileHandling as filhandl
import Service.ProgramConfigs as progConfigsDict
import Service.AnalysisAndDataHandling.trsDataAnalysis as trsAna
import Service.AnalysisAndDataHandling.csDataAnalysis as csAna

import numpy as np
import logging



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
        logging.debug('Node Name: ' + self.type + ' ... processing now')
        buf = np.zeros((len(data),), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'),
                                            ('headerIndex', 'u1'), ('payload', 'u4')])
        for i,j in enumerate(data):
            result = form.split32bData(j)
            buf[i] = result
        return buf


class NSaveRawData(Node):
    def __init__(self):
        """
        Node to store incoming Raw Data in a Binary File.
        Passes all rawdata onto next node.
        input: list of rawData
        output: list of rawData
        """
        super(NSaveRawData, self).__init__()
        self.type = "SaveRawData"

        self.buf = np.zeros(0, dtype=np.uint32)
        self.maxArraySize = 10

    def processData(self, data, pipeData):
        logging.debug('Node Name: ' + self.type + ' ... processing now')
        self.buf = np.append(self.buf, data)
        if self.buf.size > self.maxArraySize:
            print('saving to: ', filhandl.savePickle(self.buf, pipeData))
            self.clear(pipeData)
        return data

    def clear(self, pipeData):
        self.buf = np.zeros(0, dtype=np.uint32)


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
                      (pipeData['activeTrackPar']['delayticks']*10 + pipeData['activeTrackPar']['nOfBins']*10), 10,
                      dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     pipeData['activeTrackPar']['nOfBins'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)

    def processData(self, data, pipeData):
        logging.debug('Node Name: ' + self.type + ' ... processing now')
        for i, j in enumerate(data):
            if j['headerIndex'] == 1:  # not MCS/TRS data
                if j['firstHeader'] == progConfigsDict.programs['errorHandler']:  # error send from fpga
                    print('fpga sends error code: ' + str(j['payload']))
                elif j['firstHeader'] == progConfigsDict.programs['dac']:  # its a voltag step than
                    pipeData['activeTrackPar']['nOfCompletedSteps'] += 1
                    self.curVoltIndex, self.voltArray = form.findVoltage(j['payload'], self.voltArray)
            elif j['headerIndex'] == 0:   # MCS/TRS Data
                self.scalerArray = form.trsSum(j, self.curVoltIndex, self.scalerArray,
                                               pipeData['activeTrackPar']['activePmtList'])
        if trsAna.checkIfScanComplete(pipeData):
            return (self.voltArray, self.timeArray, self.scalerArray)
        else:
            return None

    def clear(self, pipeData):
        self.curVoltIndex = 0
        self.voltArray = np.zeros(pipeData['activeTrackPar']['nOfSteps'], dtype=np.uint32)
        self.timeArray = np.arange(pipeData['activeTrackPar']['delayticks']*10,
                      (pipeData['activeTrackPar']['delayticks']*10 + pipeData['activeTrackPar']['nOfBins']*10),
                      10, dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     pipeData['activeTrackPar']['nOfBins'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)


class NSaveTrsSum(Node):
    def __init__(self):
        """
        save the summed up data
        incoming must always be a tuple of form:
        (self.voltArray, self.timeArray, self.scalerArray)
        Note: will always save when data is passed to it. So only send complete structures.
        """
        super(NSaveTrsSum, self).__init__()
        self.type = "SaveTrsSum"

    def processData(self, data, pipeData):
        logging.debug('Node Name: ' + self.type + ' ... processing now')
        pipeInternals = pipeData['pipeInternals']
        file = pipeInternals['activeXmlFilePath']
        rootEle = filhandl.loadXml(file)
        form.xmlAddCompleteTrack(rootEle, pipeData, data)
        filhandl.saveXml(rootEle, file, False)
        return data

    def clear(self, pipeData):
        pass


class NAcquireOneLoopCS(Node):
    def __init__(self, pipeData):
        """
        sum up all scaler events for the incoming data
        input: splitted rawData
        output: complete Loop of CS-Data, tuple of (voltArray, scalerArray)
        """
        super(NAcquireOneLoopCS, self).__init__()
        self.type = 'AcquireOneLoopCS'
        self.voltArray = np.zeros(pipeData['activeTrackPar']['nOfSteps'], dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)
        self.curVoltIndex = 0
        self.totalnOfScalerEvents = 0

    def processData(self, data, pipeData):
        logging.debug('Node Name: ' + self.type + ' ... processing now')
        for i, j in enumerate(data):
            if j['firstHeader'] == progConfigsDict.programs['errorHandler']:  # error send from fpga
                print('fpga sends error code: ' + str(j['payload']))
            elif j['firstHeader'] == progConfigsDict.programs['dac']:  # its a voltage step than
                pipeData['activeTrackPar']['nOfCompletedSteps'] += 1
                self.curVoltIndex, self.voltArray = form.findVoltage(j['payload'], self.voltArray)
            elif j['firstHeader'] == progConfigsDict.programs['continuousSequencer']:
                self.totalnOfScalerEvents += 1
                pipeData['activeTrackPar']['nOfCompletedSteps'] = self.totalnOfScalerEvents // 8  # floored Quotient
                try:  # will fail if pmt value is not set active in the activePmtList
                    pmtIndex = pipeData['activeTrackPar']['activePmtList'].index(j['secondHeader'])
                    self.scalerArray[self.curVoltIndex, pmtIndex] += j['payload']
                except IndexError:
                    pass
        if csAna.checkIfScanComplete(pipeData):
            # one Scan over all steps is completed, transfer Data to next node.
            ret = self.scalerArray
            self.clear(pipeData)
            return ret
        else:
            return None

    def clear(self, pipeData):
        self.voltArray = np.zeros(pipeData['activeTrackPar']['nOfSteps'], dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)
        self.curVoltIndex = 0
        self.totalnOfScalerEvents = 0


class NSumCS(Node):
    def __init__(self, pipeData):
        """
        function to sum up all incoming
        """
        super(NSumCS, self).__init__()
        self.type = 'SumCS'

        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)

    def processData(self, data, pipeData):
        logging.debug('Node Name: ' + self.type + ' ... processing now')
        self.scalerArray = np.add(self.scalerArray, data[1])
        if csAna.checkIfTrackComplete(pipeData):
            return self.scalerArray
        else:
            return None

    def clear(self, pipeData):
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)


class NSaveSumCS(Node):
    def __init__(self):
        """
        function to save all incoming CS-Data
        """
        super(NSaveSumCS, self).__init__()

    def processData(self, data, pipeData):
        logging.debug('Node Name: ' + self.type + ' ... processing now')
        pipeInternals = pipeData['pipeInternals']
        file = pipeInternals['activeXmlFilePath']
        rootEle = filhandl.loadXml(file)
        form.xmlAddCompleteTrack(rootEle, pipeData, data)
        filhandl.saveXml(rootEle, file, False)
        print('saving Continous Sequencer Sum to: ', file)
        return data