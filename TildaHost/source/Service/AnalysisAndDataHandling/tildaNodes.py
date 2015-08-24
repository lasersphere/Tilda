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
import PyQtGraphPlotter
# import MPLPlotter

# import matplotlib.pyplot as mpl
import numpy as np
import logging
import copy


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
        buf = np.zeros((len(data),), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'),
                                            ('headerIndex', 'u1'), ('payload', 'u4')])
        for i, j in enumerate(data):
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
        self.maxArraySize = 5000
        self.nOfSaves = -1

    def processData(self, data, pipeData):
        if self.nOfSaves < 0:  # save pipedata, first time something is feeded to the pipelins
            self.nOfSaves = filhandl.savePipeData(pipeData, self.nOfSaves)
            pipeData['activeTrackPar'] = form.addWorkingTimeToTrackDict(pipeData['activeTrackPar'])
        self.buf = np.append(self.buf, data)
        if self.buf.size > self.maxArraySize:  # when buffer is full, store the data to disc
            self.nOfSaves = filhandl.saveRawData(self.buf, pipeData, self.nOfSaves)
            self.buf = np.zeros(0, dtype=np.uint32)
        return data

    def clear(self, pipeData):
        filhandl.saveRawData(self.buf, pipeData, 0)
        filhandl.savePipeData(pipeData, 0)  # also save the pipeData when clearing
        self.nOfSaves = -1
        self.buf = np.zeros(0, dtype=np.uint32)


class NFilterDataForPipeData(Node):
    def __init__(self):
        """
        if a dictionary is feeded to this node, the PipeData gets updated by this one.

        input: list of rawdata or pipedata dictionaries
        output: passes everything except dictionaries
        """
        self.pipe = super(NFilterDataForPipeData, self).__init__()
        self.type = "FilterPipeData"

    def processData(self, data, pipeData):
        if type(data) == dict:
            pipeData.update(data)
            data = None
        return data


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
        self.timeArray = np.arange(pipeData['activeTrackPar']['delayticks'] * 10,
                                   (pipeData['activeTrackPar']['delayticks'] * 10 + pipeData['activeTrackPar'][
                                       'nOfBins'] * 10), 10,
                                   dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     pipeData['activeTrackPar']['nOfBins'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)

    def processData(self, data, pipeData):
        for i, j in enumerate(data):
            if j['headerIndex'] == 1:  # not MCS/TRS data
                if j['firstHeader'] == progConfigsDict.programs['errorHandler']:  # error send from fpga
                    print('fpga sends error code: ' + str(j['payload']))
                elif j['firstHeader'] == progConfigsDict.programs['dac']:  # its a voltag step than
                    pipeData['activeTrackPar']['nOfCompletedSteps'] += 1
                    self.curVoltIndex, self.voltArray = form.findVoltage(j['payload'], self.voltArray)
            elif j['headerIndex'] == 0:  # MCS/TRS Data
                self.scalerArray = form.trsSum(j, self.curVoltIndex, self.scalerArray,
                                               pipeData['activeTrackPar']['activePmtList'])
        if trsAna.checkIfScanComplete(pipeData):
            return (self.voltArray, self.timeArray, self.scalerArray)
        else:
            return None

    def clear(self, pipeData):
        self.curVoltIndex = 0
        self.voltArray = np.zeros(pipeData['activeTrackPar']['nOfSteps'], dtype=np.uint32)
        self.timeArray = np.arange(pipeData['activeTrackPar']['delayticks'] * 10,
                                   (pipeData['activeTrackPar']['delayticks'] * 10 + pipeData['activeTrackPar'][
                                       'nOfBins'] * 10),
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
        pipeInternals = pipeData['pipeInternals']
        file = pipeInternals['activeXmlFilePath']
        rootEle = filhandl.loadXml(file)
        form.xmlAddCompleteTrack(rootEle, pipeData, data)
        filhandl.saveXml(rootEle, file, False)
        return data


class NAcquireOneScanCS(Node):
    def __init__(self, pipeData):
        """
        sum up all scaler events for the incoming data
        input: splitted rawData
        output: complete Loop of CS-Data, tuple of (voltArray, scalerArray)
        """
        super(NAcquireOneScanCS, self).__init__()
        self.type = 'AcquireOneLoopCS'
        self.bufIncoming = np.zeros((0,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'),
                                                 ('headerIndex', 'u1'), ('payload', 'u4')])
        self.voltArray = np.full(pipeData['activeTrackPar']['nOfSteps'], (2 ** 30), dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)
        self.curVoltIndex = 0
        self.totalnOfScalerEvents = 0

    def processData(self, data, pipeData):
        ret = None
        self.bufIncoming = np.append(self.bufIncoming, data, axis=0)
        for i, j in enumerate(copy.copy(self.bufIncoming)):
            if j['firstHeader'] == progConfigsDict.programs['errorHandler']:  # error send from fpga
                logging.error('fpga sends error code: ' + str(j['payload']) + 'or in binary: ' + str(
                    '{0:032b}'.format(j['payload'])))
                self.bufIncoming = np.delete(self.bufIncoming, 0, 0)

            elif j['firstHeader'] == progConfigsDict.programs['dac']:  # its a voltage step
                self.curVoltIndex, self.voltArray = form.findVoltage(j['payload'], self.voltArray)
                logging.debug('new Voltageindex: ' + str(self.curVoltIndex) + ' ... with voltage: ' + str(
                    form.getVoltageFrom24Bit(j['payload'])))
                self.bufIncoming = np.delete(self.bufIncoming, 0, 0)

            elif j['firstHeader'] == progConfigsDict.programs['continuousSequencer']:
                '''scaler entry '''
                self.totalnOfScalerEvents += 1
                pipeData['activeTrackPar']['nOfCompletedSteps'] = self.totalnOfScalerEvents // 8  # floored Quotient
                try:  # sort values in array, will fail if pmt value is not set active in the activePmtList
                    pmtIndex = pipeData['activeTrackPar']['activePmtList'].index(j['secondHeader'])
                    self.scalerArray[self.curVoltIndex, pmtIndex] += j['payload']
                except ValueError:
                    pass
                self.bufIncoming = np.delete(self.bufIncoming, 0, 0)
                if csAna.checkIfScanComplete(pipeData, self.totalnOfScalerEvents):
                    # one Scan over all steps is completed, add Data to return array and clear local buffer.
                    if ret is None:
                        ret = []
                    ret.append(self.scalerArray)
                    logging.debug('Voltindex: ' + str(self.curVoltIndex) +
                                  'completede steps:  ' + str(pipeData['activeTrackPar']['nOfCompletedSteps']))
                    self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                                 len(pipeData['activeTrackPar']['activePmtList'])),
                                                dtype=np.uint32)
        return ret

    def clear(self, pipeData):
        self.voltArray = np.full(pipeData['activeTrackPar']['nOfSteps'], (2 ** 30), dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)
        self.curVoltIndex = 0
        self.totalnOfScalerEvents = 0
        if np.count_nonzero(self.bufIncoming) > 0:
            logging.warning('Scan not finished, while clearing. Data left: ' + str(self.bufIncoming))
        self.bufIncoming = np.zeros((0,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'),
                                                 ('headerIndex', 'u1'), ('payload', 'u4')])


class NSortRawDatatoArray(Node):
    def __init__(self, pipeData):
        """
        Node for sorting the splitted raw data into an scaler Array. MIssing Values will be set to 0.
        input: split raw data
        output: list of scalerArrays, missing values are 0
        """
        super(NSortRawDatatoArray, self).__init__()
        self.type = 'NSortRawDatatoArray'
        self.bufIncoming = np.zeros((0,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'),
                                                 ('headerIndex', 'u1'), ('payload', 'u4')])
        self.voltArray = np.full(pipeData['activeTrackPar']['nOfSteps'], (2 ** 30), dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)
        self.curVoltIndex = 0
        self.totalnOfScalerEvents = 0

    def processData(self, data, pipeData):
        ret = None
        self.bufIncoming = np.append(self.bufIncoming, data, axis=0)
        for i, j in enumerate(copy.copy(self.bufIncoming)):
            if j['firstHeader'] == progConfigsDict.programs['errorHandler']:  # error send from fpga
                logging.error('fpga sends error code: ' + str(j['payload']) + 'or in binary: ' + str(
                    '{0:032b}'.format(j['payload'])))
                self.bufIncoming = np.delete(self.bufIncoming, 0, 0)

            elif j['firstHeader'] == progConfigsDict.programs['dac']:  # its a voltage step
                self.curVoltIndex, self.voltArray = form.findVoltage(j['payload'], self.voltArray)
                logging.debug('new Voltageindex: ' + str(self.curVoltIndex) + ' ... with voltage: ' + str(
                    form.getVoltageFrom24Bit(j['payload'])))
                self.bufIncoming = np.delete(self.bufIncoming, 0, 0)

            elif j['firstHeader'] == progConfigsDict.programs['continuousSequencer']:
                '''scaler entry '''
                self.totalnOfScalerEvents += 1
                pipeData['activeTrackPar']['nOfCompletedSteps'] = self.totalnOfScalerEvents // 8  # floored Quotient
                try:  # sort values in array, will fail if pmt value is not set active in the activePmtList
                    pmtIndex = pipeData['activeTrackPar']['activePmtList'].index(j['secondHeader'])
                    self.scalerArray[self.curVoltIndex, pmtIndex] += j['payload']
                except ValueError:
                    pass
                self.bufIncoming = np.delete(self.bufIncoming, 0, 0)
                if csAna.checkIfScanComplete(pipeData, self.totalnOfScalerEvents):
                    # one Scan over all steps is completed, add Data to return array and clear local buffer.
                    if ret is None:
                        ret = []
                    ret.append(self.scalerArray)
                    logging.debug('Voltindex: ' + str(self.curVoltIndex) +
                                  'completede steps:  ' + str(pipeData['activeTrackPar']['nOfCompletedSteps']))
                    self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                                 len(pipeData['activeTrackPar']['activePmtList'])),
                                                dtype=np.uint32)
        if ret is None:
            ret = []
        ret.append(self.scalerArray)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                                 len(pipeData['activeTrackPar']['activePmtList'])),
                                                dtype=np.uint32)
        return ret

    def clear(self, pipeData):
        self.voltArray = np.full(pipeData['activeTrackPar']['nOfSteps'], (2 ** 30), dtype=np.uint32)
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)
        self.curVoltIndex = 0
        self.totalnOfScalerEvents = 0
        if np.count_nonzero(self.bufIncoming) > 0:
            logging.warning('Scan not finished, while clearing. Data left: ' + str(self.bufIncoming))
        self.bufIncoming = np.zeros((0,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'),
                                                 ('headerIndex', 'u1'), ('payload', 'u4')])


# dont like that there will be arrays with leading zero's
# create a node which accumulates those half done arrays and passes them to the plotting
# be careful though not to add up scalers twice.
# create arithmetric scaler Node


class NSumCS(Node):
    def __init__(self, pipeData):
        """
        function to sum up all incoming complete Scans
        input: complete Scans
        output: scalerArray containing the sum of each scaler, voltage
        """
        super(NSumCS, self).__init__()
        self.type = 'SumCS'
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)

    def processData(self, data, pipeData):
        for i, j in enumerate(data):  # data can be a list of completed scans
            self.scalerArray = np.add(self.scalerArray, j)
            logging.debug('sum is: ' + str(self.scalerArray[0:2]) + str(self.scalerArray[-2:]))
        return self.scalerArray

    def clear(self, pipeData):
        self.scalerArray = np.zeros((pipeData['activeTrackPar']['nOfSteps'],
                                     len(pipeData['activeTrackPar']['activePmtList'])),
                                    dtype=np.uint32)

class NCheckIfTrackComplete(Node):
    def __init__(self):
        """
        this will only pass scalerArrays to the next node, if the track is complete.
        """
        super(NCheckIfTrackComplete, self).__init__()
        self.type = 'CheckIfTrackComplete'

    def processData(self, data, pipeData):
        ret = None
        if csAna.checkIfTrackComplete(pipeData):
            ret = data
        return ret


# class NPlotSum(Node):
#     def __init__(self, pipeData):
#         """
#         function to plot the sum of all incoming complete Scans
#         input: sum
#         output: complete Sum, when Track is finished
#         """
#         super(NPlotSum, self).__init__()
#         self.type = 'PlotSum'
#         trackd = pipeData['activeTrackPar']
#         dacStart18Bit = trackd['dacStartRegister18Bit']
#         dacStepSize18Bit = trackd['dacStepSize18Bit']
#         nOfsteps = trackd['nOfSteps']
#         dacStop18Bit = dacStart18Bit + (dacStepSize18Bit * nOfsteps)
#         self.x = np.arange(dacStart18Bit, dacStop18Bit, dacStepSize18Bit)
#
#     def processData(self, data, pipeData):
#         logging.info('plotting...')
#         MPLPlotter.plot((self.x, data))
#         file = pipeData['pipeInternals']['activeXmlFilePath'][:-4] + '.png'
#         logging.info('saving plot to' + file)
#         MPLPlotter.save(file)
#         # MPLPlotter.show()
#         return data
#
#     def clear(self, pipeData):
#         MPLPlotter.clear()


class NSaveSumCS(Node):
    def __init__(self):
        """
        function to save all incoming CS-Sum-Data
        input: complete Sum of one track
        output: same as input
        """
        super(NSaveSumCS, self).__init__()
        self.type = 'SaveSumCS'

    def processData(self, data, pipeData):
        pipeData['activeTrackPar'] = form.addWorkingTimeToTrackDict(pipeData['activeTrackPar'])
        pipeInternals = pipeData['pipeInternals']
        file = pipeInternals['activeXmlFilePath']
        rootEle = filhandl.loadXml(file)
        logging.info('saving data: ' + str(data))
        form.xmlAddCompleteTrack(rootEle, pipeData, data)
        filhandl.saveXml(rootEle, file, False)
        logging.info('saving Continous Sequencer Sum to: ' + str(file))
        return data


class NLivePlot(Node):
    def __init__(self, pipeData, pltTitle):
        """
        function to plot a sorted scaler Array
        input: sorted scaler Array
        output: same as input
        """
        super(NLivePlot, self).__init__()
        self.type = 'LivePlot'
        trackd = pipeData['activeTrackPar']
        self.x = form.createXAxisFromTrackDict(trackd)
        win = pipeData['pipeInternals']['activeGraphicsWindow']
        self.pl = PyQtGraphPlotter.addPlot(win, pltTitle)

    def processData(self, data, pipeData):
        logging.info('plotting...')
        PyQtGraphPlotter.plot(self.pl, (self.x, data), clear=True)
        return data

    def clear(self, pipeData):
        pass
