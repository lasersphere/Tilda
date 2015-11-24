"""

Created on '20.05.2015'

@author:'simkaufm'

"""
from Service.FileFormat.XmlOperations import xmlAddCompleteTrack
from Service.VoltageConversions.VoltageConversions import find_volt_in_array
import Service.Scan.ScanDictionaryOperations as SdOp
from Measurement.SpecData import SpecData

from polliPipe.node import Node
import Service.Formating as form
import Service.FolderAndFileHandling as filhandl
import Service.ProgramConfigs as progConfigsDict
import Service.AnalysisAndDataHandling.trsDataAnalysis as trsAna
import Service.AnalysisAndDataHandling.csDataAnalysis as csAna
import MPLPlotter

import numpy as np
import time
import logging
from copy import copy, deepcopy


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
            result = form.split_32b_data(j)
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
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        if self.nOfSaves < 0:  # save pipedata, first time something is fed to the pipelins
            self.nOfSaves = filhandl.savePipeData(pipeData, self.nOfSaves)
            pipeData[track_name] = form.add_working_time_to_track_dict(pipeData[track_name])
        self.buf = np.append(self.buf, data)
        if self.buf.size > self.maxArraySize:  # when buffer is full, store the data to disc
            self.nOfSaves = filhandl.saveRawData(self.buf, pipeData, self.nOfSaves)
            self.buf = np.zeros(0, dtype=np.uint32)
        return data

    def clear(self):
        filhandl.saveRawData(self.buf, self.Pipeline.pipeData, 0)
        filhandl.savePipeData(self.Pipeline.pipeData, 0)  # also save the pipeData when clearing
        self.nOfSaves = -1
        self.buf = np.zeros(0, dtype=np.uint32)


class NSumBunchesTRS(Node):
    def __init__(self):
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

    def start(self):
        pipeData = self.pipeData
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
                    self.curVoltIndex, self.voltArray = find_volt_in_array(j['payload'], self.voltArray)
            elif j['headerIndex'] == 0:  # MCS/TRS Data
                self.scalerArray = form.trs_sum(j, self.curVoltIndex, self.scalerArray,
                                                pipeData['activeTrackPar']['activePmtList'])
        if trsAna.checkIfScanComplete(pipeData):
            return (self.voltArray, self.timeArray, self.scalerArray)
        else:
            return None

    def clear(self):
        pipeData = self.Pipeline.pipeData
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
        xmlAddCompleteTrack(rootEle, pipeData, data)
        filhandl.saveXml(rootEle, file, False)
        return data


class NAcquireOneScanCS(Node):
    def __init__(self):
        """
        not up to date anymore. not sure if it is used though.
        sum up all scaler events for the incoming data
        input: splitted rawData
        output: list of completed scalerArrays
        """
        super(NAcquireOneScanCS, self).__init__()
        self.type = 'AcquireOneLoopCS'
        self.bufIncoming = None
        self.voltArray = None
        self.scalerArray = None
        self.curVoltIndex = 0
        self.totalnOfScalerEvents = 0
        logging.error('watch out, ' + self.type + ' might be outdated')

    def start(self):
        scand = self.Pipeline.pipeData
        self.voltArray = form.create_x_axis_from_scand_dict(scand)
        self.scalerArray = form.create_default_scaler_array_from_scandict(scand)
        self.bufIncoming = np.zeros((0,), dtype=[('firstHeader', 'u1'), ('secondHeader', 'u1'),
                                                 ('headerIndex', 'u1'), ('payload', 'u4')])

    def processData(self, data, pipeData):
        ret = None
        self.bufIncoming = np.append(self.bufIncoming, data, axis=0)
        for i, j in enumerate(copy(self.bufIncoming)):
            if j['firstHeader'] == progConfigsDict.programs['errorHandler']:  # error send from fpga
                logging.error('fpga sends error code: ' + str(j['payload']) + 'or in binary: ' + str(
                    '{0:032b}'.format(j['payload'])))
                self.bufIncoming = np.delete(self.bufIncoming, 0, 0)

            elif j['firstHeader'] == progConfigsDict.programs['dac']:  # its a voltage step
                self.curVoltIndex, self.voltArray = find_volt_in_array(j['payload'], self.voltArray)
                # logging.debug('new Voltageindex: ' + str(self.curVoltIndex) + ' ... with voltage: ' + str(
                # form.get_voltage_from_24bit(j['payload'])))
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

    def clear(self):
        pipeData = self.Pipeline.pipeData
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
    def __init__(self):
        """
        Node for sorting the splitted raw data into an scaler Array containing all tracks.
        Missing Values will be set to 0.
        No Value will be emitted twice.
        input: split raw data
        output: list of tuples [(scalerArray, scan_complete_flag)... ], missing values are 0
        """
        super(NSortRawDatatoArray, self).__init__()
        self.type = 'NSortRawDatatoArray'
        self.voltArray = None
        self.scalerArray = None
        self.curVoltIndex = 0
        self.totalnOfScalerEvents = []

    def start(self):
        scand = self.Pipeline.pipeData
        tracks, tracks_num_list = SdOp.get_number_of_tracks_in_scan_dict(scand)
        self.totalnOfScalerEvents = np.full((tracks,), 0)
        self.voltArray = form.create_default_volt_array_from_scandict(scand)
        self.scalerArray = form.create_default_scaler_array_from_scandict(scand)


    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        ret = None
        scan_complete = False
        for i, j in enumerate(data):
            if j['firstHeader'] == progConfigsDict.programs['errorHandler']:  # error send from fpga
                logging.error('fpga sends error code: ' + str(j['payload']) + 'or in binary: ' + str(
                    '{0:032b}'.format(j['payload'])))

            elif j['firstHeader'] == progConfigsDict.programs['dac']:  # its a voltage step
                self.curVoltIndex, self.voltArray = find_volt_in_array(j['payload'], self.voltArray, track_ind)

            elif j['firstHeader'] == progConfigsDict.programs['continuousSequencer']:
                '''scaler entry '''
                self.totalnOfScalerEvents[track_ind] += 1
                pipeData[track_name]['nOfCompletedSteps'] = self.totalnOfScalerEvents[
                                                                track_ind] // 8  # floored Quotient
                try:  # only add to scalerArray, when pmt is in activePmtList
                    pmt_index = pipeData[track_name]['activePmtList'].index(j['secondHeader'])
                    self.scalerArray[track_ind][pmt_index][self.curVoltIndex] += j['payload']  # PolliFit conform
                except ValueError:
                    pass
                if csAna.checkIfScanComplete(pipeData, self.totalnOfScalerEvents[track_ind], track_name):
                    # one Scan over all steps is completed, add Data to return array and clear local buffer.
                    scan_complete = True
                    if ret is None:
                        ret = []
                    ret.append((self.scalerArray, scan_complete))
                    logging.debug('Voltindex: ' + str(self.curVoltIndex) +
                                  'completede steps:  ' + str(pipeData[track_name]['nOfCompletedSteps']))
                    self.scalerArray = form.create_default_scaler_array_from_scandict(pipeData)
                    scan_complete = False
        if ret is None:
            ret = []
        if np.count_nonzero(self.scalerArray):
            ret.append((self.scalerArray, scan_complete))
        self.scalerArray = form.create_default_scaler_array_from_scandict(pipeData)
        return ret

    def clear(self):
        self.voltArray = None
        self.scalerArray = None
        self.curVoltIndex = 0
        self.totalnOfScalerEvents = []


class NSumCS(Node):
    def __init__(self):
        """
        function to sum up all scalerArrays.
        Since no value is emitted twice, arrays can be directly added
        input: list of scalerArrays, complete or uncomplete
        output: scalerArray containing the sum of each scaler, voltage
        """
        super(NSumCS, self).__init__()
        self.type = 'SumCS'
        self.scalerArray = None

    def start(self):
        self.scalerArray = form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)

    def processData(self, data, pipeData):
        for i, j in enumerate(data):  # data can be a list of completed scans
            self.scalerArray = np.add(self.scalerArray, j)
            logging.debug('sum is: ' + str(self.scalerArray[0:2]) + str(self.scalerArray[-2:]))
        return self.scalerArray

    def clear(self):
        self.scalerArray = form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)


class NRemoveTrackCompleteFlag(Node):
    def __init__(self):
        """
        removes the scan_complete_flag from the incoming tuple.
        input: list of tuples [(scalerArray, scan_complete_flag)... ], missing values are 0
        output: list of scalerArrays, complete or uncomplete
        """

        super(NRemoveTrackCompleteFlag, self).__init__()
        self.type = 'NRemoveTrackCompleteFlag'

    def processData(self, data, pipeData):
        data = [d for d, f in data]
        return data


class NCheckIfTrackComplete(Node):
    def __init__(self):
        """
        this will only pass scalerArrays to the next node, if the track is complete.
        """
        super(NCheckIfTrackComplete, self).__init__()
        self.type = 'CheckIfTrackComplete'

    def processData(self, data, pipeData):
        ret = None
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        if csAna.checkIfTrackComplete(pipeData, track_name):
            ret = data
        return ret


class NPlotSum(Node):
    def __init__(self):
        """
        function to plot the sum of all incoming complete Scans
        input: sum
        output: complete Sum, when Track is finished
        """
        super(NPlotSum, self).__init__()
        self.type = 'PlotSum'
        self.x = None

    def start(self):
        self.x = form.create_x_axis_from_track_dict(self.Pipeline.pipeData['activeTrackPar'])

    def processData(self, data, pipeData):
        logging.info('plotting...')
        MPLPlotter.plot((self.x, data))
        file = pipeData['pipeInternals']['activeXmlFilePath'][:-4] + '.png'
        logging.info('saving plot to' + file)
        MPLPlotter.save(file)
        MPLPlotter.show()
        return data

    def clear(self):
        MPLPlotter.clear()


class NMPlLivePlot(Node):
    def __init__(self, ax, title, plt_styles_list):
        """
        Node for plotting live Data using matplotlib.pyplot
        input: list, [(x1, y1), (x2, y2),... ] x and y are numpy arrays
        output: input
        """
        super(NMPlLivePlot, self).__init__()
        self.type = 'MPlLivePlot'

        self.ax = ax
        self.title = title
        self.plotStyles = plt_styles_list
        self.ax.set_ylabel(self.title)
        self.x = None
        self.y = None

    def start(self):
        # l = form.create_x_axis_from_scand_dict(self.Pipeline.pipeData)
        # self.x = [item for sublist in l for item in sublist]
        MPLPlotter.ion()

    def animate(self, plotlist):
        # self.ax.clear()
        # self.ax.plot(x, y)
        # self.ax.set_ylabel(self.title)
        # plt.pause(0.0001)
        MPLPlotter.plt_axes(self.ax, self.title, plotlist)
        # time.sleep(0.2)
        MPLPlotter.pause(0.0001)

    def processData(self, data, pipeData):
        logging.debug('plotting...')
        t = time.time()
        plot_list = []  # [(x1, y1), (x2, y2), ....]
        i = 0
        for dat in data:
            plot_list.append(dat[0])
            plot_list.append(dat[1])
            plot_list.append(self.plotStyles[i])
            i += 1
        self.animate(plot_list)
        logging.debug('plotting time (ms):' + str(round((time.time() - t) * 1000, 0)))
        return data

    def stop(self):
        MPLPlotter.show(block=True)  # this only work if pipeline thread is main thread. :(


class NSaveSumCS(Node):
    def __init__(self):
        """
        function to save all incoming CS-Sum-Data.
        will also add working time to each track on start() and on processData(...).
        input: complete, scalerArray containing all tracks.
        output: same as input
        """
        super(NSaveSumCS, self).__init__()
        self.type = 'SaveSumCS'

    def start(self):
        tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(self.Pipeline.pipeData)
        for tr in track_num_list:
            track_name = 'track' + str(tr)
            self.Pipeline.pipeData[track_name] = form.add_working_time_to_track_dict(
                self.Pipeline.pipeData[track_name])

    def processData(self, data, pipeData):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        pipeData[track_name] = form.add_working_time_to_track_dict(pipeData[track_name])
        pipeInternals = pipeData['pipeInternals']
        file = pipeInternals['activeXmlFilePath']
        rootEle = filhandl.loadXml(file)
        logging.info('saving data: ' + str(data))
        xmlAddCompleteTrack(rootEle, pipeData, data[track_ind], track_name)
        filhandl.saveXml(rootEle, file, False)
        logging.info('saving Continous Sequencer Sum to: ' + str(file))
        return data


class NAccumulateSingleScan(Node):
    def __init__(self):
        """
        accumulates a singel scan. This is mostly used for plotting the current scan.
        input: list of tuples [(scalerArray, scan_complete)... ], missing values are 0
        output: scalerArray, missing values are 0
        """
        super(NAccumulateSingleScan, self).__init__()
        self.type = 'AccumulateSingleScan'

        self.scalerArray = None

    def start(self):
        self.scalerArray = form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        ret = None
        for i, j in enumerate(data):
            if not j[1]:  # work on incomplete Scan
                self.scalerArray[track_ind] = np.add(self.scalerArray[track_ind], j[0][track_ind])
                ret = self.scalerArray  # return last incoming incomplete scan
            elif j[1]:  # work with complete scan
                self.scalerArray[track_ind] = np.add(self.scalerArray[track_ind], j[0][track_ind])
                ret = self.scalerArray
                # return complete scan and reset scalerarray
                self.scalerArray = form.create_default_scaler_array_from_scandict(pipeData)
        return ret

    def clear(self):
        self.scalerArray = form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)


class NSingleSpecFromSpecData(Node):
    def __init__(self, scalers):
        """
        will return a single spectrum of the given scalers
        a tuple with (volt, cts, err) of the specified scaler and track. -1 for all tracks
        input: SpecData
        ouptut: tuple, (volt, cts, err)
        """
        super(NSingleSpecFromSpecData, self).__init__()
        self.type = 'SingleSpecFromSpecData'
        self.scalers = scalers

    def processData(self, spec_data_instance, pipeData):
        ret = []
        x, y, err = spec_data_instance.getArithSpec(self.scalers, -1)
        ret.append((np.array(x), y))
        return ret


class NMultiSpecFromSpecData(Node):
    def __init__(self, scalers):
        """
        will return a single spectrum of the given scalers
        of the specified scaler and track. -1 for all tracks.
        scalers should be list of lists -> [[0, 1], [0]].
        each element as syntax as in getArithSpec in specData
        input: SpecData
        ouptut: list, [(x1, y1), (x2, y2),... ]
        """
        super(NMultiSpecFromSpecData, self).__init__()
        self.type = 'MultiSpecFromSpecData'
        self.scalers = scalers

    def processData(self, spec_data_instance, pipeData):
        ret = []
        for sc in self.scalers:
            x, y, err = spec_data_instance.getArithSpec(sc, -1)
            ret.append((np.array(x), y))
        return ret


class NSingleArrayToSpecData(Node):
    def __init__(self):
        """
        when started, will init a SpecData object of given size.
        Overwrites SpecData.cts and passes the SpecData object to the next node.
        input: scalerArray, containing all tracks
        output: SpecData
        """
        super(NSingleArrayToSpecData, self).__init__()
        self.type = 'SingleArrayToSpecData'
        self.spec_data = None

    def start(self):
        self.spec_data = SpecData()
        self.spec_data.path = self.Pipeline.pipeData['pipeInternals']['activeXmlFilePath']
        self.spec_data.type = self.Pipeline.pipeData['isotopeData']['type']
        tracks, tr_num_list = SdOp.get_number_of_tracks_in_scan_dict(self.Pipeline.pipeData)
        self.spec_data.nrLoops = [self.Pipeline.pipeData['track' + str(tr_num)]['nOfScans']
                                  for tr_num in tr_num_list]
        self.spec_data.nrTracks = tracks
        self.spec_data.accVolt = [self.Pipeline.pipeData['track' + str(tr_num)]['postAccOffsetVolt']
                                  for tr_num in tr_num_list]
        self.spec_data.laserFreq = self.Pipeline.pipeData['isotopeData']['laserFreq']
        self.spec_data.col = [self.Pipeline.pipeData['track' + str(tr_num)]['colDirTrue']
                              for tr_num in tr_num_list]
        self.spec_data.x = form.create_x_axis_from_scand_dict(self.Pipeline.pipeData)
        self.spec_data.cts = form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)
        self.spec_data.err = form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)

    def processData(self, data, pipeData):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        self.spec_data.nrScalers = len(self.Pipeline.pipeData[track_name]['activePmtList'])
        self.spec_data.cts = data
        return self.spec_data


class NScanProgWinUpdate(Node):
    def __init__(self, scan_prog_win):
        """
        update the scanprogess window by looking into the pipeData
        """
        super(NScanProgWinUpdate, self).__init__()
        self.type = 'ScanProgWinUpdate'
        self.scp_win = scan_prog_win

    def start(self):
        self.update(self.Pipeline.pipeData)

    def update(self, scand):
        tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(scand)
        self.scp_win.set_n_of_total_tracks(tracks)
