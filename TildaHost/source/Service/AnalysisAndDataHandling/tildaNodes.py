"""

Created on '20.05.2015'

@author:'simkaufm'

"""
import TildaTools
from Service.FileFormat.XmlOperations import xmlAddCompleteTrack
from Service.VoltageConversions.VoltageConversions import find_volt_in_array
import Service.Scan.ScanDictionaryOperations as SdOp
from Measurement.SpecData import SpecData

from polliPipe.node import Node
import Service.Formating as Form
import Service.FolderAndFileHandling as Filehandle
import Service.ProgramConfigs as ProgConfigsDict
import Service.AnalysisAndDataHandling.trsDataAnalysis as TrsAna
import Service.AnalysisAndDataHandling.csDataAnalysis as CsAna
import MPLPlotter

import numpy as np
import time
import logging

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation


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
            result = Form.split_32b_data(j)
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
        self.maxArraySize = 5000

        self.buf = None
        self.nOfSaves = None

    def start(self):
        if self.buf is None:
            self.buf = np.zeros(0, dtype=np.uint32)
        if self.nOfSaves is None:
            self.nOfSaves = -1

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        if self.nOfSaves < 0:  # save pipedata, first time something is fed to the pipelins
            self.nOfSaves = Filehandle.savePipeData(pipeData, self.nOfSaves)
        self.buf = np.append(self.buf, data)
        if self.buf.size > self.maxArraySize:  # when buffer is full, store the data to disc
            self.nOfSaves = Filehandle.saveRawData(self.buf, pipeData, self.nOfSaves)
            self.buf = np.zeros(0, dtype=np.uint32)
        return data

    def clear(self):
        Filehandle.saveRawData(self.buf, self.Pipeline.pipeData, 0)
        Filehandle.savePipeData(self.Pipeline.pipeData, 0)  # also save the pipeData when clearing
        self.nOfSaves = None
        self.buf = None


class NSumBunchesTRS(Node):
    def __init__(self):
        """

        Not up to date anymore need remodeling when working with TRS!!


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
                if j['firstHeader'] == ProgConfigsDict.programs['errorHandler']:  # error send from fpga
                    print('fpga sends error code: ' + str(j['payload']))
                elif j['firstHeader'] == ProgConfigsDict.programs['dac']:  # its a voltag step than
                    pipeData['activeTrackPar']['nOfCompletedSteps'] += 1
                    self.curVoltIndex, self.voltArray = find_volt_in_array(j['payload'], self.voltArray)
            elif j['headerIndex'] == 0:  # MCS/TRS Data
                self.scalerArray = Form.trs_sum(j, self.curVoltIndex, self.scalerArray,
                                                pipeData['activeTrackPar']['activePmtList'])
        if TrsAna.checkIfScanComplete(pipeData):
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
        rootEle = TildaTools.load_xml(file)
        xmlAddCompleteTrack(rootEle, pipeData, data)
        TildaTools.save_xml(rootEle, file, False)
        return data


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
        self.curVoltIndex = None
        self.totalnOfScalerEvents = None

    def start(self):
        scand = self.Pipeline.pipeData
        tracks, tracks_num_list = SdOp.get_number_of_tracks_in_scan_dict(scand)

        if self.voltArray is None:
            self.voltArray = Form.create_default_volt_array_from_scandict(scand)
        if self.scalerArray is None:
            self.scalerArray = Form.create_default_scaler_array_from_scandict(scand)
        if self.curVoltIndex is None:
            self.curVoltIndex = 0
        if self.totalnOfScalerEvents is None:
            self.totalnOfScalerEvents = np.full((tracks,), 0)

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        ret = None
        scan_complete = False
        for i, j in enumerate(data):
            if j['firstHeader'] == ProgConfigsDict.programs['errorHandler']:  # error send from fpga
                logging.error('fpga sends error code: ' + str(j['payload']) + 'or in binary: ' + str(
                    '{0:032b}'.format(j['payload'])))

            elif j['firstHeader'] == ProgConfigsDict.programs['dac']:  # its a voltage step
                self.curVoltIndex, self.voltArray = find_volt_in_array(j['payload'], self.voltArray, track_ind)

            elif j['firstHeader'] == ProgConfigsDict.programs['continuousSequencer']:
                '''scaler entry '''
                self.totalnOfScalerEvents[track_ind] += 1
                pipeData[track_name]['nOfCompletedSteps'] = self.totalnOfScalerEvents[
                                                                track_ind] // 8  # floored Quotient
                # logging.debug('total completed steps: ' + str(pipeData[track_name]['nOfCompletedSteps']))
                try:  # only add to scalerArray, when pmt is in activePmtList.
                    pmt_index = pipeData[track_name]['activePmtList'].index(j['secondHeader'])
                    self.scalerArray[track_ind][pmt_index][self.curVoltIndex] += j['payload']
                except ValueError:
                    pass
                if CsAna.checkIfScanComplete(pipeData, self.totalnOfScalerEvents[track_ind], track_name):
                    # one Scan over all steps is completed, add Data to return array and clear local buffer.
                    scan_complete = True
                    if ret is None:
                        ret = []
                    ret.append((self.scalerArray, scan_complete))
                    logging.debug('Voltindex: ' + str(self.curVoltIndex) +
                                  'completede steps:  ' + str(pipeData[track_name]['nOfCompletedSteps']))
                    self.scalerArray = Form.create_default_scaler_array_from_scandict(pipeData)  # deletes all entries
                    scan_complete = False
        try:
            if ret is None:
                ret = []
            if np.count_nonzero(self.scalerArray[track_ind]):
                ret.append((self.scalerArray, scan_complete))
            self.scalerArray = Form.create_default_scaler_array_from_scandict(pipeData)  # deletes all entries
            return ret
        except Exception as e:
            print('exception: \t ', e)

    def clear(self):
        self.voltArray = None
        self.scalerArray = None
        self.curVoltIndex = None
        self.totalnOfScalerEvents = None


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
        if self.scalerArray is None:
            self.scalerArray = Form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        try:
            for i, j in enumerate(data):
                self.scalerArray[track_ind] = np.add(self.scalerArray[track_ind], j[track_ind])
                # logging.debug('sum is: ' + str(self.scalerArray[0:2]) + str(self.scalerArray[-2:]))
            return self.scalerArray
        except Exception as e:
            print('exception: ', e)

    def clear(self):
        self.scalerArray = None


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
        if CsAna.checkIfTrackComplete(pipeData, track_name):
            ret = data
        return ret


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
        self.lines = None
        self.x = None
        self.y = None

    def start(self):
        MPLPlotter.ion()
        MPLPlotter.show()

    def animate(self, plotlist):
        MPLPlotter.plt_axes(self.ax, plotlist)
        # MPLPlotter.pause(0.0001)

    def processData(self, data, pipeData):
        t = time.time()
        plot_list = []  # [(x1, y1), (x2, y2), ....]
        for i, dat in enumerate(data):
            # if self.lines[i] is None:
            #     Line2D(linestyle=style)
            #     self.lines.append(self.ax.add_line())
            plot_list.append(dat[0])  # x-data
            plot_list.append(dat[1])  # y-data
            plot_list.append(self.plotStyles[i])
        self.animate(plot_list)
        logging.debug('plot calculating time (ms):' + str(round((time.time() - t) * 1000, 0)))
        return data


class NMPlDrawPlot(Node):
    def __init__(self):
        """
        Node for updating the live plot, each time when data is passed through it
        and when stop() is called
        input: anything
        output: input
        """
        super(NMPlDrawPlot, self).__init__()
        self.type = 'MPlDrawPlot'

    def processData(self, data, pipeData):
        t = time.time()
        MPLPlotter.draw()
        logging.debug('plotting time (ms):' + str(round((time.time() - t) * 1000, 0)))
        return data

    # def stop(self):
    #     t = time.time()
    #     MPLPlotter.draw()
    #     logging.debug('plotting time (ms):' + str(round((time.time() - t) * 1000, 0)))


class NPlotUpdater(Node, animation.TimedAnimation):
    def __init__(self, fig, ax, title, plt_styles_list):
        Node.__init__(self)
        self.type = 'PlotUpdater'
        self.fig = fig
        self.ax = ax
        self.title = title
        self.plotStyles = plt_styles_list
        self.ax.set_ylabel(self.title)
        self.x = np.arange(0, 15)
        self.y = np.arange(0, 15)
        self.line = None
        self.ani = None
        self.line = Line2D(self.x, self.y, color=self.plotStyles[0])
        self.ax.add_line(self.line)
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def start(self):
        print(self.type, ' is starting')
        self.y = Form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)[0][0]
        self.x = Form.create_x_axis_from_scand_dict(self.Pipeline.pipeData)[0]
        plt.show(block=False)

    def processData(self, data, pipeData):
        # logging.debug('plotting...')
        # t = time.time()
        print(self.type, ' is processing Data', data)
        self.x = data[0][0]
        self.y = data[0][1]
        return data

    def _draw_frame(self, framedata):
        print('_draw_frame', framedata)
        self.line.set_data(self.x, self.y)

    def new_frame_seq(self):
        print('new_frame_seq')
        return iter(range(len(self.x)))

    def _init_draw(self):
        print('initial_draw')
        self.line.set_data([], [])


class NSaveSumCS(Node):
    def __init__(self):
        """
        function to save all incoming CS-Sum-Data.
        input: complete, scalerArray containing all tracks.
        output: same as input
        """
        super(NSaveSumCS, self).__init__()
        self.type = 'SaveSumCS'

    def processData(self, data, pipeData):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        pipeInternals = pipeData['pipeInternals']
        file = pipeInternals['activeXmlFilePath']
        rootEle = TildaTools.load_xml(file)
        logging.info('saving data: ' + str(data))
        xmlAddCompleteTrack(rootEle, pipeData, data[track_ind], track_name)
        TildaTools.save_xml(rootEle, file, False)
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
        if self.scalerArray is None:
            self.scalerArray = Form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)

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
                self.scalerArray = Form.create_default_scaler_array_from_scandict(pipeData)
        return ret

    def clear(self):
        self.scalerArray = None


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
        self.spec_data.x = Form.create_x_axis_from_scand_dict(self.Pipeline.pipeData)
        self.spec_data.cts = Form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)
        self.spec_data.err = Form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)

    def processData(self, data, pipeData):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        self.spec_data.nrScalers = len(self.Pipeline.pipeData[track_name]['activePmtList'])
        self.spec_data.cts = data
        return self.spec_data

    def clear(self):
        self.spec_data = None


class NSortByPmt(Node):
    """
    Node for the Simple Counter which will store a limited amount of datapoints per pmt.
    if more datapoints are fed to the pipeline,
    than defined in init, first incoming will be ignored.
    input: splitted raw data
    output: [pmt0, pmt1, ... pmt(7)] with len(pmt0-7) = datapoints
    """

    def __init__(self, datapoints):
        super(NSortByPmt, self).__init__()
        self.type = 'NSortByPmt'
        self.datapoints = datapoints
        self.buffer = None
        self.act_pmt_list = None

    def start(self):
        if self.act_pmt_list is None:
            self.act_pmt_list = self.Pipeline.pipeData.get('activePmtList')
        if self.buffer is None:
            self.buffer = np.zeros((len(self.act_pmt_list), self.datapoints,))

    def clear(self):
        self.act_pmt_list = None
        self.buffer = None

    def processData(self, data, pipeData):
        for ind, val in enumerate(data):
            if val['secondHeader'] in self.act_pmt_list:
                pmt_ind = self.act_pmt_list.index(val['secondHeader'])
                self.buffer[pmt_ind] = np.roll(self.buffer[pmt_ind], 1)
                self.buffer[pmt_ind][0] = val['payload']
        return self.buffer


class NMovingAverage(Node):
    """
    Node for the Simple Counter which does the moving average of all given data points
    input: [pmt0, pmt1, ... pmt(7)] with len(pmt0-7) = datapoints
    output: [avg_pmt0, avg_pmt1, ... , avg_pmt7], avg_pmt(0-7) = float
    """

    def __init__(self):
        super(NMovingAverage, self).__init__()
        self.type = 'MovingAverage'

    def processData(self, data, pipeData):
        avg = [np.sum(pmt) for pmt in data]
        return avg


class NSendnOfCompletedStepsViaQtSignal(Node):
    """
    Node for sending the incoming data via a Qtsignal coming from above
    input: anything that suits qt_signal
    output: same as input
    """

    def __init__(self, qt_signal):
        super(NSendnOfCompletedStepsViaQtSignal, self).__init__()
        self.type = 'SendnOfCompletedStepsViaQtSignal'

        self.qt_signal = qt_signal

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        self.qt_signal.emit(pipeData[track_name]['nOfCompletedSteps'])
        return data


class NSendDataViaQtSignal(Node):
    """
    Node for sending the incoming data via a Qtsignal coming from above
    input: anything that suits qt_signal
    output: same as input
    """

    def __init__(self, qt_signal):
        super(NSendDataViaQtSignal, self).__init__()
        self.type = 'SendViaQtSignal'

        self.qt_signal = qt_signal

    def processData(self, data, pipeData):
        self.qt_signal.emit(data)
        return data


class NAddxAxis(Node):
    """
    Node for the Simple Counter which add an x-axis to the moving average,
    to make it plotable with NMPlLivePlot
    jnput: [avg_pmt0, avg_pmt1, ... , avg_pmt7], avg_pmt(0-7) = float
    output: [(x0, y0), (x0, y0), ...] len(x0) = len(y0) = plotPoints, y0[i] = avg_pmt0
    """

    def __init__(self):
        super(NAddxAxis, self).__init__()
        self.type = 'AddxAxis'
        self.buffer = None

    def start(self):
        plotpoints = self.Pipeline.pipeData.get('plotPoints')
        n_of_pmt = len(self.Pipeline.pipeData.get('activePmtList'))
        self.buffer = np.zeros((n_of_pmt, 2, plotpoints,))
        self.buffer[:, 0] = np.arange(0, plotpoints)

    def processData(self, data, pipeData):
        for pmt_ind, avg_pmt in enumerate(data):
            self.buffer[pmt_ind][1] = np.roll(self.buffer[pmt_ind][1], -1)
            self.buffer[pmt_ind][1][-1] = avg_pmt
        return self.buffer


class NOnlyOnePmt(Node):
    """
    Node to reduce the incoming data to just one pmt.

    """

    def __init__(self, pmt_num):
        super(NOnlyOnePmt, self).__init__()
        self.type = 'OnlyOnePmt'
        self.sel_pmt = pmt_num
        self.pmt_ind = None

    def start(self):
        self.pmt_ind = self.Pipeline.pipeData.get('activePmtList').index(self.sel_pmt)

    def processData(self, data, pipeData):
        return [data[self.pmt_ind]]


class NAddWorkingTime(Node):
    """
    Node to add the Workingtime each time data is processed.
    It also adds the workingtime when start() is called.
    :param reset: bool, set True if you want to reset the workingtime when start() is called.
    """

    def __init__(self, reset=True):
        super(NAddWorkingTime, self).__init__()
        self.type = 'AddWorkingTime'
        self.reset = reset

    def start(self):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        self.Pipeline.pipeData[track_name] = Form.add_working_time_to_track_dict(
            self.Pipeline.pipeData[track_name], self.reset)

    def processData(self, data, pipeData):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        self.Pipeline.pipeData[track_name] = Form.add_working_time_to_track_dict(
            self.Pipeline.pipeData[track_name])
        return data
