"""

Created on '20.05.2015'

@author:'simkaufm'

"""
import logging
import os
import sqlite3
import time
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np

import MPLPlotter
import Service.AnalysisAndDataHandling.csDataAnalysis as CsAna
import Service.FileOperations.FolderAndFileHandling as Filehandle
import Service.Formating as Form
import Service.Scan.ScanDictionaryOperations as SdOp
import TildaTools
from Measurement.SpecData import SpecData
from Measurement.XMLImporter import XMLImporter
from SPFitter import SPFitter
from Service.AnalysisAndDataHandling.InfoHandler import InfoHandler as InfHandl
from Service.FileOperations.XmlOperations import xmlAddCompleteTrack
from Service.ProgramConfigs import Programs as Progs
from Spectra.Straight import Straight
from polliPipe.node import Node

""" multipurpose Nodes: """


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


class NSumListsInData(Node):
    """
    Node for summing up the
    input: [pmt0, pmt1, ... pmt(7)] with len(pmt0-7) = datapoints
    output: [avg_pmt0, avg_pmt1, ... , avg_pmt7], avg_pmt(0-7) = float
    """

    def __init__(self):
        super(NSumListsInData, self).__init__()
        self.type = 'SumListsInData'

    def processData(self, data, pipeData):
        avg = [np.sum(pmt) for pmt in data]
        return avg


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
        Therefore the track is checked for 'nOfCompletedSteps'
        input: anything
        output same as input, if track complete
        """
        super(NCheckIfTrackComplete, self).__init__()
        self.type = 'CheckIfTrackComplete'

    def processData(self, data, pipeData):
        ret = None
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        if CsAna.checkIfTrackComplete(pipeData, track_name):
            ret = data
        return ret


class NCheckIfMeasurementComplete(Node):
    def __init__(self):
        """
        this will only pass the incoming data to the next node,
        if the measurement is complete.
        Therefore all tracks are checked for 'nOfCompletedSteps'
        input: anything
        output same as input, if measurement complete
        """
        super(NCheckIfMeasurementComplete, self).__init__()
        self.type = 'NCheckIfMeasurementComplete'

    def processData(self, data, pipeData):
        ret = None
        tracks, track_list = SdOp.get_number_of_tracks_in_scan_dict(pipeData)
        for track_ind, tr_num in enumerate(track_list):
            track_name = 'track%s' % tr_num
            if CsAna.checkIfTrackComplete(pipeData, track_name):
                ret = data
            else:
                return None
        return ret


class NOnlyOnePmt(Node):
    """
    Node to reduce the incoming data to just one pmt.
    input: [ list_pmt0, list_pmt1, ... ]
    output: [ list_pmt_selected ]
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
    input: anything
    output: same as input
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


class NAddWorkingTimeOnClear(Node):
    """
        Node to add the Workingtime on start() and on clear(). mostly for Tilda Passive
        :param reset: bool, set True if you want to reset the workingtime when start() is called.
        input: anything
        output: same as input
        """

    def __init__(self, reset=True):
        super(NAddWorkingTimeOnClear, self).__init__()
        self.type = 'AddWorkingTimeOnClear'
        self.reset = reset

    def start(self):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        self.Pipeline.pipeData[track_name] = Form.add_working_time_to_track_dict(
            self.Pipeline.pipeData[track_name], self.reset)

    def processData(self, data, pipeData):
        return data

    def clear(self):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        self.Pipeline.pipeData[track_name] = Form.add_working_time_to_track_dict(
            self.Pipeline.pipeData[track_name])


""" saving """


class NSaveIncomDataForActiveTrack(Node):
    def __init__(self):
        """
        Node to save the data of the active track, when data is passed to it.
        input: complete, scalerArray containing all tracks.
        output: same as input
        """
        super(NSaveIncomDataForActiveTrack, self).__init__()
        self.type = 'SaveIncomDataForActiveTrack'

    def processData(self, data, pipeData):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        pipeInternals = pipeData['pipeInternals']
        file = pipeInternals['activeXmlFilePath']
        rootEle = TildaTools.load_xml(file)
        logging.info('saving data: ' + str(data))
        xmlAddCompleteTrack(rootEle, pipeData, data[track_ind], track_name)
        TildaTools.save_xml(rootEle, file, False)
        logging.info('saving sum to: ' + str(file))
        return data


class NSaveAllTracks(Node):
    def __init__(self):
        """
        Node to save everything in data for all tracks.
        will save on clear() call!
        input: complete, scalerArray containing all tracks.
        output: same as input
        """
        super(NSaveAllTracks, self).__init__()
        self.type = 'SaveAllTracks'
        self.storage = None

    def processData(self, data, pipeData):
        self.storage = data
        return data

    def clear(self):
        if self.storage is not None:
            pipeData = self.Pipeline.pipeData
            pipeInternals = pipeData['pipeInternals']
            file = pipeInternals['activeXmlFilePath']
            rootEle = TildaTools.load_xml(file)
            tracks, track_list = SdOp.get_number_of_tracks_in_scan_dict(pipeData)
            for track_ind, tr_num in enumerate(track_list):
                track_name = 'track%s' % tr_num
                xmlAddCompleteTrack(rootEle, pipeData, self.storage[track_ind], track_name)
            TildaTools.save_xml(rootEle, file, False)
            self.storage = None


class NSaveSpecData(Node):
    def __init__(self):
        """
        Node to save specdata which is stored in a buffer until clear is called.
        will save on clear() call!
        input: complete, scalerArray containing all tracks.
        output: same as input
        """
        super(NSaveSpecData, self).__init__()
        self.type = 'SaveAllTracks'
        self.storage = None

    def processData(self, data, pipeData):
        self.storage = data
        return data

    def clear(self):
        if self.storage is not None:
            Filehandle.save_spec_data(self.storage, self.Pipeline.pipeData)
            self.storage = None


class NSaveProjection(Node):
    """
    Node for saving the incoming projectin data.
    saves on clear call.
    input: [[[v_proj_tr0_pmt0, v_proj_tr0_pmt1, ... ], [t_proj_tr0_pmt0, t_proj_tr0_pmt1, ... ]], ...]
    output: [[[v_proj_tr0_pmt0, v_proj_tr0_pmt1, ... ], [t_proj_tr0_pmt0, t_proj_tr0_pmt1, ... ]], ...]
    """

    def __init__(self):
        super(NSaveProjection, self).__init__()
        self.type = 'SaveProjection'
        self.storage = None

    def processData(self, data, pipeData):
        self.storage = data
        return data

    def clear(self):
        if self.storage is not None:
            data = self.storage
            pipeData = self.Pipeline.pipeData
            pipeInternals = pipeData['pipeInternals']
            file = pipeInternals['activeXmlFilePath']
            rootEle = TildaTools.load_xml(file)
            tracks, track_list = SdOp.get_number_of_tracks_in_scan_dict(pipeData)
            for track_ind, tr_num in enumerate(track_list):
                track_name = 'track%s' % tr_num
                xmlAddCompleteTrack(
                    rootEle, pipeData, data[track_ind][0], track_name, datatype='voltage_projection',
                    parent_ele_str='projections')
                xmlAddCompleteTrack(
                    rootEle, pipeData, data[track_ind][1], track_name, datatype='time_projection',
                    parent_ele_str='projections')
            TildaTools.save_xml(rootEle, file, False)


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
        self.maxArraySize = 500000

        self.buf = None
        self.nOfSaves = None
        self.time_of_last_save = datetime.now()
        self.max_time_between_saves = timedelta(minutes=5)  # save every five minutes even if countrate is low

    def start(self):
        if self.buf is None:
            self.buf = np.zeros(0, dtype=np.uint32)
        if self.nOfSaves is None:
            self.nOfSaves = -1

    def processData(self, data, pipeData):
        if self.nOfSaves < 0:  # save pipedata, first time something is fed to the pipelins
            self.nOfSaves = Filehandle.savePipeData(pipeData, self.nOfSaves)
            self.time_of_last_save = datetime.now()
        self.buf = np.append(self.buf, data)
        time_since_last_save = datetime.now() - self.time_of_last_save
        if self.buf.size > self.maxArraySize or time_since_last_save >= self.max_time_between_saves:
            # when buffer is full or the maximum acquisition time is reached, store the data to disc
            print('saving raw data, time since last save is: ', time_since_last_save.seconds)
            self.nOfSaves = Filehandle.saveRawData(self.buf, pipeData, self.nOfSaves)
            self.time_of_last_save = datetime.now()
            self.buf = np.zeros(0, dtype=np.uint32)
        return data

    def clear(self):
        Filehandle.saveRawData(self.buf, self.Pipeline.pipeData, 0)
        Filehandle.savePipeData(self.Pipeline.pipeData, 0)  # also save the pipeData when clearing
        self.time_of_last_save = datetime.now()
        self.nOfSaves = None
        self.buf = None


""" plotting """


class NMPlLivePlot(Node):
    def __init__(self, ax, title, line_color_list):
        """
        Node for plotting live Data using matplotlib.pyplot
        input: list, [(x1, y1), (x2, y2),... ] x and y are numpy arrays
        output: input
        """
        super(NMPlLivePlot, self).__init__()
        self.type = 'MPlLivePlot'

        self.ax = ax
        self.title = title
        self.line_colors = line_color_list
        self.ax.set_ylabel(self.title)
        self.lines = [None for i in self.line_colors]
        self.x = None
        self.y = None

    def start(self):
        MPLPlotter.ion()
        MPLPlotter.show()

    def clear(self):
        self.lines = [None for i in self.line_colors]
        self.x = None
        self.y = None

    def processData(self, data, pipeData):
        for i, dat in enumerate(data):
            if self.lines[i] is None:
                self.lines[i] = self.ax.add_line(MPLPlotter.line2d(dat[0], dat[1], self.line_colors[i]))
                self.ax.set_xlim(min(dat[0]), max(dat[0]))
                self.ax.autoscale(enable=True, axis='y', tight=False)
            self.lines[i].set_ydata(dat[1])  # only necessary to reset y-data
            self.ax.relim()
            self.ax.autoscale_view(tight=False)
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
        # t = time.time()
        MPLPlotter.draw()
        # logging.debug('plotting time (ms):' + str(round((time.time() - t) * 1000, 0)))
        return data


class NMPLImagePLot(Node):
    def __init__(self, pmt_num, as_voltage=True):
        """
        plotting node, for plotting the image data of one track and one pmt
        also the projections inside teh gates are displayed.
        """
        super(NMPLImagePLot, self).__init__()
        self.type = 'MPLImagePLot'
        self.fig, self.axes = MPLPlotter.setup_image_figure()
        self.im_ax = self.axes[0][0]
        self.cb_ax = self.axes[0][1]
        self.tproj_ax = self.axes[0][2]
        self.vproj_ax = self.axes[1][0]
        self.pmt_radio_ax = self.axes[1][1]
        self.tr_radio_ax = self.axes[1][2]
        self.save_bt_ax = self.axes[1][3]
        self.slider_ax = self.axes[2][0]
        self.selected_pmt = pmt_num
        self.selected_pmt_ind = None
        self.image = None
        self.colorbar = None
        self.tproj_line = None
        self.vproj_line = None
        self.patch = None
        self.gates_list = None  # [[[vals_pmt0],[ind_pmt0]], ...]
        self.buffer_data = None
        self.full_data = None
        self.aspect_img = 'auto'
        self.gate_anno = None
        self.volt_array = None
        self.as_voltage = as_voltage
        self.time_array = None
        self.radio_buttons_pmt = None
        self.radio_con = None
        self.radio_buttons_tr = None
        self.selected_track = None
        self.save_button = None
        self.slider = None
        MPLPlotter.ion()
        MPLPlotter.show()

    def rect_select_gates(self, eclick, erelease):
        """
        is called via left/rigth click & release events, connection see in start()
        will pass the coordinates of the selected area to self.update_gate_ind()
        """
        try:
            volt_1, time_1 = eclick.xdata, eclick.ydata
            volt_2, volt_3 = erelease.xdata, erelease.ydata
            volt_1, volt_2 = sorted((volt_1, volt_2))
            time_1, volt_3 = sorted((time_1, volt_3))
            gates_list = [volt_1, volt_2, time_1, volt_3]
            self.update_gate_ind(gates_list)
            self.gate_data_and_plot(True)
        except Exception as e:
            print('while setting the gates this happened: ', e)

    def gate_data_and_plot(self, draw=False):
        """
        uses the currently stored gates (self.gates_list) to gate the stored data in
        self.buffer_data and plots the result.
        """
        try:
            data = self.buffer_data
            g_list = self.gates_list[self.selected_pmt_ind][0]
            g_ind = self.gates_list[self.selected_pmt_ind][1]
            self.patch.set_xy((g_list[0], g_list[2]))
            self.patch.set_width((g_list[1] - g_list[0]))
            self.patch.set_height((g_list[3] - g_list[2]))
            sum_l = len(np.sum(data[g_ind[0]:g_ind[1] + 1, :], axis=0))
            data_l = len(self.tproj_line.get_ydata())
            self.tproj_line.set_xdata(
                np.sum(data[g_ind[0]:g_ind[1] + 1, :], axis=0))
            self.vproj_line.set_ydata(
                np.sum(data[:, g_ind[2]:g_ind[3] + 1], axis=1))
            # +1 due to syntax of slicing!
            self.tproj_ax.relim()
            self.tproj_ax.set_xmargin(0.05)
            self.tproj_ax.autoscale(enable=True, axis='x', tight=False)
            self.vproj_ax.relim()
            self.vproj_ax.set_ymargin(0.05)
            self.vproj_ax.autoscale(enable=True, axis='y', tight=False)
            if draw:
                MPLPlotter.draw()
        except Exception as e:
            print('while plotting projection this happened: ', e)
            print('t_proj lenghts are: ', sum_l, data_l)

    def update_gate_ind(self, gates_val_list):
        """
        gates_val_list must be in form of:
        [v_min, v_max, t_min, t_max]

        overwrites: self.Pipeline.pipeData[track_name]['softwGates']
        and stores gates in self.gates_list
        :return:self.gates_list, [[v_min, v_max, t_min, t_max], [v_min_ind, v_max_ind, t_min_ind, t_max_ind]]
        """
        try:
            v_min, v_max = sorted((gates_val_list[0], gates_val_list[1]))
            v_min_ind, v_min, vdif = TildaTools.find_closest_value_in_arr(self.volt_array, v_min)
            v_max_ind, v_max, vdif = TildaTools.find_closest_value_in_arr(self.volt_array, v_max)

            t_min, t_max = sorted((gates_val_list[2], gates_val_list[3]))
            t_min_ind, t_min, tdif = TildaTools.find_closest_value_in_arr(self.time_array, t_min)
            t_max_ind, t_max, tdif = TildaTools.find_closest_value_in_arr(self.time_array, t_max)
            gates_ind = [v_min_ind, v_max_ind, t_min_ind, t_max_ind]  # indices in data array
            gates_val_list = [v_min, v_max, t_min, t_max]
            self.gates_list[self.selected_pmt_ind] = [gates_val_list, gates_ind]
            self.Pipeline.pipeData[self.selected_track[1]]['softwGates'][self.selected_pmt_ind] = gates_val_list
            if self.gate_anno is None:
                self.gate_anno = self.im_ax.annotate('%s - %s V \n%s - %s ns'
                                                     % (self.volt_array[v_min_ind], self.volt_array[v_max_ind],
                                                        self.time_array[t_min_ind], self.time_array[t_max_ind]),
                                                     xy=(self.im_ax.get_xlim()[0], self.im_ax.get_ylim()[1] / 2),
                                                     xycoords='data', annotation_clip=False, color='white')
            self.gate_anno.set_text('%s - %s V \n%s - %s ns'
                                    % (self.volt_array[v_min_ind], self.volt_array[v_max_ind],
                                       self.time_array[t_min_ind], self.time_array[t_max_ind]))
            self.gate_anno.set_x(self.im_ax.xaxis.get_view_interval()[0])
            ymin, ymax = self.im_ax.yaxis.get_view_interval()
            self.gate_anno.set_y(ymax - (ymax - ymin) / 6)
            return self.gates_list
        except Exception as e:
            print('while updating the indice this happened: ', e)

    def setup_track(self, track_ind, track_name):
        try:
            for ax in [val for sublist in self.axes for val in sublist][:-4]:
                if ax:  # be sure ax is not 0, don't clear radio buttons, buttons and slider
                    MPLPlotter.clear_ax(ax)
            self.gate_anno = None
            self.gates_list = [[None]] * len(self.Pipeline.pipeData[track_name]['activePmtList'])
            self.volt_array = Form.create_x_axis_from_scand_dict(self.Pipeline.pipeData, as_voltage=self.as_voltage)[
                track_ind]
            v_shape = self.volt_array.shape
            self.time_array = Form.create_time_axis_from_scan_dict(self.Pipeline.pipeData, rebinning=True)[track_ind]
            t_shape = self.time_array.shape

            self.image, self.colorbar = MPLPlotter.configure_image_plot(
                self.fig, self.im_ax, self.cb_ax, self.Pipeline.pipeData, self.volt_array,
                self.time_array, self.selected_pmt, track_name)

            self.vproj_line, self.tproj_line = MPLPlotter.setup_projection(
                self.axes, self.volt_array, self.time_array)

            patch_ext = [self.volt_array[0], self.time_array[0],
                         abs(self.volt_array[v_shape[0] / 2]), abs(self.time_array[t_shape[0] / 2])]
            self.patch = MPLPlotter.add_patch(self.im_ax, patch_ext)

            MPLPlotter.add_rect_select(self.im_ax, self.rect_select_gates,
                                       self.volt_array[1] - self.volt_array[0],
                                       self.time_array[1] - self.time_array[0])
            if self.Pipeline.pipeData[track_name].get('softwGates', None) is None:
                gate_val_list = [np.amin(self.volt_array), np.amax(self.volt_array),
                                 np.amin(self.time_array), np.amax(self.time_array)]  # initial values, full frame
                self.Pipeline.pipeData[track_name]['softwGates'] = \
                    [[None]] * len(self.Pipeline.pipeData[track_name]['activePmtList'])
            else:  # read gates from input
                gate_val_list = self.Pipeline.pipeData[track_name].get('softwGates', None)[self.selected_pmt_ind]
            self.update_gate_ind(gate_val_list)
            bin_width = self.Pipeline.pipeData[self.selected_track[1]].get('softBinWidth_ns', 10)
            self.slider.valtext.set_text('{}'.format(bin_width))

            MPLPlotter.draw()
        except Exception as e:
            print('while starting this occured: ', e)

    def pmt_radio_buttons(self, label):
        try:
            self.selected_pmt = int(label[3:])
            self.selected_pmt_ind = self.Pipeline.pipeData[self.selected_track[1]]['activePmtList'].index(
                self.selected_pmt)
            print('selected pmt index is: ', int(label[3:]))
            self.buffer_data = Form.time_rebin_all_data(
                self.full_data, self.Pipeline.pipeData)[self.selected_track[0]][self.selected_pmt_ind]
            self.setup_track(*self.selected_track)
            self.image.set_data(np.transpose(self.buffer_data))
            self.colorbar.set_clim(0, np.amax(self.buffer_data))
            self.colorbar.update_normal(self.image)
            self.gate_data_and_plot()
            self.im_ax.set_aspect(self.aspect_img, adjustable='box-forced')
            MPLPlotter.draw()
        except Exception as e:
            print(e)

    def tr_radio_buttons(self, label):
        try:
            tr, tr_list = SdOp.get_number_of_tracks_in_scan_dict(self.Pipeline.pipeData)
            self.selected_track = (tr_list.index(int(label[5:])), label)
            print('selected track index is: ', int(label[5:]))
            self.buffer_data = Form.time_rebin_all_data(
                self.full_data, self.Pipeline.pipeData)[self.selected_track[0]][self.selected_pmt_ind]
            self.setup_track(*self.selected_track)
            self.image.set_data(np.transpose(self.buffer_data))
            self.colorbar.set_clim(0, np.amax(self.buffer_data))
            self.colorbar.update_normal(self.image)
            self.gate_data_and_plot()
            self.im_ax.set_aspect(self.aspect_img, adjustable='box-forced')
            MPLPlotter.draw()
        except Exception as e:
            print(e)

    def save_proj(self, bool):
        """ saves projection of all tracks """
        try:
            pipeData = self.Pipeline.pipeData
            time_arr = Form.create_time_axis_from_scan_dict(self.Pipeline.pipeData, rebinning=True)
            v_arr = Form.create_x_axis_from_scand_dict(self.Pipeline.pipeData, as_voltage=True)
            rebinned_data = Form.time_rebin_all_data(self.full_data, self.Pipeline.pipeData)
            data = Form.gate_all_data(pipeData, rebinned_data, time_arr, v_arr)
            pipeInternals = pipeData['pipeInternals']
            file = pipeInternals['activeXmlFilePath']
            rootEle = TildaTools.load_xml(file)
            tracks, track_list = SdOp.get_number_of_tracks_in_scan_dict(pipeData)
            for track_ind, tr_num in enumerate(track_list):
                track_name = 'track%s' % tr_num
                xmlAddCompleteTrack(
                    rootEle, pipeData, data[track_ind][0], track_name, datatype='voltage_projection',
                    parent_ele_str='projections')
                xmlAddCompleteTrack(
                    rootEle, pipeData, data[track_ind][1], track_name, datatype='time_projection',
                    parent_ele_str='projections')
            TildaTools.save_xml(rootEle, file, False)
        except Exception as e:
            print(e)

    def rebin_changed(self, bins_10ns):
        try:
            bins_10ns_rounded = bins_10ns // 10 * 10
            self.Pipeline.pipeData[self.selected_track[1]]['softBinWidth_ns'] = bins_10ns_rounded
            self.slider.valtext.set_text('{}'.format(bins_10ns_rounded))
            self.buffer_data = Form.time_rebin_all_data(
                self.full_data, self.Pipeline.pipeData)[self.selected_track[0]][self.selected_pmt_ind]
            self.setup_track(*self.selected_track)
            self.image.set_data(np.transpose(self.buffer_data))
            self.colorbar.set_clim(0, np.amax(self.buffer_data))
            self.colorbar.update_normal(self.image)
            self.gate_data_and_plot()
            self.im_ax.set_aspect(self.aspect_img, adjustable='box-forced')
            MPLPlotter.draw()
        except Exception as e:
            print(e)

    def start(self):
        try:
            track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
            self.selected_track = (track_ind, track_name)
            bin_width = self.Pipeline.pipeData[self.selected_track[1]]['softBinWidth_ns']
            self.selected_pmt_ind = self.Pipeline.pipeData[self.selected_track[1]]['activePmtList'].index(
                self.selected_pmt)
            self.setup_track(*self.selected_track)
            if self.radio_buttons_pmt is None:
                labels = ['pmt%s' % pmt for pmt in self.Pipeline.pipeData[self.selected_track[1]]['activePmtList']]
                self.radio_buttons_pmt, self.radio_con = MPLPlotter.add_radio_buttons(
                    self.pmt_radio_ax, labels, self.selected_pmt_ind, self.pmt_radio_buttons)
            # self.radio_buttons_pmt.set_active(self.selected_pmt_ind)  # not available before mpl 1.5.0
            if self.radio_buttons_tr is None:
                tr, tr_list = SdOp.get_number_of_tracks_in_scan_dict(self.Pipeline.pipeData)
                label_tr = ['track%s' % tr_num for tr_num in tr_list]
                self.radio_buttons_tr, con = MPLPlotter.add_radio_buttons(
                    self.tr_radio_ax, label_tr, self.selected_track[0], self.tr_radio_buttons
                )
            # self.radio_buttons_tr.set_active(self.selected_track[0])  # not available before mpl 1.5.0
            if self.save_button is None:
                self.save_button, button_con = MPLPlotter.add_button(self.save_bt_ax, 'save_proj', self.save_proj)
            if self.slider is None:
                self.slider, slider_con = MPLPlotter.add_slider(self.slider_ax, 'rebinning', 10, 100,
                                                                self.rebin_changed, valfmt=u'%3d', valinit=10)
            self.slider.valtext.set_text('{}'.format(bin_width))
        except Exception as e:
            print(e)

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        try:
            self.full_data = data
            self.buffer_data = Form.time_rebin_all_data(data, self.Pipeline.pipeData)[track_ind][self.selected_pmt_ind]
            self.image.set_data(np.transpose(self.buffer_data))
            self.colorbar.set_clim(0, np.amax(self.buffer_data))
            self.colorbar.update_normal(self.image)
            self.gate_data_and_plot()
            self.im_ax.set_aspect(self.aspect_img, adjustable='box-forced')
            pass
        except Exception as e:
            print('while updateing plot, this happened: ', e)
        return data

    def clear(self):
        # i dont want to clear this window after completion of scan
        pass


class NMPLCloseFigOnClear(Node):
    def __init__(self, fig_ref=None):
        super(NMPLCloseFigOnClear, self).__init__()
        self.fig_ref = fig_ref

    def processData(self, data, pipeData):
        return data

    def clear(self):
        if self.fig_ref is None:
            MPLPlotter.close_all_figs()
        else:
            MPLPlotter.close_fig(self.fig_ref)
        self.fig_ref = None


class NMPLCloseFigOnInit(Node):
    def __init__(self):
        super(NMPLCloseFigOnInit, self).__init__()
        self.type = 'MPLCloseFigOnInit'
        MPLPlotter.close_all_figs()

    def processData(self, data, pipeData):
        return data


""" specdata format compatible Nodes: """


class NSortedTrsArraysToSpecData(Node):
    def __init__(self, x_as_voltage=True):
        """
        when started, will init a SpecData object of given size.
        Overwrites SpecData.cts and passes the SpecData object to the next node.
        input: list of tuples of, [(scalerArray->containing all tracks, scan_complete_bool),...]
        output: SpecData
        """
        super(NSortedTrsArraysToSpecData, self).__init__()
        self.type = 'SortedTrsArraysToSpecData'
        self.spec_data = None
        self.x_as_voltage = x_as_voltage

    def start(self):
        if self.spec_data is None:
            self.spec_data = XMLImporter(None, self.x_as_voltage, self.Pipeline.pipeData)
            logging.debug('pipeline successfully loaded: %s' % self.spec_data.file)

    def processData(self, data, pipeData):
        for arr_scan_compl_tpl in data:
            for tr_ind, tr_data in enumerate(arr_scan_compl_tpl[0]):
                self.spec_data.time_res[tr_ind] += tr_data
        return self.spec_data

    def clear(self):
        self.spec_data = None


class NStartNodeKepcoScan(Node):
    def __init__(self, x_as_voltage_bool, dmm_names_sorted):
        """
        Node for handling the raw datastream which is created during a KepcoScan.
        :param x_as_voltage_bool: bool, True, if you want an x-axis in voltage, this

        input: rawdata from fpga AND dict as readback from dmm (key is name of dmm, val is np.array)
        output: specdata, cts are voltage readings.
        """
        super(NStartNodeKepcoScan, self).__init__()
        self.type = 'StartNodeKepcoScan'
        self.spec_data = None
        self.curVoltIndex = 0
        self.info_handl = InfHandl()
        self.x_as_voltage = x_as_voltage_bool
        self.dmms = dmm_names_sorted  # list with the dmm names, indeces are equal to indices in spec_data.cts, etc.

    def calc_voltage_err(self, voltage_reading, dmm_name):
        read_err, range_err = self.Pipeline.pipeData['measureVoltPars']['duringScan']['dmms'][dmm_name].get('accuracy', (None, None))
        if read_err is not None:
            return voltage_reading * read_err + range_err
        else:
            return 1

    def start(self):
        self.spec_data = XMLImporter(None, self.x_as_voltage, self.Pipeline.pipeData)
        # self.spec_data.cts[0] = np.full(self.spec_data.cts[0].shape, np.nan, dtype=np.double)

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        if isinstance(data, dict):  # readback from multimeter
            for dmm_name, dmm_readback_arr in data.items():
                if dmm_readback_arr is not None:
                    if dmm_name in self.dmms:  # this would only fail if creation of self.dmm was wrong
                        dmm_ind = self.dmms.index(dmm_name)  # raise exception when not found
                        for dmm_read in dmm_readback_arr:  # can contain more than one value
                            if np.isnan(np.min(self.spec_data.cts[track_ind][dmm_ind])):
                                # check if there still is room for another voltage reading
                                volt_ind = np.where(np.isnan(self.spec_data.cts[track_ind][dmm_ind]))[0][0]
                                self.spec_data.cts[track_ind][dmm_ind][volt_ind] = dmm_read
                                self.spec_data.err[track_ind][dmm_ind][volt_ind] = self.calc_voltage_err(
                                    dmm_read, dmm_name)
                                # print('volt_reading: ', self.spec_data.cts)
                            else:
                                # print('received more voltages than it should! check your settings!')
                                pass

        elif isinstance(data, np.ndarray):  # rawdata from fpga
            for raw_data in data:
                # plitup each element in data and analyse what it means:
                first_header, second_header, header_index, payload = Form.split_32b_data(raw_data)

                if first_header == Progs.infoHandler.value:
                    v_ind, step_completed = self.info_handl.info_handle(pipeData, payload)
                    if v_ind is not None:
                        self.curVoltIndex = v_ind
                elif first_header == Progs.errorHandler.value:
                    logging.error('fpga sends error code: ' + str(payload) + 'or in binary: ' + str(
                        '{0:032b}'.format(payload)))
                elif first_header == Progs.dac.value:
                    pass  # step complete etc. will be handled with infohandler values
                elif first_header == Progs.continuousSequencer:
                    pass  # this should not happen here.
                elif header_index == 0:
                    pass  # should also not happen here

        return self.spec_data

    def clear(self):
        self.spec_data = None


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
        self.spec_data.x = Form.create_x_axis_from_scand_dict(self.Pipeline.pipeData, as_voltage=True)
        self.spec_data.cts = Form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)
        self.spec_data.err = Form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)

    def processData(self, data, pipeData):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        self.spec_data.nrScalers = len(self.Pipeline.pipeData[track_name]['activePmtList'])
        self.spec_data.cts = data
        return self.spec_data

    def clear(self):
        self.spec_data = None


class NSingleSpecFromSpecData(Node):
    def __init__(self, scalers, track=-1):
        """
        will return a single spectrum of the given scalers
        a tuple with (volt, cts, err) of the specified scaler and track.
        :param scalers: list, of scalers pos sign for adding, negative sign for subracting
        :param track: int, index of track, -1 for all tracks.

        input: SpecData
        ouptut: tuple, (volt, cts, err)
        """
        super(NSingleSpecFromSpecData, self).__init__()
        self.type = 'SingleSpecFromSpecData'
        self.scalers = scalers
        self.track = track

    def processData(self, spec_data_instance, pipeData):
        ret = []
        x, y, err = spec_data_instance.getArithSpec(self.scalers, self.track)
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


class NMPLImagePlotSpecData(Node):
    def __init__(self, pmt_num):
        """
        plotting node, for plotting the image data of one track and one pmt
        also the projections inside teh gates are displayed.
        """
        super(NMPLImagePlotSpecData, self).__init__()
        self.type = 'MPLImagePlotSpecData'
        self.fig, self.axes = MPLPlotter.setup_image_figure()
        self.im_ax = self.axes[0][0]
        self.cb_ax = self.axes[0][1]
        self.tproj_ax = self.axes[0][2]
        self.vproj_ax = self.axes[1][0]
        self.pmt_radio_ax = self.axes[1][1]
        self.tr_radio_ax = self.axes[1][2]
        self.save_bt_ax = self.axes[1][3]
        self.slider_ax = self.axes[2][0]
        self.selected_pmt = pmt_num
        self.selected_pmt_ind = None
        self.image = None
        self.colorbar = None
        self.tproj_line = None  # mpl objekt
        self.vproj_line = None
        self.patch = None
        self.buffer_data = None
        self.full_data = None
        self.aspect_img = 'auto'
        self.gate_anno = None  # annotation for gate

        self.radio_buttons_pmt = None
        self.radio_con = None
        self.radio_buttons_tr = None
        self.selected_track = (0, 'track0')
        self.save_button = None
        self.slider = None
        MPLPlotter.ion()
        MPLPlotter.show(False)

    def rect_select_gates(self, eclick, erelease):
        """
        is called via left/rigth click & release events, connection see in start()
        will pass the coordinates of the selected area to self.update_gate_ind()
        """
        try:
            volt_1, time_1 = eclick.xdata, eclick.ydata
            volt_2, volt_3 = erelease.xdata, erelease.ydata
            volt_1, volt_2 = sorted((volt_1, volt_2))
            time_1, volt_3 = sorted((time_1, volt_3))
            gates_list = [volt_1, volt_2, time_1, volt_3]
            self.update_gate_ind(gates_list)
            self.gate_data_and_plot(True)
        except Exception as e:
            print('while setting the gates this happened: ', e)

    def gate_data_and_plot(self, draw=False):
        """
        uses the currently stored gates (self.gates_list) to gate the stored data in
        self.buffer_data and plots the result.
        """
        try:
            data = self.buffer_data
            g_ind, g_list = self.update_gate_ind(data.softw_gates[self.selected_pmt_ind])
            self.patch.set_xy((g_list[0], g_list[2]))
            self.patch.set_width((g_list[1] - g_list[0]))
            self.patch.set_height((g_list[3] - g_list[2]))
            xdata = np.sum(
                data.time_res[self.selected_track[0]][self.selected_pmt_ind][g_ind[0]:g_ind[1] + 1, :], axis=0)
            xdata = np.nan_to_num(xdata)
            ydata = np.sum(
                data.time_res[self.selected_track[0]][self.selected_pmt_ind][:, g_ind[2]:g_ind[3] + 1], axis=1)
            ydata = np.nan_to_num(ydata)
            self.tproj_line.set_xdata(xdata)
            self.vproj_line.set_ydata(ydata)
            # +1 due to syntax of slicing!
            self.tproj_ax.relim()
            self.tproj_ax.set_xmargin(0.05)
            self.tproj_ax.autoscale(enable=True, axis='x', tight=False)
            self.vproj_ax.relim()
            self.vproj_ax.set_ymargin(0.05)
            self.vproj_ax.autoscale(enable=True, axis='y', tight=False)
            if draw:
                MPLPlotter.draw()
        except Exception as e:
            print('while plotting projection this happened: ', e)
            # print('t_proj lenghts are: ',
            #       len(np.sum(data.time_res[g_ind[0]:g_ind[1] + 1, :], axis=0)), len(self.tproj_line.get_ydata()))

    def update_gate_ind(self, gates_val_list):
        """
        gates_val_list must be in form of, for one pmt in one track:
        [v_min, v_max, t_min, t_max]

        overwrites: self.Pipeline.pipeData[track_name]['softwGates']
        and stores gates in self.gates_list
        :return: tuple, ([gate_ind], [gate_vals])
        """
        try:
            volt_array = self.buffer_data.x[self.selected_track[0]]
            time_array = self.buffer_data.t[self.selected_track[0]]
            v_min, v_max = sorted((gates_val_list[0], gates_val_list[1]))
            v_min_ind, v_min, vdif = TildaTools.find_closest_value_in_arr(volt_array, v_min)
            v_max_ind, v_max, vdif = TildaTools.find_closest_value_in_arr(volt_array, v_max)

            t_min, t_max = sorted((gates_val_list[2], gates_val_list[3]))
            t_min_ind, t_min, tdif = TildaTools.find_closest_value_in_arr(time_array, t_min)
            t_max_ind, t_max, tdif = TildaTools.find_closest_value_in_arr(time_array, t_max)
            gates_ind = [v_min_ind, v_max_ind, t_min_ind, t_max_ind]  # indices in data array
            gates_val_list = [v_min, v_max, t_min, t_max]
            if self.gate_anno is None:
                self.gate_anno = self.im_ax.annotate('%s - %s V \n%s - %s ns'
                                                     % (volt_array[v_min_ind], volt_array[v_max_ind],
                                                        time_array[t_min_ind], time_array[t_max_ind]),
                                                     xy=(self.im_ax.get_xlim()[0], self.im_ax.get_ylim()[1] / 2),
                                                     xycoords='data', annotation_clip=False, color='white')
            self.gate_anno.set_text('%s - %s V \n%s - %s ns'
                                    % (volt_array[v_min_ind], volt_array[v_max_ind],
                                       time_array[t_min_ind], time_array[t_max_ind]))
            self.gate_anno.set_x(self.im_ax.xaxis.get_view_interval()[0])
            ymin, ymax = self.im_ax.yaxis.get_view_interval()
            self.gate_anno.set_y(ymax - (ymax - ymin) / 6)
            self.buffer_data.softw_gates[self.selected_pmt_ind] = gates_val_list
            return gates_ind, gates_val_list
        except Exception as e:
            print('while updating the indice this happened: ', e)
            return [0, 1, 2, 3], [-1, 1, 0, 1]

    def setup_track(self, track_ind, track_name):
        """ setup the plots for this track """
        try:
            for ax in [val for sublist in self.axes for val in sublist][:-4]:
                if ax:  # be sure ax is not 0, don't clear radio buttons, buttons and slider
                    MPLPlotter.clear_ax(ax)
            self.gate_anno = None
            volt_array = self.buffer_data.x[track_ind]
            v_shape = volt_array.shape
            time_array = self.buffer_data.t[track_ind]
            t_shape = time_array.shape

            self.image, self.colorbar = MPLPlotter.configure_image_plot(
                self.fig, self.im_ax, self.cb_ax, self.Pipeline.pipeData, volt_array,
                time_array, self.selected_pmt, track_name)

            self.vproj_line, self.tproj_line = MPLPlotter.setup_projection(
                self.axes, volt_array, time_array)

            patch_ext = [volt_array[0], time_array[0],
                         abs(volt_array[v_shape[0] / 2]), abs(time_array[t_shape[0] / 2])]
            self.patch = MPLPlotter.add_patch(self.im_ax, patch_ext)

            MPLPlotter.add_rect_select(self.im_ax, self.rect_select_gates,
                                       volt_array[1] - volt_array[0],
                                       time_array[1] - time_array[0])
            if self.buffer_data.softw_gates is None:
                gate_val_list = [np.amin(volt_array), np.amax(volt_array),
                                 np.amin(time_array), np.amax(time_array)]  # initial values, full frame
                self.buffer_data.softw_gates = \
                    [[None]] * self.buffer_data.get_scaler_step_and_bin_num(track_ind)[0]
            else:  # read gates from input
                gate_val_list = self.buffer_data.softw_gates[self.selected_pmt_ind]
            self.update_gate_ind(gate_val_list)
            bin_width = self.buffer_data.softBinWidth_ns[track_ind]
            if self.slider is not None:
                print('slider is set to:', bin_width)
                self.slider.valtext.set_text('{}'.format(bin_width))

            MPLPlotter.draw()
        except Exception as e:
            print('while starting this occured: ', e)

    def pmt_radio_buttons(self, label):
        try:
            self.selected_pmt = int(label[3:])
            self.selected_pmt_ind = self.buffer_data.active_pmt_list[self.selected_track[0]].index(self.selected_pmt)
            print('selected pmt index is: ', int(label[3:]))
            self.buffer_data = Form.time_rebin_all_spec_data(
                self.full_data, self.buffer_data.softBinWidth_ns[self.selected_track[0]])
            self.setup_track(*self.selected_track)
            self.image.set_data(np.transpose(self.buffer_data.time_res[self.selected_track[0]][self.selected_pmt_ind]))
            self.colorbar.set_clim(0,
                                   np.max(self.buffer_data.time_res[self.selected_track[0]][self.selected_pmt_ind]))
            self.colorbar.update_normal(self.image)
            self.gate_data_and_plot()
            self.im_ax.set_aspect(self.aspect_img, adjustable='box-forced')
            MPLPlotter.draw()
        except Exception as e:
            print(e)

    def tr_radio_buttons(self, label):
        try:
            tr_list = self.buffer_data.track_names
            self.selected_track = (tr_list.index(label), label)
            print('selected track index is: ', int(label[5:]))
            self.buffer_data = Form.time_rebin_all_spec_data(
                self.full_data, self.buffer_data.softBinWidth_ns[self.selected_track[0]])
            self.setup_track(*self.selected_track)
            self.image.set_data(np.transpose(self.buffer_data.time_res[self.selected_track[0]][self.selected_pmt_ind]))
            self.colorbar.set_clim(0,
                                   np.max(self.buffer_data.time_res[self.selected_track[0]][self.selected_pmt_ind]))
            self.colorbar.update_normal(self.image)
            self.gate_data_and_plot()
            self.im_ax.set_aspect(self.aspect_img, adjustable='box-forced')
            MPLPlotter.draw()
        except Exception as e:
            print(e)

    def save_proj(self, bool):
        """ saves projection of all tracks """
        # pipeData = self.Pipeline.pipeData
        # time_arr = Form.create_time_axis_from_scan_dict(self.Pipeline.pipeData, rebinning=True)
        # v_arr = Form.create_x_axis_from_scand_dict(self.Pipeline.pipeData, as_voltage=True)
        # rebinned_data = Form.time_rebin_all_data(self.full_data, self.Pipeline.pipeData)
        # data = Form.gate_all_data(pipeData, rebinned_data, time_arr, v_arr)
        # pipeInternals = pipeData['pipeInternals']
        # file = pipeInternals['activeXmlFilePath']
        # rootEle = TildaTools.load_xml(file)
        # tracks, track_list = SdOp.get_number_of_tracks_in_scan_dict(pipeData)
        # for track_ind, tr_num in enumerate(track_list):
        #     track_name = 'track%s' % tr_num
        #     xmlAddCompleteTrack(rootEle, pipeData, data[track_ind][0], track_name, datatype='voltage_projection')
        #     xmlAddCompleteTrack(rootEle, pipeData, data[track_ind][1], track_name, datatype='time_projection')
        # TildaTools.save_xml(rootEle, file, False)
        print('saving currently not implemented')

    def rebin_changed(self, bins_10ns):
        try:
            bins_10ns_rounded = bins_10ns // 10 * 10
            bins_to_combine = bins_10ns_rounded / 10
            self.slider.valtext.set_text('{}'.format(bins_10ns_rounded))
            self.buffer_data = Form.time_rebin_all_spec_data(
                self.full_data, bins_10ns_rounded)
            self.buffer_data.softBinWidth_ns[self.selected_track[0]] = bins_10ns_rounded
            for tr_ind, tr_name in enumerate(self.buffer_data.track_names):
                bins = self.full_data.t[tr_ind].size // bins_to_combine
                print('new length: ', bins, self.buffer_data.softBinWidth_ns[self.selected_track[0]])
                delay_ns = self.full_data.t[tr_ind][0]
                self.buffer_data.t[tr_ind] = np.arange(delay_ns, bins * bins_10ns_rounded + delay_ns, bins_10ns_rounded)
            self.setup_track(*self.selected_track)
            self.image.set_data(np.transpose(self.buffer_data.time_res[self.selected_track[0]][self.selected_pmt_ind]))
            self.colorbar.set_clim(0,
                                   np.max(self.buffer_data.time_res[self.selected_track[0]][self.selected_pmt_ind]))
            self.colorbar.update_normal(self.image)
            self.gate_data_and_plot()
            self.im_ax.set_aspect(self.aspect_img, adjustable='box-forced')
            MPLPlotter.draw()
        except Exception as e:
            print('Exception while rebinning:', e)

    def start(self):
        """ setup the radio buttons and sliders """
        try:
            if self.buffer_data is not None:
                print('start is called')
                track_ind, track_name = (0, 'track0')
                self.selected_track = (track_ind, track_name)
                bin_width = self.buffer_data.softBinWidth_ns[self.selected_track[0]]
                self.selected_pmt_ind = self.buffer_data.active_pmt_list[self.selected_track[0]].index(
                    self.selected_pmt)
                self.setup_track(*self.selected_track)  # setup track with
                if self.radio_buttons_pmt is None:
                    labels = ['pmt%s' % pmt for pmt in self.buffer_data.active_pmt_list[self.selected_track[0]]]
                    self.radio_buttons_pmt, self.radio_con = MPLPlotter.add_radio_buttons(
                        self.pmt_radio_ax, labels, self.selected_pmt_ind, self.pmt_radio_buttons)
                # self.radio_buttons_pmt.set_active(self.selected_pmt_ind)  # not available before mpl 1.5.0
                if self.radio_buttons_tr is None:
                    label_tr = self.buffer_data.track_names
                    self.radio_buttons_tr, con = MPLPlotter.add_radio_buttons(
                        self.tr_radio_ax, label_tr, self.selected_track[0], self.tr_radio_buttons
                    )
                # self.radio_buttons_tr.set_active(self.selected_track[0])  # not available before mpl 1.5.0
                if self.save_button is None:
                    self.save_button, button_con = MPLPlotter.add_button(self.save_bt_ax, 'save_proj', self.save_proj)
                if self.slider is None:
                    self.slider, slider_con = MPLPlotter.add_slider(self.slider_ax, 'rebinning', 10, 100,
                                                                    self.rebin_changed, valfmt=u'%3d', valinit=10)
                self.slider.valtext.set_text('{}'.format(bin_width))
                # self.setup_track(*self.selected_track)
        except Exception as e:
            print(e)

    def processData(self, data, pipeData):
        start = time.clock()
        first_call = self.buffer_data is None
        try:
            self.full_data = deepcopy(data)
            # self.buffer_data = deepcopy(data)
            ret = Form.time_rebin_all_spec_data(
                self.full_data, self.full_data.softBinWidth_ns[self.selected_track[0]])
            self.buffer_data = ret  # here t will have different dimension than nOfBins
            if first_call:
                self.start()
            self.image.set_data(np.transpose(self.buffer_data.time_res[self.selected_track[0]][self.selected_pmt_ind]))
            self.colorbar.set_clim(0,
                                   np.max(self.buffer_data.time_res[self.selected_track[0]][self.selected_pmt_ind]))
            self.colorbar.update_normal(self.image)
            self.gate_data_and_plot()
            self.im_ax.set_aspect(self.aspect_img, adjustable='box-forced')
            pass
        except Exception as e:
            print('while updateing plot, this happened: ', e)
        end = time.clock()
        print('plotting time was /ms : ', round((end - start) * 1000, 3))
        return data

    def clear(self):
        # i dont want to clear this window after completion of scan
        try:
            MPLPlotter.show(True)
        except Exception as e:
            print(e)


class NMPLImagePlotAndSaveSpecData(Node):
    def __init__(self, pmt_num, new_data_callback, new_track_callback, save_callback):
        super(NMPLImagePlotAndSaveSpecData, self).__init__()
        self.type = 'MPLImagePlotSpecData2'
        self.selected_pmt = pmt_num  # for now pmt name should be pmt_ind
        self.stored_data = None
        self.new_data_callback = new_data_callback
        self.new_track_callback = new_track_callback
        self.save_callback = save_callback

    def start(self):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        if self.new_track_callback is not None:
            self.new_track_callback.emit(((track_ind, track_name), (int(self.selected_pmt), self.selected_pmt)))

    def processData(self, data, pipeData):
        if self.new_data_callback is not None:
            self.new_data_callback.emit(deepcopy(data))
        # now there is no chance to get the gates from gui.
        self.stored_data = data
        return data

    def clear(self):
        self.save()
        # pass

    def save(self):
        if self.save_callback is not None:  # saving is done by gui
            self.save_callback.emit(deepcopy(self.Pipeline.pipeData))
        else:  # no gui available
            self.stored_data = TildaTools.gate_specdata(self.stored_data)
            Filehandle.save_spec_data(self.stored_data, self.Pipeline.pipeData)


class NSortedZeroFreeTRSDat2SpecData(Node):
    def __init__(self, x_as_voltage=True):
        """
        when started, will init a SpecData object of given size.
        Overwrites SpecData.time_res with incoming data and passes the SpecData object to the next node.
        Incoming data should be sum
        input: track_list of ndarrays, [array(('sc', 'step', 'time', 'cts'),...), array(...)]
        output: SpecData
        """
        super(NSortedZeroFreeTRSDat2SpecData, self).__init__()
        self.type = 'SortedZeroFreeTRSDat2SpecData'
        self.spec_data = None
        self.x_as_voltage = x_as_voltage

    def start(self):
        if self.spec_data is None:
            self.spec_data = XMLImporter(None, self.x_as_voltage, self.Pipeline.pipeData)
            logging.debug('pipeline successfully loaded: %s' % self.spec_data.file)

    def processData(self, data, pipeData):
        self.spec_data.time_res_zf = data

        return self.spec_data

    def clear(self):
        self.spec_data = None


class NSpecDataZeroFreeProjection(Node):
    def __init__(self):
        """
        Node to gate spec_data with the softw_gates list in the spec_data itself.
        gate will be applied on spec_data.time_res and
        the time projection will be written to spec_data.t_proj
        the voltage projection will be written to spec_data.cts
        input: specdata, time_res is zero free like
        output: SpecData
        """
        super(NSpecDataZeroFreeProjection, self).__init__()
        self.type = 'SpecDataZeroFreeProjection'

    def start(self):
        pass

    def processData(self, data, pipeData):
        ret = TildaTools.gate_zero_free_specdata(data)
        # print(ret.t_proj)
        # print(ret.t_proj[0][np.where(ret.t_proj[0])])
        return ret

    def clear(self):
        pass


""" specdata fitting nodes """


class NStraightKepcoFitOnClear(Node):
    def __init__(self, axes, dmm_names_sorted):
        super(NStraightKepcoFitOnClear, self).__init__()
        self.type = 'SpecDataFittingOnClear'
        self.spec_buffer = None
        self.axes = axes
        self.dmms = dmm_names_sorted  # list with the dmm names, indeces are equal to indices in spec_data.cts, etc.

    def processData(self, data, pipeData):
        self.spec_buffer = data
        return data

    def clear(self):
        for ind, dmm_name in enumerate(self.dmms):
            try:
                fitter = SPFitter(Straight(), self.spec_buffer, ([ind], 0))
                fitter.fit()
                result = fitter.result()
                plotdata = fitter.spec.toPlotE(0, 0, fitter.par)
                self.axes[ind].add_line(MPLPlotter.line2d(plotdata[0], plotdata[1], 'red'))
                MPLPlotter.draw()
                pipe_internals = self.Pipeline.pipeData['pipeInternals']
                file = pipe_internals['activeXmlFilePath']
                db_name = os.path.basename(pipe_internals['workingDirectory']) + '.sqlite'
                db = pipe_internals['workingDirectory'] + '\\' + db_name
                if os.path.isfile(db):  # if the database exists, write fit results to it.
                    con = sqlite3.connect(db)
                    cur = con.cursor()
                    file_base = os.path.basename(file)
                    for r in result:
                        # Only one unique result, according to PRIMARY KEY, thanks to INSERT OR REPLACE
                        cur.execute('''INSERT OR REPLACE INTO FitRes (file, iso, run, rChi, pars)
                        VALUES (?, ?, ?, ?, ?)''', (file_base, r[0], dmm_name, fitter.rchi, repr(r[1])))
                    cur.execute(''' INSERT OR REPLACE INTO Runs (run, lineVar, isoVar, scaler, track)
                                VALUES (?,?,?,?,?)''', (dmm_name, '', '', "[0]", "-1"))
                    con.commit()
                    con.close()
            except Exception as e:
                print('error while fitting:', e)
        self.spec_buffer = None

    def get_offset_voltage(self, scandict):
        mean = 0
        dmms_dict = scandict['measureVoltPars']['preScan'].get('dmms', None)
        if dmms_dict is not None:
            offset = []
            for dmm_name, dmm_dict in dmms_dict.items():
                for key, val in dmm_dict.items():
                    if key == 'preScanRead':
                        if isinstance(val, str):
                            val = float(val)
                        if dmm_dict.get('assignment') == 'offset':
                            offset.append(val)
            if np.any(offset):
                mean = np.mean(offset)
        return mean


""" continous Sequencer / Simple Counter Nodes """


class NCSSortRawDatatoArray(Node):
    def __init__(self):
        """
        Node for sorting the splitted raw data into an scaler Array containing all tracks.
        Missing Values will be set to 0.
        No Value will be emitted twice.
        input: raw data
        output: list of tuples [(scalerArray, scan_complete_flag)... ], missing values are 0
        """
        super(NCSSortRawDatatoArray, self).__init__()
        self.type = 'NSortRawDatatoArray'
        self.scalerArray = None
        self.curVoltIndex = None
        self.totalnOfScalerEvents = None
        self.comp_list = None
        self.info_handl = InfHandl()
        # could be shrinked to active pmts only to speed things up

    def start(self):
        scand = self.Pipeline.pipeData
        tracks, tracks_num_list = SdOp.get_number_of_tracks_in_scan_dict(scand)
        if self.scalerArray is None:
            self.scalerArray = Form.create_default_scaler_array_from_scandict(scand)
        if self.curVoltIndex is None:
            self.curVoltIndex = 0
        if self.totalnOfScalerEvents is None:
            self.totalnOfScalerEvents = np.full((tracks,), 0)
        self.info_handl.setup()
        if self.comp_list is None:
            track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
            self.comp_list = [2 ** j for i, j in enumerate(self.Pipeline.pipeData[track_name]['activePmtList'])]

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        ret = None
        scan_complete = False
        for i, j in enumerate(data):
            first_header, second_header, header_index, payload = Form.split_32b_data(j)
            j = {'headerIndex': header_index, 'firstHeader': first_header,
                 'secondHeader': second_header, 'payload': payload}
            try:
                if j['headerIndex'] == 0:  # its an event from the time resolved sequencer
                    header = (j['firstHeader'] << 4) + j['secondHeader']
                    for pmt_ind, pow2 in enumerate(self.comp_list):
                        if header & pow2:  # bitwise and to determine if this pmt got a count
                            try:
                                self.scalerArray[track_ind][pmt_ind][self.curVoltIndex][j['payload']] += 1
                            except Exception as e:
                                print('error while sorting pmt event into scaler array: ', e,
                                      'scalerArray: ', self.scalerArray)
                                # print('scaler event: ', track_ind, self.curVoltIndex, pmt_ind, j['payload'])
                                # timestamp equals index in time array of the given scaler
                elif j['firstHeader'] == Progs.infoHandler.value:
                    v_ind, step_completed = self.info_handl.info_handle(pipeData, j['payload'])
                    if v_ind is not None:
                        self.curVoltIndex = v_ind
                    compl_steps = pipeData[track_name]['nOfCompletedSteps']
                    nofsteps = pipeData[track_name]['nOfSteps']
                    if self.curVoltIndex > nofsteps:
                        print('voltindex exceeded number of steps, split raw_data is: ', j)
                        raise Exception
                    scan_complete = compl_steps % nofsteps == 0
                    if scan_complete and step_completed:
                        if ret is None:
                            ret = []
                        ret.append((self.scalerArray, scan_complete))
                        logging.debug('Voltindex: ' + str(self.curVoltIndex) +
                                      ' completede steps:  ' + str(pipeData[track_name]['nOfCompletedSteps']) +
                                      ' item is: ' + str(j['payload']))
                        self.scalerArray = Form.create_default_scaler_array_from_scandict(pipeData)
                        # deletes all entries

                elif j['firstHeader'] == Progs.errorHandler.value:  # error send from fpga
                    logging.error('fpga sends error code: ' + str(j['payload']) + 'or in binary: ' + str(
                        '{0:032b}'.format(j['payload'])))

                elif j['firstHeader'] == Progs.dac.value:  # its a voltage step
                    pass

                elif j['firstHeader'] == Progs.continuousSequencer.value:
                    '''scaler entry '''
                    # logging.debug('sorting pmt event to voltage index: ' + str(self.curVoltIndex))
                    self.totalnOfScalerEvents[track_ind] += 1
                    try:  # only add to scalerArray, when pmt is in activePmtList.
                        pmt_index = pipeData[track_name]['activePmtList'].index(j['secondHeader'])
                        # logging.debug('pmt_index is: ' + str(pmt_index))
                        self.scalerArray[track_ind][pmt_index][self.curVoltIndex] += j['payload']
                    except ValueError:
                        pass
            except Exception as e:
                print('error while sorting: ', e, 'split raw data is:', j)
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
        self.comp_list = None
        self.info_handl.clear()


class NCSSum(Node):
    def __init__(self):
        """
        function to sum up all scalerArrays.
        Since no value is emitted twice, arrays can be directly added
        input: list of scalerArrays, complete or uncomplete
        output: scalerArray containing the sum of each scaler, for all tracks
        """
        super(NCSSum, self).__init__()
        self.type = 'CSSum'
        self.scalerArray = None

    def start(self):
        if self.scalerArray is None:
            self.scalerArray = Form.create_default_scaler_array_from_scandict(self.Pipeline.pipeData)

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        try:
            for i, j in enumerate(data):
                self.scalerArray[track_ind] = np.add(self.scalerArray[track_ind], j[track_ind])
                # logging.debug('max pmt val is:' + str(np.amax(self.scalerArray[track_ind][0])))
                # logging.debug('sum is: ' + str(self.scalerArray[0][0:2]) + str(self.scalerArray[0][-2:]))
            return self.scalerArray
        except Exception as e:
            print('exception: ', e)

    def clear(self):
        self.scalerArray = None


class NSPSortByPmt(Node):
    """
    Node for the Simple Counter which will store a limited amount of datapoints per pmt.
    if more datapoints are fed to the pipeline at once,
    than defined in init, first incoming will be ignored.
    self.buffer = [[pmt0ct599, pmt0ct598, ..., pmt0ct0], ... [pmt7ct599,...]]
    newest values are inserted at lowest index.
    input: splitted raw data
    output: [pmt0, pmt1, ... pmt(7)] with len(pmt0-7) = datapoints
    """

    def __init__(self, datapoints):
        super(NSPSortByPmt, self).__init__()
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


class NSPAddxAxis(Node):
    """
    Node for the Simple Counter which add an x-axis to the moving average,
    to make it plotable with NMPlLivePlot
    jnput: [avg_pmt0, avg_pmt1, ... , avg_pmt7], avg_pmt(0-7) = float
    output: [(x0, y0), (x1, y1), ...] len(x0) = len(y0) = plotPoints, y0[i] = avg_pmt0
    """

    def __init__(self):
        super(NSPAddxAxis, self).__init__()
        self.type = 'SPAddxAxis'
        self.buffer = None

    def start(self):
        plotpoints = self.Pipeline.pipeData.get('plotPoints')
        n_of_pmt = len(self.Pipeline.pipeData.get('activePmtList'))
        self.buffer = np.zeros((n_of_pmt, 2, plotpoints,))
        self.buffer[:, 0] = np.arange(0, plotpoints)

    def processData(self, data, pipeData):
        for pmt_ind, avg_pmt in enumerate(data):
            self.buffer[pmt_ind][1] = np.roll(self.buffer[pmt_ind][1], 1)
            self.buffer[pmt_ind][1][0] = avg_pmt
        return self.buffer


""" time resolved Sequencer Nodes """


class NTRSSortRawDatatoArrayFast(Node):
    def __init__(self):
        """
        Node for sorting the raw data to the corresponding scaler, step and timestamp.
        No Value will be emitted twice.
        pipeData[track_name]['nOfCompletedSteps'] will be updated.
        input: numpy.ndarray, raw data (32Bit Elements)
        output: numpy.ndarray, dtype=[('tr', 'u2'), ('sc', 'u2'), ('step', 'u4'), ('time', 'u4'), ('cts', 'u4')]
        """
        super(NTRSSortRawDatatoArrayFast, self).__init__()
        self.type = 'TRSSortRawDatatoArrayFast'
        self.curVoltIndex = None
        self.comp_list = None
        self.stored_data = None  # numpy array of incoming raw data elements.
        self.total_num_of_started_scans = None
        self.completed_steps_this_track = None

        # could be shrinked to active pmts only to speed things up

    def start(self):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        if self.curVoltIndex is None:
            self.curVoltIndex = 0
        if self.total_num_of_started_scans is None:
            self.total_num_of_started_scans = 0
        if self.comp_list is None:
            self.comp_list = [2 ** j for i, j in enumerate(self.Pipeline.pipeData[track_name]['activePmtList'])]
        if self.stored_data is None:
            self.stored_data = np.zeros(0, dtype=np.uint32)
        if self.completed_steps_this_track is None:
            self.completed_steps_this_track = self.Pipeline.pipeData[track_name].get('nOfCompletedSteps', 0)
        self.Pipeline.pipeData[track_name]['nOfCompletedSteps'] = self.completed_steps_this_track  # make sure this exists

    def processData(self, data, pipeData):
        print('sort array')
        self.stored_data = np.append(self.stored_data, data)
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        step_complete = Form.add_header_to23_bit(1, 4, 0, 1)  # binary for step complete
        scan_started = Form.add_header_to23_bit(2, 4, 0, 1)  # binary for scan started
        new_bunch = Form.add_header_to23_bit(3, 4, 0, 1)  # binary for new bunch
        dac_int_key = 2 ** 29 + 2 ** 28 + 2 ** 23  # binary key for an dac element
        header_index = 2 ** 23  # binary for the headerelement
        step_complete_ind_list = np.where(self.stored_data == step_complete)[0]
        if step_complete_ind_list.size:  # only work with complete steps.
            pipeData[track_name]['nOfCompletedSteps'] += step_complete_ind_list.size
            scan_start_before_step_comp = False
            scan_started_ind_list = np.where(self.stored_data == scan_started)[0]
            if scan_started_ind_list.size:
                scan_start_before_step_comp = scan_started_ind_list[0] < step_complete_ind_list[0]
                if scan_start_before_step_comp:
                    self.total_num_of_started_scans += 1
            x_one_scan = np.arange(0, pipeData[track_name]['nOfSteps'])  # "x-axis" for one scan
            # make it also for two scans and invert on second rep if needed.
            x_two_scans = np.append(x_one_scan,
                                    np.fliplr(x_one_scan) if pipeData[track_name]['invertScan'] else x_one_scan)
            # repeat this as often as needed for all steps held in this data set.
            x_this_data = np.tile(x_two_scans,
                                  np.ceil(step_complete_ind_list.size / pipeData[track_name]['nOfSteps'] / 2))
            # roll this, so that the current step stands at position 0.
            x_this_data = np.roll(x_this_data,
                                  -(self.curVoltIndex + pipeData[track_name]['nOfSteps'])
                                  if pipeData[track_name]['invertScan'] and self.total_num_of_started_scans % 2 == 0
                                  else -self.curVoltIndex)[0:step_complete_ind_list.size]
            # print('starting with voltindex: ', x_this_data[0])
            # get position of all pmt events (trs:)
            pmt_events_ind = np.where(self.stored_data & header_index == 0)[0]  # indices of all pmt events (for trs)
            # create an array which repeatedly holds the stepnumber which is active for the element at this position
            # by indexing this, one can directly see the right stepnumber for the element with the index of interest.
            pmt_steps = np.repeat(x_this_data,
                                  np.insert(np.diff(step_complete_ind_list, 1), 0, step_complete_ind_list[0]))
            if pmt_events_ind.size:
                # cut pmt events which are not still in a completed step:
                pmt_events_ind = pmt_events_ind[pmt_events_ind < pmt_steps.size]
                # create a list of stepnumbers for all pmt events:
                pmt_steps = pmt_steps[pmt_events_ind]
                self.curVoltIndex = x_this_data[-1] + 1
                self.total_num_of_started_scans += scan_started_ind_list.size - 1 if scan_start_before_step_comp else 0
                # new_bunch_ind = np.where(self.stored_data == new_bunch)[0]  # ignore for now.
                # dac_set_ind = np.where(self.stored_data & dac_int_key == dac_int_key)[0]  # info not needed mostly
                # create a list with all timestamps
                pmt_events_time = self.stored_data[pmt_events_ind] & (2 ** 23 - 1)  # get only the time stamp
                # create a list with all scaler numbers
                pmt_events_scaler = self.stored_data[
                                        pmt_events_ind] >> 24  # get the header where the pmt info is stored.
                # combine the created arrays to one new array
                new_arr = np.zeros(len(pmt_events_ind), dtype=[('sc', 'u2'), ('step', 'u4'),
                                                               ('time', 'u4'), ('cts', 'u4')])
                new_arr['sc'] = pmt_events_scaler  # currently all are written to one so 255 = all pmts active
                new_arr['step'] = pmt_steps  # how to do this without for loop? pmt_evt_ind < step ...
                new_arr['time'] = pmt_events_time
                # create a unique array, so all double occurences of the given data are counted
                unique_arr, cts = np.unique(new_arr, return_counts=True)
                # ... and put into cts in the unique array:
                unique_arr['cts'] = cts
                # print(unique_arr)
                new_unique_arr = np.zeros(0, dtype=[('sc', 'u2'), ('step', 'u4'), ('time', 'u4'), ('cts', 'u4')])
                # check for events with multiple pmts fired at once (only for active pmts):
                for act_pmt in self.comp_list:
                    # create new array with all elements where this pmt was active:
                    if np.where(unique_arr['sc'] & act_pmt)[0].size:
                        ith_pmt_hit_list = unique_arr[np.where(unique_arr['sc'] & act_pmt)]
                        if np.any(ith_pmt_hit_list['step']):  # cannot do any for full list, must select step or so
                            ith_pmt_hit_list['sc'] = int(np.log2(act_pmt))
                            new_unique_arr = np.append(new_unique_arr, ith_pmt_hit_list)
                            # print(new_unique_arr)
                self.stored_data = self.stored_data[step_complete_ind_list[-1] + 1:]
                # new_unique_arr = np.sort(new_unique_arr, axis=0)
                # print(new_unique_arr)
                # print('current voltindex after first node:', self.curVoltIndex)

                print('sort array done')

                return new_unique_arr

    def clear(self):
        self.curVoltIndex = None
        self.comp_list = None
        self.stored_data = None
        self.total_num_of_started_scans = None
        self.completed_steps_this_track = None

    def sign_for_volt_ind(self, invert_scan):
        """
        retrun the sign for increasing or decreasing the voltindex,
        depending if it is inverted or not and if the scan is odd or even.
        :type invert_scan: bool
        """
        even_scan_num = self.total_num_of_started_scans % 2 == 0
        if invert_scan:
            if even_scan_num:  # in every even scan, scan dir turns around
                return -1
            else:
                return 1
        return 1

    def voltindex_after_scan_start(self, invert_scan):
        """
        give the voltindex for the first element in a new scan,
        either 0 or -1 depending if it is inverted or not and if the scan is odd or even.
        :param invert_scan:
        :return: -1 or 0
        """
        if invert_scan:  # if inverted, take last element on every second scan
            if self.total_num_of_started_scans % 2 == 0:
                volt_index = -1
            else:
                volt_index = 0
        else:
            volt_index = 0
        return volt_index


class NTRSSumFastArrays(Node):
    def __init__(self):
        """
        sums up incoming ndarrays of the active track.
        input: numpy.ndarray, dtype=[('tr', 'u2'), ('sc', 'u2'), ('step', 'u4'), ('time', 'u4'), ('cts', 'u4')]
        output: track_list of ndarrays, [array(('sc', 'step', 'time', 'cts'),...), array(...)]
        """
        super(NTRSSumFastArrays, self).__init__()
        self.type = 'TRSSumFastArrays'
        self.sum = None

    def start(self):
        if self.sum is None:
            tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(self.Pipeline.pipeData)
            self.sum = [np.zeros(0, dtype=[('sc', 'u2'), ('step', 'u4'),
                                           ('time', 'u4'), ('cts', 'u4')]) for tr in range(tracks)]

    def processData(self, data, pipeData):
        # sc,step,time not in list -> append
        # else: sum cts, each not unique element can only be there twice!
        # -> one from before storage, one from new incoming.
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        # before_app = self.sum[track_ind].size
        # before_app_data_sz = data.size
        if self.sum[track_ind].size:
            appended_arr = np.append(self.sum[track_ind], data)  # first append all data to sum
            # sort by 'sc', 'step', 'time' (no cts):
            sorted_arr = np.sort(appended_arr, order=['sc', 'step', 'time'])
            # find all elements that occur twice:
            unique_arr, unique_inds, uniq_cts = np.unique(sorted_arr[['sc', 'step', 'time']],
                                                          return_index=True, return_counts=True)
            sum_ind = unique_inds[np.where(uniq_cts == 2)]  # only take indexes of double occuring items
            # use indices of all twice occuring elements to add the counts of those:
            sum_cts = sorted_arr[sum_ind]['cts'] + sorted_arr[sum_ind + 1]['cts']
            np.put(sorted_arr['cts'], sum_ind, sum_cts)
            # delete all remaining items:
            self.sum[track_ind] = np.delete(sorted_arr, sum_ind + 1, axis=0)

            # alternative with using where (maybe use if not happy anymore with above solution.
            # not_in_sum = np.unique(
            #     np.where(data[['tr', 'sc', 'step', 'time']] != self.sum[['tr', 'sc', 'step', 'time']])[0])
            # self.sum = np.append(self.sum, data[not_in_sum])
            # data = np.delete(data, not_in_sum, axis=0)
            # if data.size:
            #     already_in_sum = np.unique(
            #         np.where(data[['tr', 'sc', 'step', 'time']] == self.sum[['tr', 'sc', 'step', 'time']])[0])
            #     sum_indices = np.unique(
            #         np.where(self.sum[['tr', 'sc', 'step', 'time']] == data[['tr', 'sc', 'step', 'time']])[0])
            #     print(data[already_in_sum], self.sum[sum_indices])
            #     # those produce empty lists currently. Fix this tomorrow!
            #     # make the sum here.
            #     pass
        else:  # sum was empty before, so data can be just appended.
            self.sum[track_ind] = np.append(self.sum[track_ind], data)
        # print('sum is: %s ' % self.sum)
        # print('data length before append: %s and remaining after append: %s,'
        #       ' sum length before append: %s and after append %s '
        #       % (before_app_data_sz, data.size, before_app, self.sum.size))
        return self.sum

    def clear(self):
        self.sum = None


class NTRSProjectize(Node):
    """
    Node for projectize incoming 2d Data on the voltage and time axis.
    jnput: "2d" data as comming from NSum for Trs
    output: [[[v_proj_tr0_pmt0, v_proj_tr0_pmt1, ... ], [t_proj_tr0_pmt0, t_proj_tr0_pmt1, ... ]], ...]
    """

    def __init__(self, as_voltage=True):
        super(NTRSProjectize, self).__init__()
        self.type = 'TRSProjectize'
        self.volt_array = None
        self.time_array = None
        self.as_voltage = as_voltage

    def start(self):
        self.volt_array = Form.create_x_axis_from_scand_dict(
            self.Pipeline.pipeData, as_voltage=self.as_voltage)
        self.time_array = Form.create_time_axis_from_scan_dict(
            self.Pipeline.pipeData)

    def processData(self, data, pipeData):
        return Form.gate_all_data(pipeData, data, self.time_array, self.volt_array)


class NTRSRebinAllData(Node):
    """
    This Node will rebin the incoming data which means that the timing resolution is reduced by
    combining time bins within a given time window. The time window is set via the 'softBinWidth_ns'
    item in each track dictionary.
    jnput: "2d" data as comming from NSum for Trs.
    output: "2d" data similiar to input but time axis is reduced.
    """

    def __init__(self):
        super(NTRSRebinAllData, self).__init__()
        self.type = 'TRSRebinAllData'

    def processData(self, data, pipeData):
        return Form.time_rebin_all_data(data, pipeData)


""" QT Signal Nodes """


class NSendnOfCompletedStepsViaQtSignal(Node):
    """
    Node for sending the number of completed Steps in the pipedata via a qt signal
    input: anything
    output: same as input
    """

    def __init__(self, qt_signal):
        super(NSendnOfCompletedStepsViaQtSignal, self).__init__()
        self.type = 'SendnOfCompletedStepsViaQtSignal'

        self.qt_signal = qt_signal

    def processData(self, data, pipeData):
        print('sending steps')
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        self.qt_signal.emit(pipeData[track_name]['nOfCompletedSteps'])
        print('sending steps done')
        return data


class NSendnOfCompletedStepsAndScansViaQtSignal(Node):
    """
    Node for sending the number of completed Steps in the pipedata and
    the number of started scans via a qt signal
    input: anything
    output: same as input
    """

    def __init__(self, steps_scans_callback):
        super(NSendnOfCompletedStepsAndScansViaQtSignal, self).__init__()
        self.type = 'SendnOfCompletedStepsViaQtSignal'

        self.qt_signal = steps_scans_callback

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        compl_steps = pipeData[track_name]['nOfCompletedSteps']
        steps = pipeData[track_name]['nOfSteps']
        scans = compl_steps // steps
        self.qt_signal.emit({'nOfCompletedSteps': compl_steps, 'nOfStartedScans': scans})
        return data

    def clear(self):
        # self.qt_signal.emit({'nOfCompletedSteps': 0, 'nOfStartedScans': 0})
        pass


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


""" Tilda passvie Nodes """


class NTiPaAccRawUntil2ndScan(Node):
    """
    Node for accumulating raw data until a second scan is fired.
    When the second scan is registered,
    Tilda knows how many steps are performed within each MCP Scan.
    Then the buffer is forwarded to the next nodes and the pipeData is updated.
    input: 32-Bit rawdata
    output: 32-Bit rawdata
    """

    def __init__(self, steps_scans_callback):
        super(NTiPaAccRawUntil2ndScan, self).__init__()
        self.type = 'TiPaAccRawUntil2ndScan'
        self.buffer = np.zeros(0, dtype=np.uint32)
        self.acquired_2nd_scan = False
        self.n_of_started_scans = 0
        self.n_of_compl_steps = 0
        self.steps_scans_callback = steps_scans_callback

    def processData(self, data, pipeData):
        if self.acquired_2nd_scan:
            return data
        else:
            pipeData['track0']['nOfCompletedSteps'] = 0
            self.buffer = np.append(self.buffer, data)
            step_complete = int('01000000100000000000000000000001', 2)
            new_scan = int('01000000100000000000000000000010', 2)
            scans_in_buffer = np.where(self.buffer == new_scan)[0]
            # np array of indices of the new scan signals in the buffer data
            if scans_in_buffer.size == 1:
                self.buffer = self.buffer[scans_in_buffer[0]:]
                steps = np.where(self.buffer == step_complete)[0].size
                self.emit_steps_scan_callback(steps, 0)
            if scans_in_buffer.size >= 2:
                self.acquired_2nd_scan = True
                one_scan = self.buffer[scans_in_buffer[0]:scans_in_buffer[1]]
                nOfsteps = np.where(one_scan == step_complete)[0].size
                pipeData['track0']['nOfSteps'] = nOfsteps
                logging.debug('number of steps should be %s' % nOfsteps)
                steps = np.where(self.buffer == step_complete)[0].size
                self.emit_steps_scan_callback(steps, scans_in_buffer.size)
                self.Pipeline.start()
                self.Pipeline.feed(self.buffer)
            return None

    def emit_steps_scan_callback(self, steps, scans):
        """  a gui can rcv this callback signal """
        self.steps_scans_callback.emit({'nOfCompletedSteps': steps,
                                        'nOfStartedScans': scans})


""" filtering Nodes """


class NFilterDMMDicts(Node):
    def __init__(self):
        """
        Node, that will not return any data coming from the dmm.
        Which is identified by being a dict.
        """
        super(NFilterDMMDicts, self).__init__()
        self.type = 'FilterDMMDicts'

    def processData(self, data, pipeData):
        if isinstance(data, dict):
            return None
        else:
            return data
