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
from Service.AnalysisAndDataHandling.InfoHandler import InfoHandler as InfHandl
import MPLPlotter
import numpy as np
import logging

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
        """
        super(NCheckIfTrackComplete, self).__init__()
        self.type = 'CheckIfTrackComplete'

    def processData(self, data, pipeData):
        ret = None
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        if CsAna.checkIfTrackComplete(pipeData, track_name):
            ret = data
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


""" saving """


class NSaveIncomDataForActiveTrack(Node):
    def __init__(self):
        """
        function to save all incoming CS-Sum-Data.
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
    def __init__(self, pmt_num):
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
        self.time_array = None
        self.radio_buttons_pmt = None
        self.radio_con = None
        self.radio_buttons_tr = None
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
            v_min_ind, v_min, vdif = Form.find_closest_value_in_arr(self.volt_array, v_min)
            v_max_ind, v_max, vdif = Form.find_closest_value_in_arr(self.volt_array, v_max)
            
            t_min, t_max = sorted((gates_val_list[2], gates_val_list[3]))
            t_min_ind, t_min, tdif = Form.find_closest_value_in_arr(self.time_array, t_min)
            t_max_ind, t_max, tdif = Form.find_closest_value_in_arr(self.time_array, t_max)
            gates_ind = [v_min_ind, v_max_ind, t_min_ind, t_max_ind]  # indices in data array
            gates_val_list = [v_min, v_max, t_min, t_max]
            self.gates_list[self.selected_pmt_ind] = [gates_val_list, gates_ind]
            self.Pipeline.pipeData[self.selected_track[1]]['softwGates'][self.selected_pmt_ind] = gates_val_list
            if self.gate_anno is None:
                self.gate_anno = self.im_ax.annotate('%s - %s V \n%s - %s ns'
                                                     % (self.volt_array[v_min_ind], self.volt_array[v_max_ind],
                                                        self.time_array[t_min_ind], self.time_array[t_max_ind]),
                                                     xy=(self.im_ax.get_xlim()[0], self.im_ax.get_ylim()[1]/2),
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
            for ax in [val for sublist in self.axes for val in sublist][:-2]:
                if ax:  # be sure ax is not 0, don't clear radiobuttons
                    MPLPlotter.clear_ax(ax)
            self.gate_anno = None
            self.gates_list = [[None]] * len(self.Pipeline.pipeData[track_name]['activePmtList'])
            self.volt_array = Form.create_x_axis_from_scand_dict(self.Pipeline.pipeData, as_voltage=True)[track_ind]
            v_shape = self.volt_array.shape
            self.time_array = Form.create_time_axis_from_scan_dict(self.Pipeline.pipeData)[track_ind]
            t_shape = self.time_array.shape
            self.image, self.colorbar = MPLPlotter.configure_image_plot(
                self.fig, self.axes, self.Pipeline.pipeData, self.volt_array,
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
                self.Pipeline.pipeData[track_name]['softwGates'] =\
                    [[None]] * len(self.Pipeline.pipeData[track_name]['activePmtList'])
            else:  # read gates from input
                gate_val_list = self.Pipeline.pipeData[track_name].get('softwGates', None)[self.selected_pmt_ind]
            self.update_gate_ind(gate_val_list)

            MPLPlotter.draw()
        except Exception as e:
            print('while starting this occured: ', e)

    def pmt_radio_buttons(self, label):
        self.selected_pmt = int(label[3:])
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        self.selected_pmt_ind = self.Pipeline.pipeData[track_name]['activePmtList'].index(self.selected_pmt)
        print('selected pmt index is: ', int(label[3:]))
        self.setup_track(track_ind, track_name)
        self.buffer_data = self.full_data[track_ind][self.selected_pmt_ind]
        self.image.set_data(np.transpose(self.buffer_data))
        self.colorbar.set_clim(0, np.amax(self.buffer_data))
        self.colorbar.update_normal(self.image)
        self.gate_data_and_plot()
        self.im_ax.set_aspect(self.aspect_img, adjustable='box-forced')
        MPLPlotter.draw()

    def tr_radio_buttons(self, label):
        tr, tr_list = SdOp.get_number_of_tracks_in_scan_dict(self.Pipeline.pipeData)
        self.selected_track = (tr_list.index(int(label[5:])), label)
        self.setup_track(*self.selected_track)
        print('selected track index is: ', int(label[5:]))
        self.buffer_data = self.full_data[self.selected_track[0]][self.selected_pmt_ind]
        self.image.set_data(np.transpose(self.buffer_data))
        self.colorbar.set_clim(0, np.amax(self.buffer_data))
        self.colorbar.update_normal(self.image)
        self.gate_data_and_plot()
        self.im_ax.set_aspect(self.aspect_img, adjustable='box-forced')
        MPLPlotter.draw()

    def start(self):
        track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
        self.selected_track = (track_ind, track_name)
        self.selected_pmt_ind = self.Pipeline.pipeData[self.selected_track[1]]['activePmtList'].index(self.selected_pmt)
        self.setup_track(*self.selected_track)
        if self.radio_buttons_pmt is None:
            labels = ['pmt%s' %pmt for pmt in self.Pipeline.pipeData[self.selected_track[1]]['activePmtList']]
            self.radio_buttons_pmt, self.radio_con = MPLPlotter.add_radio_buttons(
                        self.pmt_radio_ax, labels, self.selected_pmt_ind, self.pmt_radio_buttons)
        if self.radio_buttons_tr is None:
            tr, tr_list = SdOp.get_number_of_tracks_in_scan_dict(self.Pipeline.pipeData)
            label_tr = ['track%s' % tr_num for tr_num in tr_list]
            self.radio_buttons_tr, con = MPLPlotter.add_radio_buttons(
                self.tr_radio_ax, label_tr, self.selected_track[0], self.tr_radio_buttons
            )

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        try:
            self.full_data = data
            self.buffer_data = data[track_ind][self.selected_pmt_ind]
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


""" specdata format compatible Nodes: """


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


""" continous Sequencer / Simple Counter Nodes """


class NCSSortRawDatatoArray(Node):
    def __init__(self):
        """
        Node for sorting the splitted raw data into an scaler Array containing all tracks.
        Missing Values will be set to 0.
        No Value will be emitted twice.
        input: split raw data
        output: list of tuples [(scalerArray, scan_complete_flag)... ], missing values are 0
        """
        super(NCSSortRawDatatoArray, self).__init__()
        self.type = 'NSortRawDatatoArray'
        self.voltArray = None
        self.scalerArray = None
        self.curVoltIndex = None
        self.totalnOfScalerEvents = None
        self.comp_list = None
        self.info_handl = InfHandl()
        # could be shrinked to active pmts only to speed things up

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
        self.info_handl.setup()
        if self.comp_list is None:
            track_ind, track_name = self.Pipeline.pipeData['pipeInternals']['activeTrackNumber']
            self.comp_list = [2 ** j for i, j in enumerate(self.Pipeline.pipeData[track_name]['activePmtList'])]

    def processData(self, data, pipeData):
        track_ind, track_name = pipeData['pipeInternals']['activeTrackNumber']
        ret = None
        scan_complete = False
        for i, j in enumerate(data):
            if j['headerIndex'] == 0:  # its an event from the time resolved sequencer
                header = (j['firstHeader'] << 4) + j['secondHeader']
                for pmt_ind, pow2 in enumerate(self.comp_list):
                    if header & pow2:  # bitwise and to determine if this pmt got a count
                        try:
                            self.scalerArray[track_ind][pmt_ind][self.curVoltIndex][j['payload']] += 1
                        except Exception as e:
                            print('excepti : ', e)
                            # print('scaler event: ', track_ind, self.curVoltIndex, pmt_ind, j['payload'])
                            # timestamp equals index in time array of the given scaler
            elif j['firstHeader'] == ProgConfigsDict.programs['infoHandler']:
                self.info_handl.info_handle(pipeData, j['payload'])
                scan_complete = pipeData[track_name]['nOfCompletedSteps'] == pipeData[track_name]['nOfSteps']
                if scan_complete:
                    if ret is None:
                        ret = []
                    ret.append((self.scalerArray, scan_complete))
                    logging.debug('Voltindex: ' + str(self.curVoltIndex) +
                                  'completede steps:  ' + str(pipeData[track_name]['nOfCompletedSteps']))
                    self.scalerArray = Form.create_default_scaler_array_from_scandict(pipeData)  # deletes all entries
                    scan_complete = False

            elif j['firstHeader'] == ProgConfigsDict.programs['errorHandler']:  # error send from fpga
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
        self.comp_list = None
        self.info_handl.clear()


class NCSSum(Node):
    def __init__(self):
        """
        function to sum up all scalerArrays.
        Since no value is emitted twice, arrays can be directly added
        input: list of scalerArrays, complete or uncomplete
        output: scalerArray containing the sum of each scaler, voltage
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


class NTRSSaveSum(Node):
    def __init__(self):
        """
        save the summed up data
        incoming must always be a tuple of form:
        (self.voltArray, self.timeArray, self.scalerArray)
        Note: will always save when data is passed to it. So only send complete structures.
        """
        super(NTRSSaveSum, self).__init__()
        self.type = "NTRSSaveSum"

    def processData(self, data, pipeData):
        pipeInternals = pipeData['pipeInternals']
        file = pipeInternals['activeXmlFilePath']
        rootEle = TildaTools.load_xml(file)
        xmlAddCompleteTrack(rootEle, pipeData, data)
        TildaTools.save_xml(rootEle, file, False)
        return data


""" QT Signal Nodes """


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
