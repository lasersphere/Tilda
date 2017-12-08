"""
Created on 

@author: simkaufm

Module Description: GUI for displaying of data, both live and from file.
The data is analysed by a designated pipeline and then the data is emitted vie pyqtsignals to the gui.
Here it is only displayed. Gating etc. is done by the pipelines.

"""
import ast
import functools
import logging
from copy import deepcopy
from datetime import datetime

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

import Application.Config as Cfg
import PyQtGraphPlotter as Pg
from Interface.LiveDataPlottingUi.Ui_LiveDataPlotting import Ui_MainWindow_LiveDataPlotting
from Interface.ScanProgressUi.ScanProgressUi import ScanProgressUi
from Measurement.XMLImporter import XMLImporter


class TRSLivePlotWindowUi(QtWidgets.QMainWindow, Ui_MainWindow_LiveDataPlotting):
    # these callbacks should be called from the pipeline:
    # for incoming new data:
    new_data_callback = QtCore.pyqtSignal(XMLImporter)
    # if a new track is started call:
    # the tuple is of form: ((tr_ind, tr_name), (pmt_ind, pmt_name))
    new_track_callback = QtCore.pyqtSignal(tuple)
    # when the pipeline wants to save, this is emitted and it send the pipeData as a dict
    save_callback = QtCore.pyqtSignal(dict)

    # dict, fit result plot data callback
    # -> this can be emitted from a node to send a dict containing fit results:
    # 'plotData': tuple of ([x], [y]) values to plot a fit result.
    # 'result': list of result-tuples (name, pardict, fix)
    fit_results_dict_callback = QtCore.pyqtSignal(dict)

    # signal to request updated gated data from the pipeline.
    # list: software gates [[[tr0_sc0_vMin, tr0_sc0_vMax, tr0_sc0_tMin, tr0_sc0_tMax], [tr0_sc1_...
    # int: track_index to rebin -1 for all
    # list: software bin width in ns for each track
    # bool: plot bool to force a plotting even if nothing has changed.
    new_gate_or_soft_bin_width = QtCore.pyqtSignal(list, int, list, bool)

    # save request
    save_request = QtCore.pyqtSignal()

    # progress dict coming from the main
    progress_callback = QtCore.pyqtSignal(dict)

    def __init__(self, full_file_path='', parent=None, subscribe_as_live_plot=True, sum_sc_tr=None):
        """
        initilaises a liveplot window either for liveplotting of incoming data
        or to display previously saved xml data
        :param full_file_path: str, path of xml file.
        :param parent: reference to parent window
        :param subscribe_as_live_plot: bool,
            True for subscribing as live plot window (9only one allowed)
            False for displaying previously stored data
        :param sum_sc_tr: list, can be used to overwrite on startup
         which scalers of which track should be added for the sum. Syntax as in Pollifit -> Runs table scaler, track
            [[+/-sc0, +/-sc1,...], tr]
                sc0-n: index of the scaler that should be added,
                tr: index of the track that should be added, -1 for all tracks
        """
        super(TRSLivePlotWindowUi, self).__init__(parent=parent)

        self.t_proj_plt = None
        self.last_event = None
        self.pipedata_dict = None  # dict, containing all infos from the pipeline, will be passed
        #  to gui when save request is called from pipeline
        self.active_track_name = None  # str, name of the active track
        self.active_initial_scan_dict = None  # scan dict which is stored under the active iso name in the main
        self.active_file = None  # str, name of active file
        self.active_iso = None  # str, name of active iso in main
        self.overall_scan_progress = 0  # float, will be 1 when scan completed updated via scan progress dict from main
        self.setupUi(self)
        self.show()
        self.tabWidget.setCurrentIndex(1)  # time resolved
        self.setWindowTitle('plot win:     ' + full_file_path)
        self.dockWidget.setWindowTitle('progress: %s' % self.active_file)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # necessary for not keeping it in memory

        self.full_file_path = full_file_path

        self.sum_plt_data = None
        self.trs_names_list = ['trs', 'trsdummy', 'tipa']

        self.tres_image = None
        self.t_proj_plt_itm = None
        self.tres_plt_item = None
        self.spec_data = None  # spec_data to work on.
        self.new_track_no_data_yet = False  # set this to true when new track is setup

        self.last_gr_update_done_time = datetime.now()

        self.graph_font_size = int(14)

        ''' connect callbacks: '''
        # bundle callbacks:
        self.subscribe_as_live_plot = subscribe_as_live_plot
        self.get_existing_callbacks_from_main()
        self.callbacks = (self.new_data_callback, self.new_track_callback,
                          self.save_request, self.new_gate_or_soft_bin_width)
        self.subscribe_to_main()

        ''' key press '''
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Up), self,
                            functools.partial(self.raise_graph_fontsize, True))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Down), self,
                            functools.partial(self.raise_graph_fontsize, False))

        ''' sum related '''
        self.add_sum_plot()
        self.sum_x, self.sum_y, self.sum_err = None, None, None  # storage of the sum plotting values

        self.sum_scaler = [0]  # list of scalers to add
        self.sum_track = -1  # int, for selecting the track which will be added. -1 for all
        self.sum_sc_tr_external = sum_sc_tr
        if self.sum_sc_tr_external is not None:
            # overwrite with external
            self.sum_scaler = self.sum_sc_tr_external[0]
            self.sum_track = self.sum_sc_tr_external[1]
            self.lineEdit_arith_scaler_input.setText(str(self.sum_scaler))
            self.lineEdit_sum_all_pmts.setText(str(self.sum_scaler))

        self.current_step_line = None  # line to display which step is active.(used in projection)
        self.sum_current_step_line = None  # same for sum

        self.sum_list = ['add all', 'manual']
        self.comboBox_select_sum_for_pmts.addItems(self.sum_list)
        self.comboBox_select_sum_for_pmts.currentIndexChanged.connect(self.sum_scaler_changed)
        self.comboBox_select_sum_for_pmts.currentIndexChanged.emit(0)

        self.lineEdit_arith_scaler_input.textEdited.connect(self.sum_scaler_lineedit_changed)

        ''' time resolved related: '''
        self.add_time_resolved_plot()
        self.tres_sel_tr_ind = 0  # int, index of currently observed track in time resolved spectra
        self.tres_sel_tr_name = 'track0'  # str, name of track
        self.tres_sel_sc_ind = 0  # int, index of currently observed scaler in time resolved spectra
        self.tres_sel_sc_name = '0'  # str, name of pmt

        self.tableWidget_gates.itemClicked.connect(self.handle_item_clicked)
        self.tableWidget_gates.itemChanged.connect(self.handle_gate_table_change)

        self.spinBox.valueChanged.connect(self.rebin_data)
        self.checkBox.stateChanged.connect(self.apply_rebin_to_all_checkbox_changed)

        self.setup_range_please = True  # boolean to store if the range has ben setup yet or not

        ''' all pmts related: '''
        #  dict for all_pmt_plot page containing a dict with the keys:
        # 'widget', 'proxy', 'vertLine', 'indList', 'pltDataItem', 'name', 'pltItem', 'fitLine' for each plot:
        self.all_pmts_widg_plt_item_list = None

        self.all_pmts_sel_tr = 0
        self.comboBox_all_pmts_sel_tr.currentTextChanged.connect(self.cb_all_pmts_sel_tr_changed)

        self.comboBox_sum_all_pmts.addItems(self.sum_list)
        self.comboBox_sum_all_pmts.currentIndexChanged.connect(self.sum_scaler_changed)
        self.comboBox_sum_all_pmts.currentIndexChanged.emit(0)

        self.lineEdit_sum_all_pmts.textEdited.connect(self.sum_scaler_lineedit_changed)

        ''' setup window size: '''
        w = 1024
        h = 800
        self.resize(w, h)
        ''' vertical splitter between plots and table: '''
        self.splitter.setSizes([h * 8 // 10, h * 2 // 10])
        ''' horizontal splitter between tres and t_proj: '''
        self.splitter_2.setSizes([w * 2 // 3, w * 1 // 3])
        ''' vertical splitter between tres and v_proj/sum_proj: '''
        self.splitter_4.setSizes([h // 2, h // 2])
        ''' horizontal splitter between v_proj and the display widget: '''
        self.splitter_3.setSizes([w * 6 // 10, w * 4 // 10])
        ''' vertical splitter between all pmts plot and x/y coords widg '''
        self.splitter_allpmts.setSizes([h * 9 // 10, h * 1 // 10])

        ''' progress related: '''
        self.scan_prog_ui = None
        self.show_progress_window()

        self.actionProgress.setCheckable(True)
        self.actionProgress.setChecked(self.subscribe_as_live_plot)

        self.show_progress_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_P), self)

        self.show_progress_shortcut.activated.connect(self.show_progress)
        self.actionProgress.triggered.connect(self.show_progress)

        ''' font size graphs '''
        self.actionGraph_font_size.triggered.connect(self.get_graph_fontsize)

        ''' update sum  '''
        self.sum_scaler_changed()
        logging.info('LiveDataPlottingUi opened ... ')

    def show_progress(self, show=None):
        if self.scan_prog_ui is None and self.subscribe_as_live_plot:
            logging.debug('self.scan_prog_ui is None')
            self.show_progress_window()
        if show is not None:
            self.dockWidget.setVisible(show)
        else:  # just toggle
            self.dockWidget.setVisible(not self.dockWidget.isVisible())

    '''setting up the plots (no data etc. written) '''

    def add_sum_plot(self):
        """ sum plot on the sum tab """
        self.sum_wid, self.sum_plt_itm = Pg.create_x_y_widget()
        self.sum_plot_layout = QtWidgets.QVBoxLayout()
        self.sum_plot_layout.addWidget(self.sum_wid)
        self.widget_inner_sum_plot.setLayout(self.sum_plot_layout)

    def add_time_resolved_plot(self):
        """ all plots on the time resolved tab -> time_resolved, time_projection, voltage_projection, sum """
        self.tres_v_layout = QtWidgets.QVBoxLayout()
        self.tres_widg, self.tres_plt_item = Pg.create_image_view()
        self.tres_roi = Pg.create_roi([0, 0], [1, 1])
        self.tres_roi.sigRegionChangeFinished.connect(self.rect_select_gates)
        self.tres_plt_item.addItem(self.tres_roi)
        self.tres_v_layout.addWidget(self.tres_widg)
        self.widget_tres.setLayout(self.tres_v_layout)
        self.sum_proj_wid, self.sum_proj_plt_itm = Pg.create_x_y_widget(do_not_show_label=['top'], y_label='sum')
        self.sum_proj_plt_itm.showAxis('right')
        self.v_proj_view_box = Pg.create_viewbox()
        self.sum_proj_plt_itm.scene().addItem(self.v_proj_view_box)
        self.sum_proj_plt_itm.getAxis('right').linkToView(self.v_proj_view_box)
        self.v_proj_view_box.setXLink(self.tres_widg.view)
        self.sum_proj_plt_itm.getAxis('right').setLabel('cts', color='k')
        pen = Pg.pg.mkPen(color='#0000ff', width=1)  # make the sum label and tick blue
        self.sum_proj_plt_itm.getAxis('left').setPen(pen)
        self.updateViews()
        self.sum_proj_plt_itm.vb.sigResized.connect(self.updateViews)

        self.t_proj_wid, self.t_proj_plt_itm = Pg.create_x_y_widget(do_not_show_label=['left', 'bottom'],
                                                                    y_label='time / µs', x_label='cts')
        self.v_proj_layout = QtWidgets.QVBoxLayout()
        self.t_proj_layout = QtWidgets.QVBoxLayout()
        self.sum_proj_plt_itm.setXLink(self.tres_widg.view)
        self.t_proj_plt_itm.setYLink(self.tres_widg.view)
        self.v_proj_layout.addWidget(self.sum_proj_wid)
        self.t_proj_layout.addWidget(self.t_proj_wid)
        self.widget_proj_v.setLayout(self.v_proj_layout)
        self.widget_proj_t.setLayout(self.t_proj_layout)
        self.pushButton_save_after_scan.clicked.connect(self.save)
        max_rate = 60
        self.t_res_mouse_proxy = Pg.create_proxy(signal=self.tres_plt_item.scene().sigMouseMoved,
                                                 slot=functools.partial(self.mouse_moved, self.tres_plt_item.vb, True),
                                                 rate_limit=max_rate)
        self.t_proj_mouse_proxy = Pg.create_proxy(signal=self.t_proj_plt_itm.scene().sigMouseMoved,
                                                  slot=functools.partial(self.mouse_moved, self.t_proj_plt_itm.vb,
                                                                         True),
                                                  rate_limit=max_rate)
        self.sum_proj_mouse_proxy = Pg.create_proxy(signal=self.sum_proj_plt_itm.scene().sigMouseMoved,
                                                    slot=functools.partial(self.mouse_moved, self.sum_proj_plt_itm.vb,
                                                                           True),
                                                    rate_limit=max_rate)

    def add_all_pmt_plot(self):
        """
        add a plot for each scaler on the tab 'all pmts'.

        Can only be called as soon as spec_data is available!!

        keys in self.all_pmts_widg_plt_item_list:
        'widget', 'proxy', 'vertLine', 'indList', 'pltDataItem', 'name', 'pltItem', 'fitLine'
        """
        max_rate = 60
        self.all_pmts_plot_layout = QtWidgets.QVBoxLayout()
        self.all_pmts_widg_plt_item_list = Pg.create_plot_for_all_sc(
            self.all_pmts_plot_layout, self.spec_data.active_pmt_list[self.all_pmts_sel_tr],
            self.mouse_moved, max_rate, plot_sum=self.spec_data.seq_type != 'kepco',
            inf_line=self.subscribe_as_live_plot
        )
        self.widget_all_pmts_plot.setLayout(self.all_pmts_plot_layout)

    def mouse_moved(self, viewbox, trs, evt):
        point = viewbox.mapSceneToView(evt[0])
        self.print_point(trs, point)

    def print_point(self, trs, point):
        if trs:
            self.label_x_coord.setText(str(round(point.x(), 3)))
            self.label_y_coord.setText(str(round(point.y(), 3)))
        else:  # plot it in the all pmts tab
            self.label_x_coord_all_pmts.setText(str(round(point.x(), 3)))
            self.label_y_coor_all_pmts.setText(str(round(point.y(), 3)))

    def updateViews(self):
        """ update the view for the overlayed plot of sum and current scaler """
        self.v_proj_view_box.setGeometry(self.sum_proj_plt_itm.vb.sceneBoundingRect())
        self.v_proj_view_box.linkedViewChanged(self.sum_proj_plt_itm.vb, self.v_proj_view_box.XAxis)

    def setup_new_track(self, rcv_tpl):
        """
        setup a new track -> set the indices for track and scaler
        """
        logging.info('livbeplot window received new track with %s %s ' % rcv_tpl)
        self.tres_sel_tr_ind, self.tres_sel_tr_name = rcv_tpl[0]
        if self.subscribe_as_live_plot:
            logging.info(
                'liveplot window received new track with updating index in comboBox_all_pmts_sel_tr to %s'
                % self.tres_sel_tr_ind)
            self.comboBox_all_pmts_sel_tr.setCurrentIndex(self.tres_sel_tr_ind)
        self.tres_sel_sc_ind, self.tres_sel_sc_name = rcv_tpl[1]
        self.new_track_no_data_yet = True
        # need to reset stuff here if number of steps have changed.

    ''' plot font size change etc. '''

    def get_graph_fontsize(self):
        try:
            dial = QtWidgets.QInputDialog(self)
            font_size_int, ok = QtWidgets.QInputDialog.getInt(dial, 'set the font size of the graphs',
                                                              'font size (pt)', self.graph_font_size, 0, 80)
            logging.info('liveplotterui: font size is: %s' % font_size_int)
            if ok:
                self.change_font_size_all_graphs(font_size_int)
                self.graph_font_size = font_size_int
        except Exception as e:
            logging.error('liveplotterui, error while getting font size: %s' % e, exc_info=True)

    def change_font_size_all_graphs(self, font_size):
        plots = [
            self.sum_plt_itm, self.sum_proj_plt_itm, self.t_proj_plt_itm, self.tres_plt_item
        ]
        plots += [each['pltItem'] for each in self.all_pmts_widg_plt_item_list]
        font = QtGui.QFont()
        font.setPixelSize(font_size)
        for plot in plots:
            for ax in ['left', 'bottom', 'top', 'right']:
                axis = plot.getAxis(ax)
                axis.tickFont = font
                axis.setStyle(tickTextOffset=int(font_size - 5))
                axis.label.setFont(font)

        axis = self.tres_widg.getHistogramWidget().axis
        axis.tickFont = font
        axis.setStyle(tickTextOffset=int(font_size - 5))
        axis.label.setFont(font)
        self.label_x_coord.setFont(font)
        self.label_2.setFont(font)
        self.label_5.setFont(font)
        self.label_y_coord.setFont(font)

        self.label_x_coord_all_pmts.setFont(font)
        self.label_y_coor_all_pmts.setFont(font)
        self.label_7.setFont(font)
        self.label_8.setFont(font)

    def raise_graph_fontsize(self, up_or_down_bool):
        if up_or_down_bool:  # increase
            self.graph_font_size += 1
        else:  # decrease
            self.graph_font_size -= 1
        self.change_font_size_all_graphs(self.graph_font_size)

    def convert_xaxis_for_step_mode(self, x_axis):
        x_axis_step = np.mean(np.ediff1d(x_axis))
        x_axis = np.append(x_axis, [x_axis[-1] + x_axis_step])
        x_axis += -0.5 * abs(x_axis_step)
        return x_axis

    ''' receive and plot new incoming data '''

    def new_data(self, spec_data):
        """
        call this to pass a new dataset to the gui.
        """
        try:
            st = datetime.now()
            valid_data = False
            self.spec_data = deepcopy(spec_data)

            self.update_all_plots(self.spec_data)
            if self.spec_data.seq_type in self.trs_names_list:
                self.spinBox.blockSignals(True)
                self.spinBox.setValue(self.spec_data.softBinWidth_ns[self.tres_sel_tr_ind])
                self.spinBox.blockSignals(False)
                self.update_gates_list()
            valid_data = True
            if valid_data and self.new_track_no_data_yet:  # this means it is first call
                # refresh the line edit by calling this here:
                self.sum_scaler_changed(self.comboBox_sum_all_pmts.currentIndex())

                self.new_track_no_data_yet = False
            self.last_gr_update_done_time = datetime.now()
            elapsed = self.last_gr_update_done_time - st
            # logging.debug('done updating plot, plotting took %.2f ms' % (elapsed.microseconds / 1000))
        except Exception as e:
            logging.error('error in liveplotterui while receiving new data: %s ' % e, exc_info=True)

    ''' updating the plots from specdata '''

    def update_all_plots(self, spec_data):
        """ wrapper to update all plots """
        try:
            self.update_sum_plot(spec_data)
            if spec_data.seq_type in self.trs_names_list:
                self.update_tres_plot(spec_data)
                self.update_projections(spec_data)
            self.update_all_pmts_plot(spec_data)
        except Exception as e:
            logging.error('error in updating plots: ' + str(e), exc_info=True)

    def update_sum_plot(self, spec_data):
        """ update the sum plot and store the values in self.sum_x, self.sum_y, self.sum_err"""
        if self.sum_scaler is not None:
            self.sum_x, self.sum_y, self.sum_err = spec_data.getArithSpec(self.sum_scaler, self.sum_track)
            if self.sum_plt_data is None:
                self.sum_plt_data = self.sum_plt_itm.plot(
                    self.convert_xaxis_for_step_mode(self.sum_x), self.sum_y, stepMode=True, pen='k')
                if self.subscribe_as_live_plot:
                    self.sum_current_step_line = Pg.create_infinite_line(self.spec_data.x[self.tres_sel_tr_ind][0],
                                                                         pen='r')
                    self.sum_plt_itm.addItem(self.sum_current_step_line, ignoreBounds=True)
            else:
                self.sum_plt_data.setData(self.convert_xaxis_for_step_mode(self.sum_x), self.sum_y, stepMode=True)
            self.sum_plt_itm.setLabel('bottom', spec_data.x_units.value)

    def update_tres_plot(self, spec_data):
        """ update the time resolved plot including the roi """
        try:

            gates = self.spec_data.softw_gates[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
            # x_range = (float(np.min(spec_data.x[self.tres_sel_tr_ind])), np.max(spec_data.x[self.tres_sel_tr_ind]))
            x_range = (float(spec_data.x[self.tres_sel_tr_ind][0]), spec_data.x[self.tres_sel_tr_ind][-1])
            x_scale = np.mean(np.ediff1d(spec_data.x[self.tres_sel_tr_ind]))
            y_range = (np.min(spec_data.t[self.tres_sel_tr_ind]), np.max(spec_data.t[self.tres_sel_tr_ind]))
            y_scale = np.mean(np.ediff1d(spec_data.t[self.tres_sel_tr_ind]))
            self.tres_widg.setImage(spec_data.time_res[self.tres_sel_tr_ind][self.tres_sel_sc_ind],
                                    pos=[x_range[0] - 0.5 * x_scale,  # changed from abs(...)
                                         y_range[0] - abs(0.5 * y_scale)],
                                    scale=[x_scale, y_scale],
                                    autoRange=False)
            self.tres_plt_item.setLabel('top', spec_data.x_units.value)
            if self.new_track_no_data_yet or self.setup_range_please:  # set view range in first call
                logging.debug('setting x_range to: %s and y_range to: %s' % (str(x_range), str(y_range)))
                self.tres_plt_item.setAspectLocked(False)
                self.tres_plt_item.setRange(xRange=x_range, yRange=y_range, padding=0.05)
                self.setup_range_please = False
            self.tres_roi.setPos((gates[0], gates[2]), finish=False)
            self.tres_roi.setSize((abs(gates[0] - gates[1]), abs(gates[2] - gates[3])), finish=False)
        except Exception as e:
            logging.error('error, while plotting time resolved, this happened: %s ' % e, exc_info=True)

    def rect_select_gates(self, evt):
        """
        is called via left/rigth click & release events, connection see in start()
        will pass the coordinates of the selected area to self.update_gate_ind()
        """
        try:
            x_min, t_min = self.tres_roi.pos()
            x_max, t_max = self.tres_roi.size()
            x_max += x_min
            t_max += t_min
            gates_list = [x_min, x_max, t_min, t_max]
            # print(datetime.datetime.now().strftime('%H:%M:%S'), ' gate select yields:', gates_list)
            self.spec_data.softw_gates[self.tres_sel_tr_ind][self.tres_sel_sc_ind] = gates_list
            self.gate_data(self.spec_data)
            # else:
            #     print('this very event already has happened')
        except Exception as e:
            logging.error('error in LiveDataPlotting, while setting the gates this happened: %s ' % e, exc_info=True)
        pass

    def update_projections(self, spec_data):
        """
        update the projections, if no plot has been done yet, create plotdata items for every plot
        """
        try:
            if self.sum_scaler is not None:
                t_proj_x = spec_data.t_proj[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
                t_proj_y = spec_data.t[self.tres_sel_tr_ind]
                v_proj_x = spec_data.x[self.tres_sel_tr_ind]
                v_proj_y = spec_data.cts[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
                gates = self.spec_data.softw_gates[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
                sum_x, sum_y, sum_err = spec_data.getArithSpec(self.sum_scaler, self.tres_sel_tr_ind)

                if self.t_proj_plt is None:
                    self.t_proj_plt = self.t_proj_plt_itm.plot(t_proj_x, t_proj_y, pen='k')
                    self.sum_proj_plt_data = self.sum_proj_plt_itm.plot(
                        self.convert_xaxis_for_step_mode(sum_x), sum_y, pen='b', stepMode=True)
                    self.v_proj_plt = Pg.create_plot_data_item(
                        self.convert_xaxis_for_step_mode(v_proj_x), v_proj_y, pen='k', stepMode=True)
                    self.v_proj_view_box.addItem(self.v_proj_plt)

                    if self.subscribe_as_live_plot:
                        self.current_step_line = Pg.create_infinite_line(self.spec_data.x[self.tres_sel_tr_ind][0],
                                                                         pen='r')
                        self.sum_proj_plt_itm.addItem(self.current_step_line, ignoreBounds=True)

                    self.v_min_line = Pg.create_infinite_line(gates[0])
                    self.v_max_line = Pg.create_infinite_line(gates[1])
                    self.t_min_line = Pg.create_infinite_line(gates[2], angle=0)
                    self.t_max_line = Pg.create_infinite_line(gates[3], angle=0)

                    self.sum_proj_plt_itm.addItem(self.v_min_line)
                    self.sum_proj_plt_itm.addItem(self.v_max_line)
                    self.t_proj_plt_itm.addItem(self.t_min_line)
                    self.t_proj_plt_itm.addItem(self.t_max_line)
                else:
                    self.t_proj_plt.setData(t_proj_x, t_proj_y)
                    self.sum_proj_plt_data.setData(self.convert_xaxis_for_step_mode(sum_x), sum_y, stepMode=True)
                    self.v_proj_plt.setData(self.convert_xaxis_for_step_mode(v_proj_x), v_proj_y, stepMode=True)
                    self.v_min_line.setValue(gates[0])
                    self.v_max_line.setValue(gates[1])
                    self.t_min_line.setValue(gates[2])
                    self.t_max_line.setValue(gates[3])
                self.sum_proj_plt_itm.setLabel('bottom', spec_data.x_units.value)

        except Exception as e:
            logging.error('error, while plotting projection, this happened: %s' % e, exc_info=True)

    def update_all_pmts_plot(self, spec_data, autorange_pls=True):
        if self.all_pmts_widg_plt_item_list is None:
            if spec_data.seq_type not in self.trs_names_list:
                self.tabWidget.setCurrentIndex(2)
            self.comboBox_all_pmts_sel_tr.blockSignals(True)
            tr_list = deepcopy(spec_data.track_names)
            tr_list.append('all')
            self.comboBox_all_pmts_sel_tr.addItems(tr_list)
            # self.cb_all_pmts_sel_tr_changed(self.comboBox_all_pmts_sel_tr.currentText())
            if self.sum_sc_tr_external is not None:
                self.set_tr_sel_by_index(self.sum_track)
            self.comboBox_all_pmts_sel_tr.blockSignals(False)

            self.add_all_pmt_plot()
        Pg.plot_all_sc(self.all_pmts_widg_plt_item_list, spec_data, self.all_pmts_sel_tr, stepMode=True)
        if autorange_pls:
            [each['pltItem'].autoRange() for each in self.all_pmts_widg_plt_item_list]
            self.all_pmts_widg_plt_item_list[-1]['pltItem'].setLabel('bottom', spec_data.x_units.value)
            self.all_pmts_widg_plt_item_list[0]['pltItem'].setLabel('top', spec_data.x_units.value)

    ''' buttons, comboboxes and listwidgets: '''

    def sum_scaler_changed(self, index=None):
        """
        this will set the self.sum_scaler list to the values set in the gui.
        :param index: int, index of the element in the combobox
        """
        logging.info('liveplotterui: sum_scaler_changed was called with index: %s ' % index)
        if index is None:
            index = self.comboBox_select_sum_for_pmts.currentIndex() if self.sum_sc_tr_external is None else 1
        if index == 0:  # add all
            if self.spec_data is not None:
                if self.spec_data.seq_type != 'kepco':
                    self.sum_scaler = self.spec_data.active_pmt_list[0]  # should be the same for all tracks
                else:
                    self.sum_scaler = [0]
                self.sum_track = -1
                self.lineEdit_arith_scaler_input.setText(str(self.sum_scaler))
                self.lineEdit_sum_all_pmts.setText(str(self.sum_scaler))
            else:
                logging.info('liveplotterui: but specdata is None, so line edit is not set.')
            self.lineEdit_arith_scaler_input.setDisabled(True)
            self.lineEdit_sum_all_pmts.setDisabled(True)
        elif index == 1:  # manual
            # self.sum_scaler = self.valid_scaler_input
            self.lineEdit_arith_scaler_input.setDisabled(False)
            self.lineEdit_sum_all_pmts.setDisabled(False)

        # synchronize both comboboxes
        self.comboBox_sum_all_pmts.setCurrentIndex(index)
        self.comboBox_select_sum_for_pmts.setCurrentIndex(index)

    def sum_scaler_lineedit_changed(self, text):
        """
        this will check if the input text in the line edit will result is a list
         and only contain valid scaler entries.
        :param text:str, in the form of type [+i, -j, +k], resulting in s[i]-s[j]+s[k]
        :return:
        """
        try:
            curs_pos_sum = self.lineEdit_arith_scaler_input.cursorPosition()
            curs_pos_all_pmts = self.lineEdit_sum_all_pmts.cursorPosition()
            hopefully_list = ast.literal_eval(text)
            if isinstance(hopefully_list, list):
                isinteger = len(hopefully_list) > 0
                for scaler in hopefully_list:
                    isinteger = isinteger and isinstance(scaler, int) and abs(scaler) < self.spec_data.nrScalers[0]
                if isinteger:
                    self.sum_scaler = hopefully_list
                    self.label_arith_scaler_set.setText(str(hopefully_list))
                    self.update_sum_plot(self.spec_data)
                    if self.spec_data.seq_type in self.trs_names_list:
                        self.update_projections(self.spec_data)
                    if self.all_pmts_widg_plt_item_list is not None:
                        self.all_pmts_widg_plt_item_list[-1]['indList'] = hopefully_list
                        self.update_all_pmts_plot(self.spec_data)
                    self.lineEdit_sum_all_pmts.setText(text)
                    self.lineEdit_arith_scaler_input.setText(text)
                    if curs_pos_sum:
                        self.lineEdit_arith_scaler_input.setCursorPosition(curs_pos_sum)
                    if curs_pos_all_pmts:
                        self.lineEdit_sum_all_pmts.setCursorPosition(curs_pos_all_pmts)
        except Exception as e:
            logging.error('error on changing line edit of summed scalers in liveplotterui: %s' % e, exc_info=True)

    def cb_all_pmts_sel_tr_changed(self, text):
        """ handle changes in the combobox in the all pmts tab """
        if text == 'all':
            tr_ind = -1
        else:
            tr_ind = self.comboBox_all_pmts_sel_tr.currentIndex()
        self.all_pmts_sel_tr = tr_ind
        if self.spec_data is not None and self.all_pmts_widg_plt_item_list is not None:
            self.update_all_pmts_plot(self.spec_data, autorange_pls=True)

    def set_tr_sel_by_index(self, tr_ind):
        new_ind = tr_ind != self.all_pmts_sel_tr
        if tr_ind == -1:
            tr_ind = self.comboBox_all_pmts_sel_tr.count() - 1
        logging.debug('setting current index of comboBox_all_pmts_sel_tr to: %s' % tr_ind)
        self.comboBox_all_pmts_sel_tr.blockSignals(True)
        self.comboBox_all_pmts_sel_tr.setCurrentIndex(tr_ind)
        self.comboBox_all_pmts_sel_tr.blockSignals(False)
        if new_ind:
            self.cb_all_pmts_sel_tr_changed(self.comboBox_all_pmts_sel_tr.currentText())

    ''' gating: '''

    def gate_data(self, spec_data, plot_bool=False):
        rebin_track = -1 if self.checkBox.isChecked() else self.tres_sel_tr_ind
        if spec_data is None:
            spec_data = self.spec_data
        self.new_gate_or_soft_bin_width.emit(
            spec_data.softw_gates, rebin_track, spec_data.softBinWidth_ns, plot_bool)

    ''' table operations: '''

    def handle_gate_table_change(self, item):
        # print('item was changed: ', item.row(), item.column(), item.text())
        gate_columns = range(2, 6)
        if item.column() in gate_columns:  # this means a gate value was changed.
            sel_items = self.tableWidget_gates.selectedItems()
            try:
                new_val = float(item.text())
            except Exception as e:
                logging.error(
                    'error, could not convert value to float: %s, error is: %s' % (item.text(), e), exc_info=True)
                item.setText('0.0')
                return None
            if len(sel_items):
                self.tableWidget_gates.blockSignals(True)
                for each in sel_items:
                    sel_tr = int(self.tableWidget_gates.item(each.row(), 0).text()[5:])
                    sel_sc = int(self.tableWidget_gates.item(each.row(), 1).text())
                    gate_ind = each.column() - 2
                    self.spec_data.softw_gates[sel_tr][sel_sc][gate_ind] = new_val
                    self.gate_data(self.spec_data)
                self.tableWidget_gates.blockSignals(False)

    def handle_item_clicked(self, item):
        """ this will select which track and scaler one is viewing. """
        if item.column() == self.tableWidget_gates.columnCount() - 1:
            if item.checkState() == QtCore.Qt.Checked:
                currently_selected = self.find_one_scaler_track(self.tres_sel_tr_name, self.tres_sel_sc_ind)
                if currently_selected.row() != item.row():
                    curr_checkb_item = self.tableWidget_gates.item(currently_selected.row(), 6)
                    curr_checkb_item.setCheckState(QtCore.Qt.Unchecked)
                    self.tres_sel_tr_ind = int(self.tableWidget_gates.item(item.row(), 0).text()[5:])
                    self.tres_sel_tr_name = self.tableWidget_gates.item(item.row(), 0).text()
                    self.tres_sel_sc_ind = int(self.tableWidget_gates.item(item.row(), 1).text())
                    # print('new scaler, track: ', self.tres_sel_tr_ind, self.tres_sel_sc_ind)
                    self.rebin_data(self.spec_data.softBinWidth_ns[self.tres_sel_tr_ind])
                    self.update_tres_plot(self.spec_data)
                    self.update_projections(self.spec_data)
                    # else:
                    #     item.setCheckState(QtCore.Qt.Checked)
        currently_selected = self.find_one_scaler_track(self.tres_sel_tr_name, self.tres_sel_sc_ind)
        curr_checkb_item = self.tableWidget_gates.item(currently_selected.row(), 6)
        curr_checkb_item.setCheckState(QtCore.Qt.Checked)

    def select_scaler_tr(self, tr_name, sc_ind):
        for row in range(self.tableWidget_gates.rowCount()):
            curr_checkb_item = self.tableWidget_gates.item(row, 6)
            curr_checkb_item.setCheckState(QtCore.Qt.Unchecked)
        currently_selected = self.find_one_scaler_track(tr_name, sc_ind)
        curr_checkb_item = self.tableWidget_gates.item(currently_selected.row(), 6)
        curr_checkb_item.setCheckState(QtCore.Qt.Checked)

    def update_gates_list(self):
        """
        read all software gate entries from the specdata and fill it to the displaying table.
        all entries before will be deleted.
        """
        if self.tableWidget_gates.rowCount() == 0:
            self.tableWidget_gates.blockSignals(True)
            # self.tableWidget_gates.clear()
            for tr_ind, tr_name in enumerate(self.spec_data.track_names):
                if tr_name != 'all':
                    offset = self.tableWidget_gates.rowCount()
                    for pmt_ind, pmt_name in enumerate(self.spec_data.active_pmt_list[tr_ind]):
                        row_ind = pmt_ind + offset
                        self.tableWidget_gates.insertRow(row_ind)
                        tr_item = QtWidgets.QTableWidgetItem(tr_name)
                        tr_item.setFlags(QtCore.Qt.ItemIsSelectable)
                        self.tableWidget_gates.setItem(row_ind, 0, tr_item)
                        pmt_item = QtWidgets.QTableWidgetItem()
                        pmt_item.setData(QtCore.Qt.DisplayRole, pmt_name)
                        pmt_item.setFlags(QtCore.Qt.ItemIsSelectable)
                        self.tableWidget_gates.setItem(row_ind, 1, pmt_item)
                        checkbox_item = QtWidgets.QTableWidgetItem()
                        checkbox_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                        state = QtCore.Qt.Unchecked
                        if self.tres_sel_tr_name == tr_name and self.tres_sel_sc_ind == pmt_ind:
                            state = QtCore.Qt.Checked
                        checkbox_item.setCheckState(state)
                        self.tableWidget_gates.setItem(row_ind, self.tableWidget_gates.columnCount() - 1, checkbox_item)
                        for i, gate in enumerate(self.spec_data.softw_gates[tr_ind][pmt_ind]):
                            gate_item = QtWidgets.QTableWidgetItem()
                            gate_item.setData(QtCore.Qt.EditRole, gate)
                            self.tableWidget_gates.setItem(row_ind, 2 + i, gate_item)
                            # print('this was set: ', tr_name, pmt_ind)
            self.tableWidget_gates.blockSignals(False)
        else:
            self.select_scaler_tr(self.tres_sel_tr_name, self.tres_sel_sc_ind)
        self.write_all_gates_to_table(self.spec_data)

    def extract_one_gate_from_gui(self, tr, sc):
        item = self.find_one_scaler_track(tr, sc)
        gate_lis = []
        if item is not None:
            for i in range(2, self.tableWidget_gates.columnCount() - 1):
                gate_lis.append(float(self.tableWidget_gates.item(item.row(), i).text()))
        return gate_lis

    def extract_all_gates_from_gui(self):
        return deepcopy(self.spec_data.softw_gates)

    def insert_one_gate_to_gui(self, tr, sc, liste):
        item = self.find_one_scaler_track(tr, sc)
        if item is not None:
            row = item.row()
            for i in range(2, self.tableWidget_gates.columnCount() - 1):
                val = liste[i - 2]
                self.tableWidget_gates.item(row, i).setData(QtCore.Qt.EditRole, str(round(val, 2)))

    def write_all_gates_to_table(self, spec_data):
        self.tableWidget_gates.blockSignals(True)
        for tr_ind, tr_name in enumerate(spec_data.track_names):
            if tr_name != 'all':
                for sc in range(spec_data.nrScalers[tr_ind]):
                    gates = spec_data.softw_gates[tr_ind][sc]
                    self.insert_one_gate_to_gui(tr_name, sc, gates)
        self.tableWidget_gates.blockSignals(False)

    def find_one_scaler_track(self, tr, sc):
        """
        find the scaler of the given track and return the item
        :param tr: str, name of track
        :param sc: int, index of scaler
        :return: item
        """
        result = None
        tr_found = self.tableWidget_gates.findItems(tr, QtCore.Qt.MatchExactly)
        if any(tr_found):
            tr_row_lis = [item.row() for item in tr_found]
            pmt_found = self.tableWidget_gates.findItems(str(sc), QtCore.Qt.MatchExactly)
            for item in pmt_found:
                if item.row() in tr_row_lis and item.column() == 1:
                    result = item
        return result

    def reset_table(self):
        """
        remove all entries from the table.
        """
        logging.debug('lievplotUI resetting table entries')
        while self.tableWidget_gates.rowCount() > 0:
            self.tableWidget_gates.removeRow(0)
            # logging.debug('livePlotUi, row count after reset of table: ', self.tableWidget_gates.rowCount())

    ''' saving '''

    def save(self, pipedata_dict=None):
        im_path = self.full_file_path.split('.')[0] + '_' + str(self.tres_sel_tr_name) + '_' + str(
            self.tres_sel_sc_name) + '.png'
        test = self.grab()
        test.save(im_path)
        logging.info('livbeplotterui, saved image to: %s' % str(im_path))
        logging.info('liveplot emitting save signal now')
        self.save_request.emit()

    ''' closing '''

    # def closeEvent(self, event):
    #     if self.subscribe_as_live_plot:
    #         self.unsubscribe_from_main()

    ''' subscription to main '''

    def get_existing_callbacks_from_main(self):
        """ check wether existing callbacks are still around in the main adn then connect to those. """
        if Cfg._main_instance is not None and self.subscribe_as_live_plot:
            logging.info('TRSLivePlotWindowUi is connecting to existing callbacks in main')
            callbacks = Cfg._main_instance.gui_live_plot_subscribe()
            self.new_data_callback = callbacks[0]
            self.new_track_callback = callbacks[1]
            self.save_request = callbacks[2]
            self.new_gate_or_soft_bin_width = callbacks[3]
            self.fit_results_dict_callback = callbacks[4]
            self.progress_callback = callbacks[5]

    def subscribe_to_main(self):
        if self.subscribe_as_live_plot:
            self.progress_callback.connect(self.handle_progress_dict)
            # overwrite gui callbacks with callbacks from main
            self.get_existing_callbacks_from_main()
        self.save_callback.connect(self.save)
        self.new_track_callback.connect(self.setup_new_track)
        self.new_data_callback.connect(self.new_data)
        self.fit_results_dict_callback.connect(self.rcvd_fit_res_dict)

    ''' rebinning '''

    def rebin_data(self, rebin_factor_ns=None):
        start = datetime.now()
        if rebin_factor_ns is None:
            if self.active_initial_scan_dict is not None:
                rebin_factor_ns = self.spec_data.softBinWidth_ns[self.tres_sel_tr_ind]
        rebin_factor_ns = rebin_factor_ns // 10 * 10
        self.spec_data.softBinWidth_ns[self.tres_sel_tr_ind] = rebin_factor_ns
        self.spinBox.blockSignals(True)
        self.spinBox.setValue(rebin_factor_ns)
        self.spinBox.blockSignals(False)
        logging.debug('rebinning data to bins of  %s' % rebin_factor_ns)
        rebin_track = -1 if self.checkBox.isChecked() else self.tres_sel_tr_ind
        self.new_gate_or_soft_bin_width.emit(
            self.extract_all_gates_from_gui(), rebin_track, self.spec_data.softBinWidth_ns, False)
        stop = datetime.now()
        dif = stop - start
        # print('rebinning took: %s' % dif)

    def apply_rebin_to_all_checkbox_changed(self, state):
        if state == 2:  # the checkbox is checked
            # only rebin if true, otherwise anyhow individual gates are lost before
            self.rebin_data(self.spinBox.value())

    ''' progress related '''

    def handle_progress_dict(self, progress_dict_from_main):
        """
        will be emitted from the main, each time the progress is updated
        :param progress_dict_from_main: dict, keys:
        ['activeIso', 'overallProgr', 'timeleft', 'activeTrack', 'totalTracks',
        'trackProgr', 'activeScan', 'totalScans', 'activeStep',
        'totalSteps', 'trackName', 'activeFile']
        """
        self.active_iso = progress_dict_from_main['activeIso']
        if not self.new_track_no_data_yet:
            act_step_ind = max(progress_dict_from_main['actStepIndex'], 0)
            act_tr_ind = progress_dict_from_main['activeTrack'] - 1
            self.update_step_indication_lines(act_step_ind, act_tr_ind)
        if self.active_file != progress_dict_from_main['activeFile']:
            self.active_file = progress_dict_from_main['activeFile']
            self.setWindowTitle('plot:  %s' % self.active_file)
            self.dockWidget.setWindowTitle('progress: %s' % self.active_file)

        self.active_initial_scan_dict = Cfg._main_instance.scan_pars[self.active_iso]
        self.active_track_name = progress_dict_from_main['trackName']
        self.overall_scan_progress = progress_dict_from_main['overallProgr']

    def update_step_indication_lines(self, act_step, act_tr):
        try:
            if self.all_pmts_widg_plt_item_list is not None:
                val = self.spec_data.x[act_tr][act_step]
                print('active track is: %s and active step is: %s val is: %s ' % (act_tr, act_step, val))
                if self.current_step_line is not None:
                    self.current_step_line.setValue(val)
                if self.sum_current_step_line is not None:
                    self.sum_current_step_line.setValue(val)
                if self.subscribe_as_live_plot:
                    [each['vertLine'].setValue(val) for each in self.all_pmts_widg_plt_item_list]
        except Exception as e:
            logging.error('error in liveplotterui while updating step indication line: %s' % e)

    def reset(self):
        """
        reset all stored data etc. in order to prepare for a new isotope or so.
        :return:
        """
        self.spec_data = None
        self.setWindowTitle('plot: ...  loading  ... ')
        self.scan_prog_ui.reset()
        self.reset_table()
        self.reset_sum_plots()
        self.reset_all_pmt_plots()
        self.reset_t_res_plot()

    def show_progress_window(self):
        try:
            if self.subscribe_as_live_plot:
                if self.scan_prog_ui is None:
                    logging.info('creating scan progress window.')

                    self.scan_prog_ui = ScanProgressUi(self)
                    self.scan_prog_ui.destroyed.connect(self.scan_progress_ui_destroyed)
                    self.scan_prog_layout = QtWidgets.QVBoxLayout()
                    self.scan_prog_layout.addWidget(self.scan_prog_ui)
                    self.widget_progress.setLayout(self.scan_prog_layout)
            else:
                self.show_progress(False)
        except Exception as e:
            logging.error('error while adding scanprog: %s' % e, exc_info=True)

    def scan_progress_ui_destroyed(self):
        logging.info('scan progress window was destroyed.')
        self.scan_prog_ui = None

    ''' fit related '''

    def rcvd_fit_res_dict(self, fit_res_dict):
        # print('rcvd fit result dict: %s' % fit_res_dict['result'])
        # print('index is: %s' % fit_res_dict['index'])
        x, y = fit_res_dict['plotData']
        plot_dict = self.all_pmts_widg_plt_item_list[fit_res_dict['index']]
        plt_item = plot_dict['pltItem']
        plot_dict['fitLine'] = plt_item.plot(x, y, pen='r')
        # print(fit_res_dict['result'])
        display_text = ''
        for i, fit_res_tuple in enumerate(fit_res_dict['result']):
            for key, val in fit_res_tuple[1].items():
                display_text += '%s: %g +/- %g (fixed: %s) \n' % (key, val[0], val[1], val[2])
            display_text += '\n'
        anchor = plt_item.getViewBox().viewRange()
        anchor = (anchor[0][0], anchor[1][1])
        txt = Pg.create_text_item(display_text, color='r')
        txt.setPos(*anchor)
        plt_item.addItem(txt)
        plot_dict['fitText'] = txt

    def reset_all_pmt_plots(self):
        """
        remove everything in the all plots tab
        :return:
        """
        self.all_pmts_widg_plt_item_list = None  # therefore it will be created again
        #  when update_all_pmts_plot is called
        self.comboBox_all_pmts_sel_tr.clear()
        self.all_pmts_sel_tr = 0
        try:
            if isinstance(self.all_pmts_plot_layout, QtWidgets.QVBoxLayout):
                QtWidgets.QWidget().setLayout(self.all_pmts_plot_layout)
        except Exception as e:
            logging.error('error: while resetting the all_pmt_plot tab/plot: %s' % e, exc_info=True)
        # will be called within update_plots() when first data arrives
        self.comboBox_sum_all_pmts.currentIndexChanged.emit(self.comboBox_sum_all_pmts.currentIndex())

    def reset_sum_plots(self):
        """
        reset the sum plot is not really necessary, because it will always look the same more or less.
        """
        QtWidgets.QWidget().setLayout(self.sum_plot_layout)
        self.sum_plt_data = None
        self.sum_scaler = None
        self.sum_track = None
        self.add_sum_plot()

    def reset_t_res_plot(self):
        """
        since their is always just one, set all data to 0 and its ok.
        """
        QtWidgets.QWidget().setLayout(self.tres_v_layout)
        QtWidgets.QWidget().setLayout(self.v_proj_layout)
        QtWidgets.QWidget().setLayout(self.t_proj_layout)
        self.t_proj_plt = None  # in order to trigger for new data
        self.current_step_line = None
        self.sum_current_step_line = None
        self.add_time_resolved_plot()

# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     ui = TRSLivePlotWindowUi()
#     ui.show()
#     app.exec()
