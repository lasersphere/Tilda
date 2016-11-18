"""
Created on 

@author: simkaufm

Module Description:
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
    new_gate_or_soft_bin_width = QtCore.pyqtSignal(list, int, list)

    # save request
    save_request = QtCore.pyqtSignal()


    # progress dict coming from the main
    progress_callback = QtCore.pyqtSignal(dict)

    def __init__(self, full_file_path='', parent=None, subscribe_as_live_plot=True):
        super(TRSLivePlotWindowUi, self).__init__()

        self.t_proj_plt = None
        self.last_event = None
        self.pipedata_dict = None  # dict, containing all infos from the pipeline, will be passed to gui when save request is called from pipeline
        self.active_track_name = None  # str, name of the active track
        self.active_initial_scan_dict = None  # scan dict which is stored under the active iso name in the main
        self.active_file = None  # str, name of active file
        self.active_iso = None  # str, name of active iso in main
        self.setupUi(self)
        self.show()
        self.tabWidget.setCurrentIndex(1)  # time resolved
        self.setWindowTitle('plot win:     ' + full_file_path)
        self.dockWidget.setWindowTitle('progress: %s' % self.active_file)

        self.full_file_path = full_file_path

        self.parent = parent
        self.sum_plt_data = None
        self.trs_names_list = ['trs', 'trsdummy', 'tipa']

        self.tres_image = None
        self.t_proj_plt_itm = None
        self.tres_plt_item = None
        self.spec_data = None  # spec_data to work on.
        self.new_track_no_data_yet = False  # set this to true when new track is setup
        ''' connect callbacks: '''
        # bundle callbacks:
        self.subscribe_as_live_plot = subscribe_as_live_plot
        self.callbacks = (self.new_data_callback, self.new_track_callback,
                          self.save_request, self.new_gate_or_soft_bin_width,
                          )
        self.subscribe_to_main()

        ''' sum related '''
        self.add_sum_plot()
        self.sum_x, self.sum_y, self.sum_err = None, None, None  # storage of the sum plotting values

        self.sum_scaler = [0]  # list of scalers to add
        self.sum_track = -1  # int, for selecting the track which will be added. -1 for all

        self.current_step_line = None  # line to display which step is active.

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

        try:
            if self.subscribe_as_live_plot:
                self.scan_prog_ui = ScanProgressUi(self.parent)
                self.scan_prog_layout = QtWidgets.QVBoxLayout()
                self.scan_prog_layout.addWidget(self.scan_prog_ui)
                self.widget_progress.setLayout(self.scan_prog_layout)
        except Exception as e:
            print('error while adding scanprog: %s' % e)

        self.show_progress_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_P), self)

        self.show_progress_shortcut.activated.connect(self.show_progress)
        self.actionProgress.triggered.connect(self.show_progress)

    def show_progress(self):
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
        self.v_proj_view_box.setXLink(self.sum_proj_plt_itm)
        self.sum_proj_plt_itm.getAxis('right').setLabel('cts', color='k')
        self.sum_proj_plt_itm.getAxis('left').setLabel('sum', color='#0000ff')
        self.updateViews()
        self.sum_proj_plt_itm.vb.sigResized.connect(self.updateViews)

        self.t_proj_wid, self.t_proj_plt_itm = Pg.create_x_y_widget(do_not_show_label=['left', 'bottom'],
                                                                    y_label='time [µs]', x_label='cts')
        self.v_proj_layout = QtWidgets.QVBoxLayout()
        self.t_proj_layout = QtWidgets.QVBoxLayout()
        self.sum_proj_plt_itm.setXLink(self.tres_plt_item)
        self.t_proj_plt_itm.setYLink(self.tres_plt_item)
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
                                                  slot=functools.partial(self.mouse_moved, self.t_proj_plt_itm.vb, True),
                                                  rate_limit=max_rate)
        self.sum_proj_mouse_proxy = Pg.create_proxy(signal=self.sum_proj_plt_itm.scene().sigMouseMoved,
                                                    slot=functools.partial(self.mouse_moved, self.sum_proj_plt_itm.vb, True),
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
            self.mouse_moved, max_rate, plot_sum=self.spec_data.seq_type != 'kepco'
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
        """ update teh view for the overlayed plot of sum and current scaler """
        self.v_proj_view_box.setGeometry(self.sum_proj_plt_itm.vb.sceneBoundingRect())
        self.v_proj_view_box.linkedViewChanged(self.sum_proj_plt_itm.vb, self.v_proj_view_box.XAxis)

    def setup_new_track(self, rcv_tpl):
        """
        setup a new track -> set the indices for track and scaler
        """
        print('receivec new track with %s %s ' % rcv_tpl)
        self.tres_sel_tr_ind, self.tres_sel_tr_name = rcv_tpl[0]
        self.comboBox_all_pmts_sel_tr.setCurrentIndex(self.tres_sel_tr_ind)
        self.tres_sel_sc_ind, self.tres_sel_sc_name = rcv_tpl[1]
        self.new_track_no_data_yet = True
        # need to reset stuff here if number of steps have changed.

    ''' receive and plot new incoming data '''

    def new_data(self, spec_data):
        """
        call this to pass a new dataset to the gui.
        """
        try:
            valid_data = False
            self.spec_data = deepcopy(spec_data)
            self.spinBox.blockSignals(True)
            self.spinBox.setValue(self.spec_data.softBinWidth_ns[self.tres_sel_tr_ind])
            self.spinBox.blockSignals(False)

            self.update_all_plots(self.spec_data)
            if self.spec_data.seq_type in self.trs_names_list:
                self.update_gates_list()
            valid_data = True
            if valid_data and self.new_track_no_data_yet:  # this means it is first call
                # refresh the line edit by calling this here:
                self.sum_scaler_changed(self.comboBox_sum_all_pmts.currentIndex())

                self.new_track_no_data_yet = False
        except Exception as e:
            print('error in liveplotterui while receiving new data: ', e)

    ''' updating the plots from specdata '''

    def update_all_plots(self, spec_data):
        """ wrapper to update all plots """
        self.update_sum_plot(spec_data)
        if spec_data.seq_type in self.trs_names_list:
            self.update_tres_plot(spec_data)
            self.update_projections(spec_data)
        self.update_all_pmts_plot(spec_data)

    def update_sum_plot(self, spec_data):
        """ update the sum plot and store the values in self.sum_x, self.sum_y, self.sum_err"""
        self.sum_x, self.sum_y, self.sum_err = spec_data.getArithSpec(self.sum_scaler, self.sum_track)
        if self.sum_plt_data is None:
            self.sum_plt_data = self.sum_plt_itm.plot(self.sum_x, self.sum_y, pen='k')
        else:
            self.sum_plt_data.setData(self.sum_x, self.sum_y)

    def update_tres_plot(self, spec_data):
        """ update the time resolved plot including the roi """
        try:
            gates = self.spec_data.softw_gates[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
            x_range = (float(np.min(spec_data.x[self.tres_sel_tr_ind])), np.max(spec_data.x[self.tres_sel_tr_ind]))
            x_scale = np.mean(np.ediff1d(spec_data.x[self.tres_sel_tr_ind]))
            y_range = (np.min(spec_data.t[self.tres_sel_tr_ind]), np.max(spec_data.t[self.tres_sel_tr_ind]))
            y_scale = np.mean(np.ediff1d(spec_data.t[self.tres_sel_tr_ind]))
            self.tres_widg.setImage(spec_data.time_res[self.tres_sel_tr_ind][self.tres_sel_sc_ind],
                                    pos=[x_range[0] - abs(0.5 * x_scale),
                                         y_range[0] - abs(0.5 * y_scale)],
                                    scale=[x_scale, y_scale])
            self.tres_plt_item.setAspectLocked(False)
            self.tres_plt_item.setRange(xRange=x_range, yRange=y_range, padding=0.05)
            self.tres_roi.setPos((gates[0], gates[2]), finish=False)
            self.tres_roi.setSize((abs(gates[0] - gates[1]), abs(gates[2] - gates[3])), finish=False)
        except Exception as e:
            print('error, while plotting time resolved, this happened: ', e)

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
            print('error in LiveDataPlotting, while setting the gates this happened: ', e)
        pass

    def update_projections(self, spec_data):
        """
        update the projections, if no plot has been done yet, create plotdata items for every plot
        """
        try:
            t_proj_x = spec_data.t_proj[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
            t_proj_y = spec_data.t[self.tres_sel_tr_ind]
            v_proj_x = spec_data.x[self.tres_sel_tr_ind]
            v_proj_y = spec_data.cts[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
            gates = self.spec_data.softw_gates[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
            sum_x, sum_y, sum_err = spec_data.getArithSpec(self.sum_scaler, self.tres_sel_tr_ind)

            if self.t_proj_plt is None:
                self.t_proj_plt = self.t_proj_plt_itm.plot(t_proj_x, t_proj_y, pen='k')
                self.sum_proj_plt_data = self.sum_proj_plt_itm.plot(sum_x, sum_y, pen='b')
                self.v_proj_plt = Pg.create_plot_data_item(v_proj_x, v_proj_y, pen='k')
                self.v_proj_view_box.addItem(self.v_proj_plt)

                self.current_step_line = Pg.create_infinite_line(self.spec_data.x[self.tres_sel_tr_ind][0],
                                                                 pen='r')

                self.v_min_line = Pg.create_infinite_line(gates[0])
                self.v_max_line = Pg.create_infinite_line(gates[1])
                self.t_min_line = Pg.create_infinite_line(gates[2], angle=0)
                self.t_max_line = Pg.create_infinite_line(gates[3], angle=0)

                self.sum_proj_plt_itm.addItem(self.current_step_line)
                self.sum_proj_plt_itm.addItem(self.v_min_line)
                self.sum_proj_plt_itm.addItem(self.v_max_line)
                self.t_proj_plt_itm.addItem(self.t_min_line)
                self.t_proj_plt_itm.addItem(self.t_max_line)
            else:
                self.t_proj_plt.setData(t_proj_x, t_proj_y)
                self.sum_proj_plt_data.setData(sum_x, sum_y)
                self.v_proj_plt.setData(v_proj_x, v_proj_y)
                self.v_min_line.setValue(gates[0])
                self.v_max_line.setValue(gates[1])
                self.t_min_line.setValue(gates[2])
                self.t_max_line.setValue(gates[3])

        except Exception as e:
            print('error, while plotting projection, this happened: ', e)

    def update_all_pmts_plot(self, spec_data, autorange_pls=False):
        if self.all_pmts_widg_plt_item_list is None:
            if spec_data.seq_type not in self.trs_names_list:
                self.tabWidget.setCurrentIndex(2)
            self.comboBox_all_pmts_sel_tr.blockSignals(True)
            tr_list = deepcopy(spec_data.track_names)
            tr_list.append('all')
            self.comboBox_all_pmts_sel_tr.addItems(tr_list)
            # self.cb_all_pmts_sel_tr_changed(self.comboBox_all_pmts_sel_tr.currentText())
            self.comboBox_all_pmts_sel_tr.blockSignals(False)

            self.add_all_pmt_plot()
        Pg.plot_all_sc(self.all_pmts_widg_plt_item_list, spec_data, self.all_pmts_sel_tr)
        if autorange_pls:
            [each['pltItem'].autoRange() for each in self.all_pmts_widg_plt_item_list]

    ''' buttons, comboboxes and listwidgets: '''

    def sum_scaler_changed(self, index=None):
        """
        this will set the self.sum_scaler list to the values set in the gui.
        :param index: int, index of the element in the combobox
        """
        print('sum_scaler_changed was called with index: %s ' % index)
        if index is None:
            index = self.comboBox_select_sum_for_pmts.currentIndex()
        if index == 0:
            if self.spec_data is not None:
                self.sum_scaler = self.spec_data.active_pmt_list[0]  # should be the same for all tracks
                self.sum_track = -1
                self.lineEdit_arith_scaler_input.setText(str(self.sum_scaler))
                self.lineEdit_sum_all_pmts.setText(str(self.sum_scaler))
            else:
                print('but specdata is None, so line edit is not set.')
            self.lineEdit_arith_scaler_input.setDisabled(True)
            self.lineEdit_sum_all_pmts.setDisabled(True)
        elif index == 1:
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
            print(e)

    def cb_all_pmts_sel_tr_changed(self, text):
        """ handle changes in the combobox in the all pmts tab """
        if text == 'all':
            tr_ind = -1
        else:
            tr_ind = self.comboBox_all_pmts_sel_tr.currentIndex()
        self.all_pmts_sel_tr = tr_ind
        if self.spec_data is not None and self.all_pmts_widg_plt_item_list is not None:
            self.update_all_pmts_plot(self.spec_data, autorange_pls=True)

    ''' gating: '''

    def gate_data(self, spec_data, plot_bool=True):
        rebin_track = -1 if self.checkBox.isChecked() else self.tres_sel_tr_ind
        self.new_gate_or_soft_bin_width.emit(
            spec_data.softw_gates, rebin_track, spec_data.softBinWidth_ns)

    ''' table operations: '''

    def handle_gate_table_change(self, item):
        # print('item was changed: ', item.row(), item.column(), item.text())
        gate_columns = range(2, 6)
        if item.column() in gate_columns:  # this means a gate value was changed.
            sel_tr = int(self.tableWidget_gates.item(item.row(), 0).text()[5:])
            sel_sc = int(self.tableWidget_gates.item(item.row(), 1).text())
            gate_ind = item.column() - 2
            new_val = float(item.text())
            self.spec_data.softw_gates[sel_tr][sel_sc][gate_ind] = new_val
            self.gate_data(self.spec_data)
            # print('gate of tr %s on scaler %s was changed to %f' % (sel_tr, sel_sc, new_val))

    def handle_item_clicked(self, item):
        """ this will select which track and scaler one is viewing. """
        if item.checkState() == QtCore.Qt.Checked:
            currently_selected = self.find_one_scaler_track(self.tres_sel_tr_name, self.tres_sel_sc_ind)
            curr_checkb_item = self.tableWidget_gates.item(currently_selected.row(), 6)
            curr_checkb_item.setCheckState(QtCore.Qt.Unchecked)
            self.tres_sel_tr_ind = int(self.tableWidget_gates.item(item.row(), 0).text()[5:])
            self.tres_sel_tr_name = self.tableWidget_gates.item(item.row(), 0).text()
            self.tres_sel_sc_ind = int(self.tableWidget_gates.item(item.row(), 1).text())
            # print('new scaler, track: ', self.tres_sel_tr_ind, self.tres_sel_sc_ind)
            self.rebin_data(self.spec_data.softBinWidth_ns[self.tres_sel_tr_ind])
            self.update_tres_plot(self.spec_data)
            self.update_projections(self.spec_data)

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
                        self.tableWidget_gates.setItem(row_ind, 0, QtWidgets.QTableWidgetItem(tr_name))
                        pmt_item = QtWidgets.QTableWidgetItem()
                        pmt_item.setData(QtCore.Qt.DisplayRole, pmt_name)
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
        # im_path = self.full_file_path.split('.')[0] + '_' + str(self.tres_sel_tr_name) + '_' + str(
        #     self.tres_sel_sc_name) + '.png'
        print('liveplot emitting save signal now')
        self.save_request.emit()

    ''' closing '''

    def closeEvent(self, *args, **kwargs):
        self.unsubscribe_from_main()
        if self.parent is not None:
            if self.subscribe_as_live_plot:
                self.parent.close_live_plot_win()
            else:
                self.parent.close_file_plot_win(self.full_file_path)

    ''' subscription to main '''

    def subscribe_to_main(self):
        self.save_callback.connect(self.save)
        self.new_track_callback.connect(self.setup_new_track)
        self.new_data_callback.connect(self.new_data)
        self.fit_results_dict_callback.connect(self.rcvd_fit_res_dict)
        if self.subscribe_as_live_plot:
            self.progress_callback.connect(self.handle_progress_dict)
            Cfg._main_instance.gui_live_plot_subscribe(
                self.callbacks, self.progress_callback, self.fit_results_dict_callback)

    def unsubscribe_from_main(self):
        if self.subscribe_as_live_plot:
            Cfg._main_instance.gui_live_plot_unsubscribe()

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
            self.extract_all_gates_from_gui(), rebin_track, self.spec_data.softBinWidth_ns)
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
            act_step = max(progress_dict_from_main['activeStep'] - 1, 0)
            act_tr_ind = progress_dict_from_main['activeTrack'] - 1
            self.update_step_indication_lines(act_step, act_tr_ind)
        if self.active_file != progress_dict_from_main['activeFile']:
            self.active_file = progress_dict_from_main['activeFile']
            self.setWindowTitle('plot:  %s' % self.active_file)
            self.dockWidget.setWindowTitle('progress: %s' % self.active_file)

        self.active_initial_scan_dict = Cfg._main_instance.scan_pars[self.active_iso]
        self.active_track_name = progress_dict_from_main['trackName']

    def update_step_indication_lines(self, act_step, act_tr):
        try:
            if self.all_pmts_widg_plt_item_list is not None:
                val = self.spec_data.x[act_tr][act_step]
                # print('active track is: %s and active step is: %s val is: %s ' % (act_tr, act_step, val))
                if self.current_step_line is not None:
                    self.current_step_line.setValue(val)
                [each['vertLine'].setValue(val) for each in self.all_pmts_widg_plt_item_list]
        except Exception as e:
            print(e)

    def reset(self):
        """
        reset all stored data etc. in order to prepare for a new isotope or so.
        :return:
        """
        self.spec_data = None
        self.setWindowTitle('plot: ...  loading  ... ')
        self.reset_table()
        self.reset_sum_plots()
        self.reset_all_pmt_plots()
        self.reset_t_res_plot()

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
        QtWidgets.QWidget().setLayout(self.all_pmts_plot_layout)
        # self.add_all_pmt_plot()  # do not call without specdata present!
        # will be called within update_plots() when first data arrives
        self.comboBox_sum_all_pmts.currentIndexChanged.emit(self.comboBox_sum_all_pmts.currentIndex())

    def reset_sum_plots(self):
        """
        reset the sum plot is not really necessary, because it will always look the same more or less.
        """
        QtWidgets.QWidget().setLayout(self.sum_plot_layout)
        self.sum_plt_data = None
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
        self.add_time_resolved_plot()


# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     ui = TRSLivePlotWindowUi()
#     ui.show()
#     app.exec()
