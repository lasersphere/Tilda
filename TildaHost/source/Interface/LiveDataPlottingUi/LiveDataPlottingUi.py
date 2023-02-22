"""
Created on 

@author: simkaufm

Module Description: GUI for displaying of data, both live and from file.
The data is analysed by a designated pipeline and then the data is emitted vie pyqtsignals to the gui.
Here it is only displayed. Gating etc. is done by the pipelines.

"""
import ast
import os
import functools
import logging
import time
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import Application.Config as Cfg
import PyQtGraphPlotter as Pg
from Interface.LiveDataPlottingUi.PreDurPostMeasUi import PrePostTabWidget
from Interface.LiveDataPlottingUi.Ui_LiveDataPlotting import Ui_MainWindow_LiveDataPlotting
from Interface.ScanProgressUi.ScanProgressUi import ScanProgressUi
from Measurement.XMLImporter import XMLImporter
import TildaTools as TiTs


class TRSLivePlotWindowUi(QtWidgets.QMainWindow, Ui_MainWindow_LiveDataPlotting):
    # these callbacks should be called from the pipeline:
    # for incoming new data:
    new_data_callback = QtCore.pyqtSignal(XMLImporter)
    # if a new track is started call:
    # the tuple is of form: ((tr_ind, tr_name), (pmt_ind, pmt_name))
    new_track_callback = QtCore.pyqtSignal(tuple)
    # when the pipeline wants to save, this is emitted and it send the pipeData as a dict
    save_callback = QtCore.pyqtSignal(dict)

    # TODO comment
    pre_post_meas_data_dict_callback = QtCore.pyqtSignal(dict)

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

    # float: self.needed_plot_update_time_ms, time which the last plot took in ms
    needed_plotting_time_ms_callback = QtCore.pyqtSignal(float)

    # save request
    save_request = QtCore.pyqtSignal()

    # progress dict coming from the main
    progress_callback = QtCore.pyqtSignal(dict)

    def __init__(self, full_file_path='', parent=None, subscribe_as_live_plot=True, sum_sc_tr=None, application=None):
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
        self.overall_scan_progress = 0  # float, will be 100 when scan completed updated via scan progress dict from main
        self.setupUi(self)
        self.show()
        self.tableWidget_gates.horizontalHeaderItem(4).setText('t_min [µs]')
        self.tableWidget_gates.horizontalHeaderItem(5).setText('t_max [µs]')
        self.tabWidget.setCurrentIndex(1)  # time resolved
        self.setWindowTitle('plot win:     ' + full_file_path)
        self.dockWidget.setWindowTitle('progress: %s' % self.active_file)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # necessary for not keeping it in memory

        # application can be given from top in order to force processing events
        self.application = application

        self.full_file_path = full_file_path
        # used to find out if plotting is currently in progress, e.g. before exporting a screenshot
        self.updating_plot = False

        self.allowed_update_time_ms = 200  # if this is exceeded, the tres is not plotted
        #  anymore in order to speed up plotting
        self.needed_plot_update_time_ms = 0.0
        self.calls_since_last_time_res_plot_update = 0  # to force update of tres plot if the plot is too slow.

        self.sum_plt_data = None
        self.trs_names_list = ['trs', 'trsdummy', 'tipa']

        self.tres_image = None
        self.t_proj_plt_itm = None
        self.tres_plt_item = None
        self.spec_data = None  # spec_data to work on.
        self.new_track_no_data_yet = False  # set this to true when new track is setup

        self.last_gr_update_done_time = datetime.now()
        self.last_rebin_time_stamp = datetime.now()
        self.allowed_rebin_update_rate = timedelta(milliseconds=500)
        self.graph_font_size = int(14)

        ''' connect callbacks: '''
        # bundle callbacks:
        self.subscribe_as_live_plot = subscribe_as_live_plot
        self.get_existing_callbacks_from_main()
        self.callbacks = (self.new_data_callback, self.new_track_callback,
                          self.save_request, self.new_gate_or_soft_bin_width,
                          self.pre_post_meas_data_dict_callback,
                          self.needed_plotting_time_ms_callback)
        self.subscribe_to_main()

        ''' key press '''
        self.actionGraph_font_size.setText(self.actionGraph_font_size.text() + '\tUp, Down')
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Up), self,
                            functools.partial(self.raise_graph_fontsize, True))
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Down), self,
                            functools.partial(self.raise_graph_fontsize, False))
        self.action_update.triggered.connect(functools.partial(self.update_all_plots, None, True))
        QtWidgets.QShortcut(QtGui.QKeySequence("CTRL+S"), self, self.export_screen_shot)  # Deprecated

        ''' sum related '''
        self.add_sum_plot()
        self.sum_x, self.sum_y, self.sum_err = None, None, None  # storage of the sum plotting values

        self.sum_scaler = [0]  # list of scalers to evaluate with each other
        self.function = None    # str: function defined by the user to calculate the sum plot from. either list of int
                                # e. g. '[0, 1]' or sth like '5 * s1 - ( s0 + s1 )'
        self.sum_track = -1  # int, for selecting the track which will be added. -1 for all
        self.sum_sc_tr_external = sum_sc_tr
        if self.sum_sc_tr_external is not None:
            # overwrite with external
            self.sum_scaler = self.sum_sc_tr_external[0]
            self.function = str(self.sum_scaler)
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
        self.lineEdit_arith_scaler_input.setToolTip(self.sum_scaler_lineedit_changed.__doc__)

        self.func_list = self.get_functions()
        try:
            self.comboBox_sum_tr.addItems(self.func_list)
        except Exception as e:
            self.comboBox_sum_tr.addItems(self.sum_scaler)
        self.comboBox_sum_tr.currentIndexChanged.connect(self.function_chosen)

        ''' time resolved related: '''  # TODO if timegate change, sum not correct anymore
        self.add_time_resolved_plot()
        self.tres_sel_tr_ind = 0  # int, index of currently observed track in time resolved spectra
        self.tres_sel_tr_name = 'track0'  # str, name of track
        self.tres_sel_sc_ind = 0  # int, index of currently observed scaler in time resolved spectra
        self.tres_sel_sc_name = '0'  # str, name of pmt

        self.tableWidget_gates.itemClicked.connect(self.handle_item_clicked)
        self.tableWidget_gates.itemChanged.connect(self.handle_gate_table_change)

        self.spinBox.valueChanged.connect(self.rebin_data)
        self.checkBox.stateChanged.connect(self.apply_rebin_to_all_checkbox_changed)

        self.pushButton_save_after_scan.setText('save current view')
        self.pushButton_save_after_scan.clicked.connect(self.export_screen_shot)

        self.setup_range_please = True  # boolean to store if the range has ben setup yet or not

        self.tres_offline_txt_itm = None  # Textitem to display, when tres plot is not plotted currently

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
        self.lineEdit_sum_all_pmts.setToolTip(self.sum_scaler_lineedit_changed.__doc__)

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
        self.splitter_3.setSizes([w * 55 // 100, int(w * 42.5 // 100)])
        ''' vertical splitter between all pmts plot and x/y coords widg '''
        self.splitter_allpmts.setSizes([h * 9 // 10, h * 1 // 10])

        ''' progress related: '''
        self.scan_prog_ui = None
        self.show_progress_window()

        self.actionProgress.setCheckable(True)
        self.actionProgress.setChecked(self.subscribe_as_live_plot)
        self.actionProgress.triggered.connect(self.show_progress)

        ''' screenshot related '''
        self.action_screenshot.triggered.connect(self.screenshot)
        self.action_screenshot_all.triggered.connect(self.screenshot_all)

        ''' font size graphs '''
        self.actionGraph_font_size.triggered.connect(self.get_graph_fontsize)

        ''' preset norm of all graphs '''
        self.scan_prog_array = None
        self.actionidentity.triggered.connect(self.toggle_norm_menu)
        self.actionscans.triggered.connect(self.toggle_norm_menu)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+N'), self, functools.partial(self.next_norm, True))
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Shift+N'), self, functools.partial(self.next_norm, False))
        self.menunorm.setTitle(self.menunorm.title() + '\tCtrl+N  ')

        ''' preset bin/error bar mode for all graphs '''
        self.stepMode = self.actionshow_bins.isChecked()
        self.actionshow_bins.triggered.connect(self.update_show_bins)

        ''' update sum  '''
        self.sum_scaler_changed()
        logging.info('LiveDataPlottingUi opened ... ')

        ''' pre / post / during related  '''
        self.pre_post_tab_widget = None
        self.tab_layout = None
        self.mutex = QtCore.QMutex()

        ''' detach/attach functionality all tabs '''
        # connect double click to detach function
        self.tabWidget.tabBarDoubleClicked.connect(self.detach_tab)
        # prepare tabs for re-attachment
        for index in range(0, self.tabWidget.count()):
            widget = self.tabWidget.widget(index)
            widget.index = index  # store the original indices of the widgets for re-attaching them at the correct index
            widget.closeEvent = self.make_new_close(widget)  # overwrite closeEvent for re-attaching

    def detach_tab(self, index):
        """
        removes the tab from the tabWidget and opens it as a new Window
        :param index: index of the tab to be detached
        """
        widget = self.tabWidget.widget(index)
        window_title = self.tabWidget.tabText(index)
        if index != -1:  # only if double click was on a tab header
            self.tabWidget.removeTab(index)
            widget.setWindowFlags(QtCore.Qt.Window)
            widget.setWindowTitle(window_title)
            # widget.setParent(None)  # necessary to make the new window independent from the old one
            widget.show()
            logging.info('Widget %s detached' % str(window_title))

    def make_new_close(self, parent):
        """
        provide a function to overwrite the closeEvent of the parent
        :param parent: widget whose closeEvent is to be re-defined
        :return: function that closes the window and calls the attach_tab function
        """
        tab_widget = self  # for passing self into the new function

        def close_detached(QCloseEvent):
            tab_widget.attach_tab(parent)

        return close_detached

    def attach_tab(self, widget):
        """
        re-attaches a previously detached widget to TabWidget at its original index
        :param widget: the widget to be re-attached
        """
        if self.tabWidget.indexOf(widget) == -1:  # for whatever reason the widget might be inside the tabWidget already
            widget.setWindowFlags(QtCore.Qt.Widget)
            # widget.setParent(self.tabWidget.widget(0).parent())  # parent was removed during detaching

            self.tabWidget.insertTab(widget.index, widget, widget.windowTitle())
            widget.show()
            logging.info('Window %s re-attached' % str(widget.windowTitle()))

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
        self.err_sum_plt_item = Pg.create_error_item()
        if not self.actionshow_bins.isChecked():  # Add errorbar plot if self.actionshow_bins is already unchecked.
            self.stepMode = False
            self.sum_plt_itm.addItem(self.err_sum_plt_item)
        self.sum_plot_layout = QtWidgets.QVBoxLayout()
        self.sum_plot_layout.addWidget(self.sum_wid)
        self.widget_inner_sum_plot.setLayout(self.sum_plot_layout)

    def add_time_resolved_plot(self):
        """ all plots on the time resolved tab -> time_resolved, time_projection, voltage_projection, sum """
        # time resolved related:
        self.tres_v_layout = QtWidgets.QVBoxLayout()
        self.tres_widg, self.tres_plt_item = Pg.create_image_view()
        self.tres_roi = Pg.create_roi([0, 0], [1, 1])
        self.tres_roi.sigRegionChangeFinished.connect(self.rect_select_gates)
        self.tres_plt_item.addItem(self.tres_roi)
        self.tres_v_layout.addWidget(self.tres_widg)
        self.widget_tres.setLayout(self.tres_v_layout)

        # sum / voltage projection related:
        self.sum_proj_wid, self.sum_proj_plt_itm = Pg.create_x_y_widget(do_not_show_label=['top'], y_label='sum')
        self.sum_proj_plt_itm.showAxis('right')
        self.err_sum_proj_plt_item = Pg.create_error_item()
        if not self.actionshow_bins.isChecked():  # Add errorbar plot if self.actionshow_bins is already unchecked.
            self.stepMode = False
            self.sum_proj_plt_itm.addItem(self.err_sum_proj_plt_item)
        self.v_proj_pltitem = Pg.create_plotitem()
        self.err_v_proj_plt_item = Pg.create_error_item()
        if not self.actionshow_bins.isChecked():  # Add errorbar plot if self.actionshow_bins is already unchecked.
            self.stepMode = False
            self.v_proj_pltitem.addItem(self.err_v_proj_plt_item)
        # self.sum_proj_plt_itm.scene().addItem(self.v_proj_pltitem.vb)
        self.sum_proj_plt_itm.scene().addItem(self.v_proj_pltitem.vb)
        # self.sum_proj_plt_itm.scene().addItem(self.err_v_proj_plt_item.getViewBox())
        # self.sum_proj_plt_itm.addItem(self.v_proj_pltitem)
        self.sum_proj_plt_itm.getAxis('right').linkToView(self.v_proj_pltitem.vb)
        # self.sum_proj_plt_itm.getAxis('right').linkToView(self.v_proj_pltitem)
        self.sum_proj_plt_itm.getAxis('right').setLabel('cts', color='k')
        pen = Pg.pg.mkPen(color='#0000ff', width=1)  # make the sum label and tick blue
        self.sum_proj_plt_itm.getAxis('left').setPen(pen)
        self.updateViews()
        self.sum_proj_plt_itm.vb.sigResized.connect(self.updateViews)
        self.v_proj_layout = QtWidgets.QVBoxLayout()
        self.v_proj_layout.addWidget(self.sum_proj_wid)
        self.widget_proj_v.setLayout(self.v_proj_layout)

        # self.sum_proj_plt_itm.getAxis('bottom').setTicks(self.tres_plt_item.getAxis('bottom').tick)

        # time projection related:
        self.t_proj_wid, self.t_proj_plt_itm = Pg.create_x_y_widget(do_not_show_label=['left', 'bottom'],
                                                                    y_label='time / µs', x_label='cts')
        self.t_proj_layout = QtWidgets.QVBoxLayout()
        self.t_proj_layout.addWidget(self.t_proj_wid)
        self.widget_proj_t.setLayout(self.t_proj_layout)
        # self.pushButton_save_after_scan.clicked.connect(self.save)
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

        # adjust ranges:
        t_res_range = self.tres_plt_item.vb.viewRange()
        # # -> [[-4.2249999999999996, 5.2249999999999996], [-0.10000000000000001, 1.1000000000000001]]
        # #  [[xmin, xmax], [ymin, ymax]]
        self.tres_plt_item.setRange(xRange=(t_res_range[0][0], t_res_range[0][1]),
                                    yRange=(t_res_range[1][0], t_res_range[1][1]),
                                    padding=0,
                                    update=True)
        self.tres_plt_item.setAspectLocked(False)

        self.sum_proj_plt_itm.setXRange(t_res_range[0][0], t_res_range[0][1], padding=0)
        # ranges needed to be linked manually since "setXLink" caused some really strange offset in the linked axis
        self.tres_plt_item.sigXRangeChanged.connect(self.tres_x_range_changed)
        self.sum_proj_plt_itm.sigXRangeChanged.connect(self.sum_x_range_changed)
        self.v_proj_pltitem.sigXRangeChanged.connect(self.v_proj_x_range_changed)

        self.t_proj_plt_itm.setYRange(t_res_range[1][0], t_res_range[1][1], padding=0)
        self.tres_plt_item.sigYRangeChanged.connect(self.tres_y_range_changed)
        self.t_proj_plt_itm.sigYRangeChanged.connect(self.t_proj_y_range_changed)

    def tres_x_range_changed(self, vb, xmin_xmax):
        self.sum_proj_plt_itm.setXRange(xmin_xmax[0], xmin_xmax[1], padding=0)
        self.v_proj_pltitem.vb.setXRange(xmin_xmax[0], xmin_xmax[1], padding=0)

    def sum_x_range_changed(self, vb, xmin_xmax):
        self.tres_plt_item.setXRange(xmin_xmax[0], xmin_xmax[1], padding=0)

    def v_proj_x_range_changed(self, vb, xmin_xmax):
        self.tres_plt_item.setXRange(xmin_xmax[0], xmin_xmax[1], padding=0)

    def tres_y_range_changed(self, vb, ymin_ymax):
        self.t_proj_plt_itm.setYRange(ymin_ymax[0], ymin_ymax[1], padding=0)

    def t_proj_y_range_changed(self, vb, ymin_ymax):
        self.tres_plt_item.setYRange(ymin_ymax[0], ymin_ymax[1], padding=0)

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
        if not self.actionshow_bins.isChecked():  # Add errorbar plot if self.actionshow_bins is already unchecked.
            self.stepMode = False
            for p in self.all_pmts_widg_plt_item_list:
                p['pltItem'].addItem(p['pltErrItem'])
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
        self.v_proj_pltitem.vb.setGeometry(self.sum_proj_plt_itm.vb.sceneBoundingRect())
        self.v_proj_pltitem.vb.linkedViewChanged(self.sum_proj_plt_itm.vb, self.v_proj_pltitem.vb.XAxis)

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
            # self.comboBox_all_pmts_sel_tr.setCurrentIndex(self.tres_sel_tr_ind)  # if uncommented: track in
                                                                                   # all-pmts-tab set to current track
        self.tres_sel_sc_ind, self.tres_sel_sc_name = rcv_tpl[1]
        self.new_track_no_data_yet = True
        self.setup_range_please = True
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

        if self.tres_offline_txt_itm is not None:
            self.tres_offline_txt_itm.setFont(font)

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

    def next_norm(self, _next):
        actions = self.menunorm.actions() if _next else self.menunorm.actions()[::-1]
        for i, a in enumerate(actions):
            if a.isChecked():
                actions[(i + 1) % len(actions)].trigger()
                return

    def toggle_norm_menu(self):
        action = self.sender()
        if action.isChecked():
            for a in self.menunorm.actions():
                a.setChecked(False)
        action.toggle()
        self.update_spec_data_norm()

    def update_spec_data_norm(self):
        """
        Update the normalization info in the spec_data object.

        :returns: None.
        """
        if self.spec_data is None:
            return
        if self.actionscans.isChecked():
            self.spec_data.norm_mode = 'scans'
            self.update_all_plots(self.spec_data)
        else:
            self.spec_data.norm_mode = '1'
            self.update_all_plots(self.spec_data)

    def update_show_bins(self):
        # self.new_data_callback.blockSignal(True)
        if self.actionshow_bins.isChecked():
            self.stepMode = True
            self.sum_plt_itm.removeItem(self.err_sum_plt_item)
            self.sum_proj_plt_itm.removeItem(self.err_sum_proj_plt_item)
            self.v_proj_pltitem.removeItem(self.err_v_proj_plt_item)
            if self.all_pmts_widg_plt_item_list is None:
                return
            for p in self.all_pmts_widg_plt_item_list:
                p['pltItem'].removeItem(p['pltErrItem'])
        else:
            self.stepMode = False
            self.sum_plt_itm.addItem(self.err_sum_plt_item)
            self.sum_proj_plt_itm.addItem(self.err_sum_proj_plt_item)
            self.v_proj_pltitem.addItem(self.err_v_proj_plt_item)
            if self.all_pmts_widg_plt_item_list is None:
                return
            for p in self.all_pmts_widg_plt_item_list:
                p['pltItem'].addItem(p['pltErrItem'])

        # self.new_data_callback.blockSignal(False)
        self.update_all_plots(self.spec_data)

    ''' receive and plot new incoming data '''

    def new_data(self, spec_data):
        """
        call this to pass a new dataset to the gui.
        """
        if spec_data is not None:
            # do not overwrite the spec_data with None values!
            try:
                st = datetime.now()
                valid_data = False
                self.spec_data = deepcopy(spec_data)

                update_time_ms = self.allowed_update_time_ms
                max_calls_without_plot = 5
                update_time_res_spec = self.needed_plot_update_time_ms <= update_time_ms \
                                       or self.calls_since_last_time_res_plot_update > max_calls_without_plot or \
                                       not self.subscribe_as_live_plot
                # update the time resolved spec if the last time the plot was faster plotted than 100ms
                # 150 ms should be ok to update all other plots
                # anyhow every fifth plot it will force to plot the time res
                if update_time_res_spec:
                    self.calls_since_last_time_res_plot_update = 0
                else:
                    logging.warning('did not update time resolved plot, because the last plotting time'
                                    ' was %.1f ms and this is longer than %.1f ms and it'
                                    ' was only missed %s times yet but %s are allowed'
                                    % (self.needed_plot_update_time_ms, update_time_ms,
                                       self.calls_since_last_time_res_plot_update, max_calls_without_plot))
                    self.calls_since_last_time_res_plot_update += 1

                self.update_spec_data_norm()
                self.update_all_plots(self.spec_data, update_tres=update_time_res_spec)
                if self.spec_data.seq_type in self.trs_names_list:
                    if not self.spinBox.hasFocus():
                        # only update when user is not entering currently
                        self.spinBox.blockSignals(True)
                        # blockSignals is necessary to avoid a loop since spinBox is connected to rebin_data(), which will
                        # emit a new_gate_or_soft_bin_width signal connected to rcvd_gates_and_rebin() Node that will again
                        # emit a new_data_callback that brings us back here...
                        self.spinBox.setValue(self.spec_data.softBinWidth_ns[self.tres_sel_tr_ind])
                        self.spinBox.blockSignals(False)
                    self.update_gates_list()
                valid_data = True
                if valid_data and self.new_track_no_data_yet:  # this means it is first call
                    # refresh the line edit by calling this here:
                    self.sum_scaler_changed(self.comboBox_sum_all_pmts.currentIndex())
                    if self.function == None:
                        self.function = str(self.sum_scaler)    # update to default function (list of all scalers)
                        self.add_func_to_options()

                    self.new_track_no_data_yet = False

                if not self.subscribe_as_live_plot:
                    # if not subscribed as live plot create scan dict from spec data once
                    # and emit, so that pre post meas is displayed properly
                    scan_dict = TiTs.create_scan_dict_from_spec_data(self.spec_data, self.full_file_path)
                    logging.debug('emitting %s, from %s, value is %s'
                                  % ('pre_post_meas_data_dict_callback',
                                     'Interface.LiveDataPlottingUi.LiveDataPlottingUi.TRSLivePlotWindowUi#new_data',
                                     str(scan_dict)))
                    self.pre_post_meas_data_dict_callback.emit(scan_dict)
                self.last_gr_update_done_time = datetime.now()
                elapsed_ms = (self.last_gr_update_done_time - st).total_seconds() * 1000
                self.needed_plot_update_time_ms = elapsed_ms
                logging.debug('emitting %s, from %s, value is %s'
                              % ('needed_plotting_time_ms_callback',
                                 'Interface.LiveDataPlottingUi.LiveDataPlottingUi.TRSLivePlotWindowUi#new_data',
                                 str(self.needed_plot_update_time_ms)))
                self.needed_plotting_time_ms_callback.emit(self.needed_plot_update_time_ms)

                # logging.debug('done updating plot, plotting took %.2f ms' % self.needed_plot_update_time_ms)
            except Exception as e:
                logging.error('error in liveplotterui while receiving new data: %s ' % e, exc_info=True)

    ''' updating the plots from specdata '''

    def update_rebin_spinbox_enable(self, force_enable=False, force_disable=False):
        """ call this to enable the rebinning spinbox automatically after a certain time """
        st = datetime.now()
        if self.overall_scan_progress == 100 or self.overall_scan_progress == 0:
            # scan complete or not started yet, don't restrict rebinning
            force_enable = True
            force_disable = False
        enable = (st - self.last_rebin_time_stamp) > self.allowed_rebin_update_rate or force_enable
        enable = enable and not force_disable  # will be always False if force_disable is True
        self.spinBox.setEnabled(enable)
        self.tableWidget_gates.setEnabled(enable)

    def update_all_plots(self, spec_data, update_tres=True):
        """ wrapper to update all plots """
        try:
            self.update_rebin_spinbox_enable()
            if spec_data is None and self.spec_data is not None:
                #  for updating by F5
                spec_data = self.spec_data
            logging.debug('updating all plots with %s %s' % (str(spec_data), str(update_tres)))
            self.updating_plot = True
            try:
                self.update_sum_plot(spec_data)
            except SyntaxError:
                logging.info('Your user function is invalid.')
            if '0.11' in pg.__version__:
                for num, track in enumerate(self.spec_data.cts):
                    # np.nan seems to be making trouble with plotting for pyqtgraph version 0.11.x TODO: Remove at some point?
                    self.spec_data.cts[num] = np.nan_to_num(track)
            if spec_data.seq_type in self.trs_names_list:
                if update_tres:
                    self.update_tres_plot(spec_data)
                    if self.tres_offline_txt_itm is not None:
                        self.tres_plt_item.removeItem(self.tres_offline_txt_itm)
                    self.tres_offline_txt_itm = None
                else:
                    # this time it will not update the time projection
                    # -> display a message to the user in the upper left corner
                    if self.tres_offline_txt_itm is None:
                        self.tres_offline_txt_itm = Pg.pg.TextItem(
                            text='this plot is currently offline, wait or press "F5"', color=(255, 0, 0))
                        font = QtGui.QFont()
                        font.setPixelSize(self.graph_font_size)
                        self.tres_offline_txt_itm.setFont(font)
                        total_w = self.tres_plt_item.width()
                        self.tres_offline_txt_itm.setTextWidth(int(total_w - 2 * total_w / 10))
                        self.tres_plt_item.addItem(self.tres_offline_txt_itm, ignoreBounds=True)
                        # ignoreBounds True -> this text will not be relevant when determining the autorange
                    tres_cur_range = self.tres_plt_item.viewRange()
                    text_x_pos = tres_cur_range[0][0] + (tres_cur_range[0][1] - tres_cur_range[0][0]) / 100
                    text_y_pos = tres_cur_range[1][1] + (tres_cur_range[1][1] - tres_cur_range[1][0]) / 100
                    logging.debug('writing offline message to position: x = %s, \t y = %s' % (text_x_pos, text_y_pos))
                    self.tres_offline_txt_itm.setPos(text_x_pos, text_y_pos)
                self.update_projections(spec_data)
            try:
                self.update_all_pmts_plot(spec_data)
            except SyntaxError:
                logging.info('Your user function is invaldi.')
            if self.application is not None:
                self.application.processEvents()
        except Exception as e:
            logging.error('error in updating plots: ' + str(e), exc_info=True)
        finally:
            self.updating_plot = False

    def update_sum_plot(self, spec_data):
        """
        update the sum plot and store the values in self.sum_x, self.sum_y, self.sum_err
        :param spec_data: SpecData, spectrum to plot
        :return:
        """
        if self.sum_scaler is not None:
            self.sum_x, self.sum_y, self.sum_err = spec_data.getArithSpec(
                self.sum_scaler, self.sum_track, self.function)
            if self.sum_plt_data is None:
                self.sum_plt_data = Pg.plot_std(self.sum_x, self.sum_y, self.sum_err, self.sum_plt_itm,
                                                self.err_sum_plt_item, stepMode=self.stepMode)
                # self.sum_plt_data = self.sum_plt_itm.plot(
                #     Pg.convert_xaxis_for_step_mode(self.sum_x), self.sum_y, stepMode=True, pen='k')
                if self.subscribe_as_live_plot:
                    self.sum_current_step_line = Pg.create_infinite_line(self.spec_data.x[self.tres_sel_tr_ind][0],
                                                                         pen='r')
                    self.sum_plt_itm.addItem(self.sum_current_step_line, ignoreBounds=True)
            else:
                Pg.set_data_std(self.sum_x, self.sum_y, self.sum_err, self.sum_plt_data, self.err_sum_plt_item,
                                stepMode=self.stepMode)
            self.sum_plt_itm.setLabel('bottom', spec_data.x_units.value)

    def update_tres_plot(self, spec_data):
        """ update the time resolved plot including the roi """
        try:

            gates = self.spec_data.softw_gates[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
            # x_range = (float(np.min(spec_data.x[self.tres_sel_tr_ind])), np.max(spec_data.x[self.tres_sel_tr_ind]))
            x_copy = deepcopy(spec_data.x[self.tres_sel_tr_ind])
            x_copy = Pg.convert_xaxis_for_step_mode(x_copy)
            x_range = (x_copy[0], x_copy[-1])
            x_scale = np.mean(np.ediff1d(spec_data.x[self.tres_sel_tr_ind]))
            y_range = (np.min(spec_data.t[self.tres_sel_tr_ind]), np.max(spec_data.t[self.tres_sel_tr_ind]))
            y_scale = np.mean(np.ediff1d(spec_data.t[self.tres_sel_tr_ind]))
            self.tres_widg.setImage(spec_data.time_res[self.tres_sel_tr_ind][self.tres_sel_sc_ind],
                                    pos=[x_range[0],
                                         y_range[0] - abs(0.5 * y_scale)],
                                    scale=[x_scale, y_scale],
                                    autoRange=False)
            self.tres_plt_item.setAspectLocked(False)
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

    def set_time_range(self, t_min=-1.0, t_max=-1.0, padding=0.05):
        """
        set the time range to display of the plot according to t_min / t_max
        :param t_min: float, -1.0 for automatic around gate
        :param t_max: float, -1.0 for automatic around gate
        :return:
        """
        if t_min >= 0 and t_max > 0:
            y_range = (t_min, t_max)
        else:
            gates = self.spec_data.softw_gates[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
            y_range = (gates[2], gates[3])
        self.tres_plt_item.setRange(yRange=y_range, padding=padding)

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
        :param spec_data: SpecData, spectrum to plot
        """
        try:
            if self.sum_scaler is not None:
                t_proj_x = spec_data.t_proj[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
                t_proj_y = spec_data.t[self.tres_sel_tr_ind]
                # v_proj_x = spec_data.x[self.tres_sel_tr_ind]
                # v_proj_y = spec_data.cts[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
                # v_proj_err = spec_data.err[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
                gates = self.spec_data.softw_gates[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
                v_proj_x, v_proj_y, v_proj_err = \
                    spec_data.getArithSpec([self.tres_sel_sc_ind, ], self.tres_sel_tr_ind, eval_on=False)
                sum_x, sum_y, sum_err = spec_data.getArithSpec(self.sum_scaler, self.tres_sel_tr_ind, self.function)

                if self.t_proj_plt is None:
                    self.t_proj_plt = self.t_proj_plt_itm.plot(t_proj_x, t_proj_y, pen='k')
                    self.sum_proj_plt_data = Pg.plot_std(sum_x, sum_y, sum_err, self.sum_proj_plt_itm,
                                                         self.err_sum_proj_plt_item, stepMode=self.stepMode)
                    self.v_proj_plt = Pg.plot_std(v_proj_x, v_proj_y, v_proj_err, self.v_proj_pltitem,
                                                  self.err_v_proj_plt_item, stepMode=self.stepMode, color='b')
                    self.v_proj_pltitem.vb.addItem(self.v_proj_plt)

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
                    Pg.set_data_std(self.sum_x, self.sum_y, self.sum_err, self.sum_proj_plt_data,
                                    self.err_sum_proj_plt_item, stepMode=self.stepMode, color='b')
                    Pg.set_data_std(v_proj_x, v_proj_y, v_proj_err, self.v_proj_plt,
                                    self.err_v_proj_plt_item, stepMode=self.stepMode)
                    self.v_min_line.setValue(gates[0])
                    self.v_max_line.setValue(gates[1])
                    self.t_min_line.setValue(gates[2])
                    self.t_max_line.setValue(gates[3])
                self.sum_proj_plt_itm.setLabel('bottom', spec_data.x_units.value)

        except SyntaxError:
            logging.info('Your user function is invalid')

    def update_all_pmts_plot(self, spec_data, autorange_pls=False):
        """
        updates the all pmts tab
        :param spec_data: SpecData, used spectrum
        :param autorange_pls:
        """
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

        Pg.plot_all_sc_new(self.all_pmts_widg_plt_item_list, spec_data, self.all_pmts_sel_tr,
                           self.function, stepMode=self.stepMode)

        self.all_pmts_widg_plt_item_list[-1]['pltItem'].setLabel('bottom', spec_data.x_units.value)
        self.all_pmts_widg_plt_item_list[0]['pltItem'].setLabel('top', spec_data.x_units.value)

        if autorange_pls:
            [each['pltItem'].autoRange() for each in self.all_pmts_widg_plt_item_list]

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
                self.lineEdit_arith_scaler_input.setText(str(self.sum_scaler))  # TODO change
                self.lineEdit_sum_all_pmts.setText(str(self.sum_scaler))  # TODO change
                self.function = str(self.sum_scaler)
            else:
                logging.info('liveplotterui: but specdata is None, so line edit is not set.')
            self.lineEdit_arith_scaler_input.setDisabled(True)
            self.lineEdit_sum_all_pmts.setDisabled(True)
        elif index == 1:  # manual
            if self.spec_data is not None:
                if self.spec_data.seq_type != 'kepco':
                    if self.sum_scaler is None:
                        self.sum_scaler_lineedit_changed(self.function)
                if self.sum_track is None:
                    self.sum_track = -1
            self.lineEdit_arith_scaler_input.setDisabled(False)
            self.lineEdit_sum_all_pmts.setDisabled(False)

        # synchronize all comboboxes
        try:
            self.add_func_to_options()
        except Exception as e:
            logging.error('No function defined yet')
        self.comboBox_sum_all_pmts.setCurrentIndex(index)
        self.comboBox_select_sum_for_pmts.setCurrentIndex(index)

    def get_functions(self):
        func_dict = Cfg._main_instance.get_option('FUNCTIONS')
        func_dict = Cfg._main_instance.local_options.get_functions()
        func_list = []
        for key, value in func_dict.items():
            func_list.append(value)
        return func_list

    def function_chosen(self):
        index = self.comboBox_sum_tr.currentIndex()
        try:
            functions = self.get_functions()
            self.sum_scaler_lineedit_changed(functions[index])
            self.comboBox_select_sum_for_pmts.setCurrentIndex(1)
        except Exception as e:
            logging.error('no such function in options')

    def sum_scaler_lineedit_changed(self, text):
        """
        define your own function in the form of "s0 + s1" or "s3 / ( s2 + s1 )"
        (need blanks inbetween!)
        +, -, *, /, ** are allowed operators
        "[0, 1, 2]" gives sum of scaler 0, 1 and 2
        """

        ''' process user input '''
        indList = []  # indList for list_of_widgest sum-widget and self.sum_scaler

        if text[0] != '[':  # if user input is not a list of scalers
            input_list = text.split()   # separate numbers, variables and operators
            operators = ['+', '-', '*', '/', '(', ')', '**']  # allowed operators
            input_numbers = []
            input_vars = []

            '''sort by number, variable, operator'''
            for each in input_list:
                if each in operators:   # check if operator
                    pass
                else:
                    try:
                        float(each)  # check if number
                        input_numbers.append(each)
                    except ValueError:
                        input_vars.append(each)  # declare as variable

                    ''' check if variables are ok '''
                    vars = []   # allowed variables
                    i = 0
                    while i < self.spec_data.nrScalers[0]:
                        vars.append('s'+ str(i))
                        i += 1
                    try:
                        for each in input_vars: # check if input vars ok
                            if each not in vars:
                                raise Exception("Invalid Syntax: only %s are allowed variable names" % vars)
                        for var in input_vars:  # create index list
                            indList.append(int(var[1]))
                    except Exception as e:
                        logging.info('Error %s' % e)
        else:   # if user input is a list of scalers
            try:
                indList = ast.literal_eval(text)
            except Exception as e:
                logging.error('error on changing line edit of summed scalers in liveplotterui: %s' % e, exc_info=True)

        ''' update plots '''
        try:
            isinteger = len(indList) > 0
            if isinteger:
                self.sum_scaler = indList
                self.function = text
                self.update_sum_plot(self.spec_data)
                self.update_projections(self.spec_data)
                if self.all_pmts_widg_plt_item_list is not None:
                    self.all_pmts_widg_plt_item_list[-1]['indList'] = indList
                    self.update_all_pmts_plot(self.spec_data)
                cursor_all_pmts = self.lineEdit_sum_all_pmts.cursorPostion()
                cursor_sum = self.lineEdit_arith_scaler_input.cursorPosition()
                self.lineEdit_sum_all_pmts.setText(text)
                self.lineEdit_arith_scaler_input.setText(text)
                self.lineEdit_sum_all_pmts.setCursorPosition(cursor_all_pmts)
                self.lineEdit_sum_all_pmts.setCursorPosition(cursor_sum)
                self.add_func_to_options()
                #self.set_preset_function_menue(text)
        except Exception as e:
            logging.info('incorrect user input for function')

    def add_func_to_options(self):
        func_new = True
        functions = Cfg._main_instance.local_options.get_functions()
        for key, value in functions.items():
            if value.strip() == self.function.strip():
                func_new = False
                self.comboBox_sum_tr.setCurrentIndex(key)
        if func_new:
            self.func_list.append(self.function)
            self.comboBox_sum_tr.addItems([self.function])
            Cfg._main_instance.local_options.add_function(self.function)
            Cfg._main_instance.save_options()
            self.comboBox_sum_tr.setCurrentIndex(len(functions))

    def cb_all_pmts_sel_tr_changed(self, text):
        """ handle changes in the combobox in the all pmts tab """
        if text == 'all':
            tr_ind = -1
        else:
            tr_ind = self.comboBox_all_pmts_sel_tr.currentIndex()
        self.all_pmts_sel_tr = tr_ind
        if self.spec_data is not None and self.all_pmts_widg_plt_item_list is not None:
            self.update_all_pmts_plot(self.spec_data, autorange_pls=True)
            [each['pltItem'].enableAutoRange(each['pltItem'].getViewBox().XYAxes) for each in
             self.all_pmts_widg_plt_item_list]

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
        logging.debug('emitting %s, from %s, value is %s'
                      % ('new_gate_or_soft_bin_width',
                         'Interface.LiveDataPlottingUi.LiveDataPlottingUi.TRSLivePlotWindowUi#gate_data',
                         str((spec_data.softw_gates, rebin_track, spec_data.softBinWidth_ns, plot_bool))))
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
        logging.debug('emitting %s, from %s, value is %s'
                      % ('save_request',
                         'Interface.LiveDataPlottingUi.LiveDataPlottingUi.TRSLivePlotWindowUi#save',
                         str('-')))
        self.save_request.emit()

    ''' closing '''

    # def closeEvent(self, event):
    #     if self.subscribe_as_live_plot:
    #         self.unsubscribe_from_main()

    ''' subscription to main '''

    def get_existing_callbacks_from_main(self):
        """ check wether existing callbacks are still around in the main and then connect to those. """
        if Cfg._main_instance is not None and self.subscribe_as_live_plot:
            logging.info('TRSLivePlotWindowUi is connecting to existing callbacks in main')
            callbacks = Cfg._main_instance.gui_live_plot_subscribe()
            self.new_data_callback = callbacks[0]
            self.new_track_callback = callbacks[1]
            self.save_request = callbacks[2]
            self.new_gate_or_soft_bin_width = callbacks[3]
            self.fit_results_dict_callback = callbacks[4]
            self.progress_callback = callbacks[5]
            self.pre_post_meas_data_dict_callback = callbacks[6]
            self.needed_plotting_time_ms_callback = callbacks[7]

    def subscribe_to_main(self):
        if self.subscribe_as_live_plot:
            self.progress_callback.connect(self.handle_progress_dict)
            # overwrite gui callbacks with callbacks from main
            self.get_existing_callbacks_from_main()
        self.save_callback.connect(self.save)
        self.new_track_callback.connect(self.setup_new_track)
        self.new_data_callback.connect(self.new_data)
        self.fit_results_dict_callback.connect(self.rcvd_fit_res_dict)
        self.pre_post_meas_data_dict_callback.connect(self.rebuild_pre_post_meas_gui)

    ''' rebinning '''

    def rebin_data(self, rebin_factor_ns=None):
        """
        request a rebinning from the pipeline,
        will only request this rebinning if the appropiate
        amount of time (self.allowed_rebin_update_rate) has passed.
        """
        start = datetime.now()
        if start - self.last_rebin_time_stamp > self.allowed_rebin_update_rate:
            # only allow update after self.allowed_rebin_update_rate has passed
            self.last_rebin_time_stamp = start  # set to last time stamp
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
            logging.debug('emitting %s, from %s, value is %s'
                          % ('new_gate_or_soft_bin_width',
                             'Interface.LiveDataPlottingUi.LiveDataPlottingUi.TRSLivePlotWindowUi#rebin_data',
                             str((self.extract_all_gates_from_gui(), rebin_track,
                                  self.spec_data.softBinWidth_ns, False))))
            self.update_rebin_spinbox_enable(force_disable=True)  # now disable until enough time has passed.
            self.new_gate_or_soft_bin_width.emit(
                self.extract_all_gates_from_gui(), rebin_track,
                self.spec_data.softBinWidth_ns, False)
            stop = datetime.now()
            dif = stop - start
            # print('rebinning took: %s' % dif)
        else:
            logging.warning('tried to rebin, but not enough time has passed since last rebin.'
                            'time passed: %.1f, minimum time between: %.1f'
                            % ((start - self.last_rebin_time_stamp).total_seconds(),
                               self.allowed_rebin_update_rate.total_seconds()))

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
                # logging.debug('active track is: %s and active step is: %s val is: %s ' % (act_tr, act_step, val))
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
        logging.info('resetting LiveDataPlottingUi now ... ')
        self.spec_data = None
        self.setWindowTitle('plot: ...  loading  ... ')
        self.scan_prog_ui.reset()
        self.reset_table()
        self.reset_sum_plots()
        self.reset_all_pmt_plots()
        self.reset_t_res_plot()
        self.reset_pre_post_meas_gui()

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
        logging.debug('emitting %s, from %s, value is %s'
                      % ('comboBox_sum_all_pmts',
                         'Interface.LiveDataPlottingUi.LiveDataPlottingUi.TRSLivePlotWindowUi#reset_all_pmt_plots',
                         str(self.comboBox_sum_all_pmts.currentIndex())))
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
        self.comboBox_select_sum_for_pmts.currentIndexChanged.emit(self.comboBox_select_sum_for_pmts.currentIndex())

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

    def rebuild_pre_post_meas_gui(self, pre_post_meas_dict):
        """
        add a tab for each track on the tab 'pre/during/post scan measurements'.
        if it exists, update the data inside
        :param pre_post_meas_dict: the new data
        """
        if self.pre_post_tab_widget is None:
            if self.tab_layout is None:
                logging.info('creating tab layout')
                self.tab_layout = QtWidgets.QGridLayout(self.tab_pre_post_meas)
            self.mutex.lock()

            self.pre_post_tab_widget = PrePostTabWidget(pre_post_meas_dict, self.subscribe_as_live_plot)
            self.tab_layout.addWidget(self.pre_post_tab_widget)
            self.mutex.unlock()
        else:
            self.mutex.lock()
            self.pre_post_tab_widget.update_data(pre_post_meas_dict)
            self.mutex.unlock()

    def reset_pre_post_meas_gui(self):
        '''
        call the self destruction of the widget
        '''
        if self.pre_post_tab_widget is not None:
            # if the tab_widget exists, delete it.
            logging.debug('deleting pre_pre_post_tab_widget')
            self.mutex.lock()  # Make sure no othe fuction call tries to access the widget in between.
            self.pre_post_tab_widget.deleteLater()
            self.pre_post_tab_widget = None  # Make sure it's None so its in a definite state
            self.mutex.unlock()  # Release the widget for other function calls.

    ''' screenshot related '''

    @staticmethod
    def _image_to_clipboard(image):
        mime_data = QtCore.QMimeData()
        mime_data.setImageData(image)

        QtGui.QGuiApplication.clipboard().setMimeData(mime_data)

    @staticmethod
    def _screenshot_widget(widget):
        QtGui.QPainter(widget).end()
        pixmap = QtGui.QPixmap(widget.size())
        widget.render(pixmap)
        return pixmap.toImage()

    def screenshot(self):
        # Screenshot current view
        c_image = self._screenshot_widget(self)

        # Prepare run info
        run_info = self.windowTitle()
        run_height = 20

        # Create composite image
        image = QtGui.QImage(c_image.width(), c_image.height() + run_height, c_image.format())
        painter = QtGui.QPainter(image)

        # Add run info
        white = QtGui.QColor('white')
        painter.setPen(white)
        painter.setBrush(white)
        painter.drawRect(0, 0, image.width(), run_height)
        painter.setPen(QtGui.QColor('black'))
        font = QtGui.QFont()
        font.setPixelSize(16)
        painter.setFont(font)
        painter.drawText(5, run_height - 5, run_info)

        # Add image
        painter.drawImage(0, run_height, c_image)

        painter.end()  # Important!
        self._image_to_clipboard(image)

    def screenshot_all(self):
        # Prepare run info
        run_info = self.windowTitle()
        run_height = 20

        # Screenshot all tabs
        index = self.tabWidget.currentIndex()
        images = []
        for i in range(self.tabWidget.count()):
            self.tabWidget.setCurrentIndex(i)
            images.append(self._screenshot_widget(self.tabWidget.widget(i)))
        self.tabWidget.setCurrentIndex(index)

        # Screenshot progress window
        p_width, p_height = 0, 0
        if self.actionProgress.isChecked():
            p_image = self._screenshot_widget(self.widget_progress)
            p_width, p_height = p_image.width(), p_image.height()

        # Create composite image
        c_image = images[index]
        width = c_image.width()
        height = c_image.height()
        image = QtGui.QImage(2 * width + p_width, 2 * height + run_height, c_image.format())
        painter = QtGui.QPainter(image)

        # Add run info
        white = QtGui.QColor('white')
        painter.setPen(white)
        painter.setBrush(white)
        painter.drawRect(0, 0, image.width(), run_height)
        painter.setPen(QtGui.QColor('black'))
        font = QtGui.QFont()
        font.setPixelSize(16)
        painter.setFont(font)
        painter.drawText(5, run_height - 5, run_info)

        # Add images
        x_pos = [0, 1, 0, 1]
        y_pos = [0, 0, 1, 1]
        for x, y, _image in zip(x_pos, y_pos, images):
            painter.drawImage(x * width, y * height + run_height, _image)

        # Add progress window
        if self.actionProgress.isChecked():
            p_image = self._screenshot_widget(self.widget_progress)
            painter.drawImage(2 * x_pos[-1] * width, y_pos[0] + run_height, p_image)
            grey = self.palette().color(self.backgroundRole())
            painter.setPen(grey)
            painter.setBrush(grey)
            painter.drawRect(2 * x_pos[-1] * width, p_height + run_height, p_width, image.height() - p_height)

        painter.end()  # Important!
        self._image_to_clipboard(image)

    def export_screen_shot(self, storage_path='', quality=100):  # deprecated
        """
        function to export a screenshot fo the full window.
        and export all currently shown plots of the time resolved tab
        as .png, .svg and their data as .csv using the export functions of pyqtgraph.
        :param storage_path: str, full storage path, leave empty to open dialog
        :param quality: int, -1 default, range 0-100, 0 -> low quality save, 100 -> high quality save
        :return: path
        """
        start = datetime.now()
        if not storage_path:
            dial = QtWidgets.QFileDialog(self)
            if self.full_file_path:
                init_filter = self.full_file_path.split('.')[0] + '.png'
            elif self.active_file:
                init_filter = self.active_file.split('.')[0] + '.png'
            else:
                init_filter = '.png'
            logging.debug('initial filter for q-Dialog is: %s' % init_filter)
            storage_path, ending = QtWidgets.QFileDialog.getSaveFileName(dial,
                                                                         'Chooose a storage location for the screenshot',
                                                                         init_filter,
                                                                         '*.png'
                                                                         )

            if not storage_path:  # cancel clicked
                return None
        timeout = 0
        while self.updating_plot and timeout < 10:
            time.sleep(0.01)
            timeout += 0.01
        logging.info('grabbing screen now')
        self.update()
        pixm = self.grab()
        ending = os.path.splitext(storage_path)[1]
        # logging.debug('ending is %s' % ending)
        if pixm.save(storage_path, ending[1:], quality):
            stop = datetime.now()
            dif = stop - start
            dif_sec = dif.microseconds * 10 ** -6
            logging.info('saved screenshot with a quality of: %s after %.3fs to: %s' % (quality, dif_sec, storage_path))
        else:
            logging.warning('saving went wrong, did not save to: %s ' % storage_path)
        # with pyqtgraph:
        import pyqtgraph.exporters as pgexp
        plots = [
            self.sum_proj_plt_itm, self.t_proj_plt_itm, self.tres_plt_item #, self.v_proj_pltitem  currently somehow this causes an error
        ]
        names = ['_sum', '_t_proj', '_tres', '_v_proj']
        try:
            # try loop because this is only now optimized for time resolved plots etc.
            t_res_pixelsize_vb = (self.tres_plt_item.vb.width(), self.tres_plt_item.vb.height())  # (width, height)
            width_scale = 1024 / t_res_pixelsize_vb[0]  # viewbox should be 1024 wide
            t_res_vb_final_height = int(width_scale * t_res_pixelsize_vb[1])
            # store final height of time resolved viewbox
        except Exception as e:
            logging.error(e, exc_info=True)
        for i, pl in enumerate(plots):
            try:
                base_file_name = os.path.splitext(storage_path)[0]

                # export as .svg, which sometimes does not look that nice :(
                exporter_svg = pgexp.SVGExporter(pl)
                exporter_svg.export(base_file_name + names[i] + '.svg')

                # export as .png:
                pl_w, pl_h = (pl.width(), pl.height())
                pl_vb_w, pl_vb_h = (pl.vb.width(), pl.vb.height())
                width_scale = 1024 / pl_vb_w  # viewbox should be 1024 wide in the end
                height_scale = t_res_vb_final_height / pl_vb_h
                exporter_png = pgexp.ImageExporter(pl)
                if names[i] == '_t_proj':
                    # this is a t_proj were the height shjould match with t_res
                    # scale height to same as vb of final tres
                    exporter_png.parameters()['height'] = int(pl_h * height_scale)
                    logging.debug('scaled height to: %s' % int(pl_h * height_scale))
                else:
                    exporter_png.parameters()['width'] = int(pl_w * width_scale)
                    # viewbox should be 1024 wide and proportional height
                exporter_png.export(base_file_name + names[i] + '.png')

                #  export the currently visible as .csv
                if 'tres' in names[i]:
                    tres_csv_file = base_file_name + names[i] + '.csv'
                    # copy current matrix of counts with all zeros... but witht he applied time binning
                    cts_float = deepcopy(self.spec_data.time_res[self.tres_sel_tr_ind][self.tres_sel_sc_ind]).astype(dtype=np.float)
                    t_axis = deepcopy(self.spec_data.t[self.tres_sel_tr_ind])
                    t_axis = np.insert(t_axis, 0, 0)  # expand t_axis by one for coord 0,0 in matrix
                    x_axis = deepcopy(self.spec_data.x[self.tres_sel_tr_ind])
                    # cts_float = value.astype(dtype=np.float)
                    # print(cts_float.shape)
                    cts_float = np.insert(cts_float, 0, x_axis, axis=1)
                    # print(cts_float_pl_x.shape)
                    cts_float = np.insert(cts_float, 0, t_axis, axis=0)
                    # print(cts_float_pl_x_y.shape)
                    np.savetxt(
                        tres_csv_file, cts_float, delimiter='\t', fmt='%.5f',
                        header='column indicator is time axis, row indicator is volt axis. Works fine with origin')

                else:
                    # time resolved array causes problems as csv
                    exporter_csv = pgexp.CSVExporter(pl)
                    exporter_csv.parameters()['separator'] = 'tab'
                    exporter_csv.parameters()['precision'] = 100  # 10 ns
                    exporter_csv.export(base_file_name + names[i] + '.csv')
            except Exception as e:
                logging.error('error while saving %s : %s' % (names[i], e), exc_info=True)
        return storage_path


if __name__ == "__main__":
    import sys
    from Service.AnalysisAndDataHandling.DisplayData import DisplayData

    app_log = logging.getLogger()
    # app_log.setLevel(getattr(logging, args.log_level))
    app_log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # ch.setFormatter(log_formatter)
    app_log.addHandler(ch)

    app_log.info('****************************** starting ******************************')
    app_log.info('Log level set to DEBUG')

    test_file = 'C:\\Users\\Laura Renth\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\' \
                'Measurement_and_Analysis_Simon\\Ni_workspace2017\\Ni_2017\\sums\\60_Ni_trs_run114.xml'

    app = QtWidgets.QApplication(sys.argv)
    ui = TRSLivePlotWindowUi(test_file, subscribe_as_live_plot=False, application=app)
    spec = XMLImporter(test_file)
    spec.softBinWidth_ns = [100]
    disp_data = DisplayData(test_file, ui, x_as_volt=True, loaded_spec=spec)

    # test_dict = {'track0': {'preScan':
    #                             {'dmm': {'dmm1': {'data': [1, 2, 3, 4, 5, 6, 7], 'required': 9}},
    #                              'triton':  {'dev0': {'data': [1, 2, 3, 4, 5], 'required': 10}}},
    #                         'duringScan':
    #                             {'dmm': {'dmm1': {'data': [], 'required': 100},
    #                                      'dmm2': {'data': [1,2], 'required': 2}},
    #                              'triton': {'dev0': {'data': [1, 2, 3, 4, 5], 'required': 10},
    #                                         'dev1': {'data': [1, 2, 3, 4, 5], 'required': 10},
    #                                         'dev2': {'data': [1, 2, 3, 4, 5], 'required': 5}}},
    #                         'postScan':
    #                             {'dmm': {'dmm1': {'data': [1, 2, 3, 4, 5, 6, 7], 'required': 9}},
    #                              'triton': {'dev0': {'data': [1, 2, 3, 4, 5], 'required': 10}}}
    #                         },
    #              'track1': {'preScan':
    #                             {'dmm': {'dmm1': {'data': [1, 2, 3, 4, 5, 6, 7], 'required': 9}},
    #                              'triton':  {'dev0': {'data': [1, 2, 3, 4, 5], 'required': 10}}},
    #                         'postScan':
    #                             {'dmm': {'dmm1': {'data': [1, 2, 3, 4, 5, 6, 7], 'required': 9}},
    #                              'triton': {'dev0': {'data': [1, 2, 3, 4, 5], 'required': 10}}}
    #                         }}
    # test_dict = {'isotopeData': {'type': 'trsdummy', 'nOfTracks': 1, 'accVolt': 3000.0, 'version': '1.20',
    #                              'isotopeStartTime': '2017-12-20 14:32:13', 'isotope': 'anew', 'laserFreq': 120.0},
    #              'pipeInternals': {'activeTrackNumber': (0, 'track0'), 'workingDirectory': 'C:\\TRITON_TILDA\\Temp',
    #                                'activeXmlFilePath': 'C:\\TRITON_TILDA\\Temp\\sums\\anew_trsdummy_run132.xml',
    #                                'curVoltInd': 0},
    #              'track0': {'activePmtList': [0, 1],
    #                         'measureVoltPars': {'duringScan': {'measVoltPulseLength25ns': 400,
    #                                                            'dmms': {'dummy_somewhere': {'triggerDelay_s': 0.0,
    #                                                                                         'sampleCount': 0,
    #                                                                                         'readings': [1.0, 1.0, 1.0,
    #                                                                                                      1.0, 1.0, 1.0,
    #                                                                                                      1.0, 1.0, 1.0,
    #                                                                                                      1.0],
    #                                                                                         'triggerCount': 0
    #                                                                                         }
    #                                                                     }
    #                                                            },
    #                                             'postScan': {},
    #                                             'preScan': {'dmms': {'dummy_somewhere': {'sampleCount': 10,
    #                                                                                      'readings': [1.0, 1.0, 1.0,
    #                                                                                                   1.0, 1.0, 1.0,
    #                                                                                                   1.0, 1.0, 1.0,
    #                                                                                                   1.0]
    #                                                                                      }
    #                                                                  }
    #                                                         }
    #                                             },
    #                         'triton': {'duringScan': {'dev0': {'ch0': {'aquired': 6,
    #                                                                    'data': [1, 2, 3, 4, 5, 6],
    #                                                                    'required': 10
    #                                                                    }
    #                                                            }
    #                                                   },
    #                                    'preScan': {},
    #                                    'postScan': {}
    #                                    }
    #                         },
    #              'track1': {'measureVoltPars': {'duringScan': {'dmms': {'dummy_somewhere': {'sampleCount': 0,
    #                                                                                         'readings': [1.0, 1.0, 1.0,
    #                                                                                                      1.0, 1.0, 1.0,
    #                                                                                                      1.0, 1.0, 1.0,
    #                                                                                                      1.0]
    #                                                                                         }
    #                                                                     }
    #                                                            },
    #                                             'postScan': {},
    #                                             'preScan': {'dmms': {'dummy_somewhere': {'sampleCount': 10,
    #                                                                                      'readings': [1.0, 1.0, 1.0,
    #                                                                                                   1.0, 1.0, 1.0,
    #                                                                                                   1.0, 1.0, 1.0,
    #                                                                                                   1.0]
    #                                                                                      }
    #                                                                  }
    #                                                         }
    #                                             },
    #                         'triton': {'duringScan': {'dev0': {'ch0': {'aquired': 10,
    #                                                                    'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                                                                    'required': 10
    #                                                                    }
    #                                                            }
    #                                                   },
    #                                    'preScan': {},
    #                                    'postScan': {'dev0': {'ch0': {'aquired': 6,
    #                                                                  'data': [1, 2, 3, 4, 5, 6],
    #                                                                  'required': 10
    #                                                                  }
    #                                                          }
    #                                                 }
    #                                    }
    #                         }
    #              }
    #
    # app = QtWidgets.QApplication(sys.argv)
    # ui = TRSLivePlotWindowUi()
    # ui.tabWidget.setCurrentIndex(3)  # time resolved
    #
    # ui.pre_post_meas_data_dict_callback.emit(test_dict)
    # test_dict2 = deepcopy(test_dict)
    # test_dict2['track0']['triton']['duringScan'] = {'dev0': {'ch1': {'aquired': 13,
    #                                                                  'data': [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    #                                                                           0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    #                                                                           13,
    #                                                                           0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    #                                                                           13],
    #                                                                  'required': 0
    #                                                                  }
    #                                                          }
    #                                                 }
    # ui.show()
    # ui.tabWidget.setCurrentIndex(3)  # time resolved
    #
    # for i in range(0, 100):
    #     ui.pre_post_meas_data_dict_callback.emit(test_dict2)
    # # ui.pre_post_meas_data_dict_callback.emit(test_dict2)
    # # ui.pre_post_meas_data_dict_callback.emit(test_dict2)
    #
    # # time.sleep(2)

    app.exec()
