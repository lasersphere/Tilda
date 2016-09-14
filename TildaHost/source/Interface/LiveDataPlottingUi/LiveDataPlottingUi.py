"""
Created on 

@author: simkaufm

Module Description:
"""
import ast
import logging
from copy import deepcopy

import numpy as np
from PyQt5 import QtWidgets, QtCore

import Application.Config as Cfg
import PyQtGraphPlotter as Pg
import Service.FileOperations.FolderAndFileHandling as FileHandl
import Service.Formating as Form
import TildaTools as TiTs
from Interface.LiveDataPlottingUi.Ui_LiveDataPlotting import Ui_MainWindow_LiveDataPlotting
from Measurement.XMLImporter import XMLImporter


class TRSLivePlotWindowUi(QtWidgets.QMainWindow, Ui_MainWindow_LiveDataPlotting):
    # these callbacks should be called from the pipeline:
    # for incoming new data:
    new_data_callback = QtCore.pyqtSignal(XMLImporter)
    # if a new track is started call:
    # the tuple is of form: ((tr_ind, tr_name), (pmt_ind, pmt_name))
    new_track_callback = QtCore.pyqtSignal(tuple)
    # when teh pipeline wants to save, this is emitted and it send the pipeData as a dict
    save_callback = QtCore.pyqtSignal(dict)

    # progress dict coming from the main
    progress_callback = QtCore.pyqtSignal(dict)

    def __init__(self, full_file_path='', parent=None):
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
        self.setWindowTitle('plot win:     ' + full_file_path)

        self.x_label_sum = 'DAC Volt [V]'
        self.full_file_path = full_file_path

        self.parent = parent
        self.sum_plt_data = None

        self.tres_image = None
        self.t_proj_plt_itm = None
        self.tres_plt_item = None
        self.spec_data = None  # spec_data to work on.
        self.storage_data = None  # will not be touched except when gating before saving.
        ''' connect callbacks: '''
        # bundle callbacks:
        self.callbacks = (self.new_data_callback, self.new_track_callback, self.save_callback)
        self.subscribe_to_main()

        ''' sum related '''
        self.add_sum_plot()

        self.sum_scaler = [0]  # list of scalers to add
        self.sum_track = -1  # int, for selecting the track which will be added. -1 for all

        self.sum_list = ['add all', 'manual']
        self.comboBox_select_sum_for_pmts.addItems(self.sum_list)
        self.comboBox_select_sum_for_pmts.currentIndexChanged.connect(self.sum_scaler_changed)
        self.comboBox_select_sum_for_pmts.currentIndexChanged.emit(0)

        self.lineEdit_arith_scaler_input.textChanged.connect(self.sum_scaler_lineedit_changed)

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
        self.checkBox_sum_zoom.stateChanged.connect(self.set_sum_zoomable)
        self.checkBox_volt_line_zoom.stateChanged.connect(self.set_v_proj_zoomabel)

        self.checkBox_sum_zoom.setCheckState(QtCore.Qt.Checked)
        self.checkBox_volt_line_zoom.setCheckState(QtCore.Qt.Checked)

        ''' setup window size: '''
        self.resize(1024, 768)
        size_plt, size_table = self.splitter.sizes()
        sum_size = size_plt + size_table
        self.splitter.setSizes([sum_size * 9 // 10, sum_size // 10])

    '''setting up the plots (no data etc. written) '''

    def add_sum_plot(self):
        self.sum_wid, self.sum_plt_itm = Pg.create_x_y_widget()
        self.sum_plot_layout = QtWidgets.QVBoxLayout()
        self.sum_plot_layout.addWidget(self.sum_wid)
        self.widget_inner_sum_plot.setLayout(self.sum_plot_layout)
        # ??? achsen beschrft...

    def add_time_resolved_plot(self):
        print('adding tres')
        self.tres_v_layout = QtWidgets.QGridLayout()
        self.tres_widg, self.tres_plt_item = Pg.create_image_view()
        self.tres_v_layout.addWidget(self.tres_widg, 0, 0)
        self.v_proj_wid, self.v_proj_plt_itm = Pg.create_x_y_widget()
        self.v_proj_plt_itm.showAxis('right')
        self.sum_proj_view_box = Pg.create_viewbox()
        self.v_proj_plt_itm.scene().addItem(self.sum_proj_view_box)
        self.v_proj_plt_itm.getAxis('right').linkToView(self.sum_proj_view_box)
        self.sum_proj_view_box.setXLink(self.v_proj_plt_itm)
        self.v_proj_plt_itm.getAxis('right').setLabel('sum', color='#0000ff')
        self.updateViews()
        self.v_proj_plt_itm.vb.sigResized.connect(self.updateViews)

        self.t_proj_wid, self.t_proj_plt_itm = Pg.create_x_y_widget()
        self.v_proj_plt_itm.setXLink(self.tres_plt_item)
        self.t_proj_plt_itm.setYLink(self.tres_plt_item)
        self.tres_v_layout.addWidget(self.v_proj_wid, 1, 0)
        self.tres_v_layout.addWidget(self.t_proj_wid, 0, 1)
        self.pushButton_save_after_scan.clicked.connect(self.save)
        self.widget_tres_plot.setLayout(self.tres_v_layout)
        print('adding tres complete')

    def updateViews(self):
        self.sum_proj_view_box.setGeometry(self.v_proj_plt_itm.vb.sceneBoundingRect())
        self.sum_proj_view_box.linkedViewChanged(self.v_proj_plt_itm.vb, self.sum_proj_view_box.XAxis)

    def setup_new_track(self, rcv_tpl):
        self.tres_sel_tr_ind, self.tres_sel_tr_name = rcv_tpl[0]
        self.tres_sel_sc_ind, self.tres_sel_sc_name = rcv_tpl[1]
        self.reset_plots(True)

    def new_data(self, spec_data):
        """
        call this to pass a new dataset to the gui.
        """
        try:
            gates = None
            if self.spec_data is not None:
                gates = deepcopy(self.spec_data.softw_gates)
            self.spec_data = deepcopy(spec_data)
            self.storage_data = deepcopy(spec_data)
            if gates is not None:
                self.spec_data.softw_gates = gates
                self.storage_data.softw_gates = gates
            self.sum_scaler_changed(0)
            self.update_gates_list()
            self.rebin_data(self.spec_data.softBinWidth_ns[self.tres_sel_tr_ind])

            # self.update_all_plots(self.spec_data)  # plotting already done within self.rebin_data
        except Exception as e:
            print('error in liveplotterui while receiving new data: ', e)

    def set_sum_zoomable(self, state):
        # ???
        # if state == QtCore.Qt.Unchecked:
        #     self.tres_axes['sum_proj'].set_navigate(False)
        # elif state == QtCore.Qt.Checked:
        #     self.tres_axes['sum_proj'].set_navigate(True)
        pass

    def set_v_proj_zoomabel(self, state):
        # ???
        # if state == QtCore.Qt.Unchecked:
        #     self.tres_axes['v_proj'].set_navigate(False)
        # elif state == QtCore.Qt.Checked:
        #     self.tres_axes['v_proj'].set_navigate(True)
        pass

    ''' updating the plots from specdata '''

    def update_all_plots(self, spec_data, draw=True):
        self.update_sum_plot(spec_data, draw)
        self.update_tres_data(spec_data, draw)
        self.update_projections(spec_data, draw)

    def update_sum_plot(self, spec_data, draw=True):
        self.sum_x, self.sum_y, self.sum_err = spec_data.getArithSpec(self.sum_scaler, self.sum_track)
        if self.sum_plt_data is None:
            self.sum_plt_data = self.sum_plt_itm.plot(self.sum_x, self.sum_y)
        else:
            self.sum_plt_data.setData(self.sum_x, self.sum_y)
            # ??? autorange
            # self.sum_ax.relim()
            # self.sum_ax.set_xmargin(0.01)
            # self.sum_ax.set_ymargin(0.01)
            # self.sum_ax.autoscale(enable=True, axis='both', tight=False)
            # self.sum_ax.set_ylim(bottom=0)
            # self.update_projections(spec_data)
            # if draw:
            #     self.sum_canv.draw()

    def update_tres_data(self, spec_data, draw=True):
        try:
            gates = self.spec_data.softw_gates[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
            # if self.tres_plt_item is None:  # create image in first plot call.
            #     self.add_time_resolved_plot()
            x_range = (float(np.min(spec_data.x[self.tres_sel_tr_ind])), np.max(spec_data.x[self.tres_sel_tr_ind]))
            x_scale = np.mean(np.ediff1d(spec_data.x[self.tres_sel_tr_ind]))
            y_range = (np.min(spec_data.t[self.tres_sel_tr_ind]), np.max(spec_data.t[self.tres_sel_tr_ind]))
            y_scale = np.mean(np.ediff1d(spec_data.t[self.tres_sel_tr_ind]))
            self.tres_widg.setImage(spec_data.time_res[self.tres_sel_tr_ind][self.tres_sel_sc_ind],
                                    pos=[x_range[0] - abs(0.5 * x_scale),
                                         y_range[0] - abs(0.5 * y_scale)],
                                    scale=[x_scale, y_scale])
            self.tres_plt_item.setAspectLocked(False)
            self.tres_plt_item.setRange(xRange=x_range, yRange=y_range)
            self.tres_widg.updateImage()
            self.tres_widg.updateNorm()
            # print(x_range, x_scale, y_range, y_scale)
            # if draw:
            # self.tres_axes['image'].draw_artist(self.tres_image)
            # self.draw_trs()
        except Exception as e:
            print('error, while plotting time resolved, this happened: ', e)

    def rect_select_gates(self, eclick, erelease):
        """
        is called via left/rigth click & release events, connection see in start()
        will pass the coordinates of the selected area to self.update_gate_ind()
        """
        # ???
        # try:
        #     if self.last_event != eclick:
        #         # somehow the event is emitted multiple times
        #         # to prevent multiple executions, it stored and compared.
        #         self.last_event = eclick
        #         volt_1, time_1 = eclick.xdata, eclick.ydata
        #         volt_2, volt_3 = erelease.xdata, erelease.ydata
        #         volt_1, volt_2 = sorted((volt_1, volt_2))
        #         time_1, volt_3 = sorted((time_1, volt_3))
        #         gates_list = [volt_1, volt_2, time_1, volt_3]
        #         # print(datetime.datetime.now().strftime('%H:%M:%S'), ' gate select yields:', gates_list)
        #         self.spec_data.softw_gates[self.tres_sel_tr_ind][self.tres_sel_sc_ind] = gates_list
        #         self.gate_data(self.spec_data)
        #     # else:
        #     #     print('this very event already has happened')
        # except Exception as e:
        #     print('while setting the gates this happened: ', e)
        pass

    def update_projections(self, spec_data, draw=True):
        try:
            t_proj_x = spec_data.t_proj[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
            t_proj_y = spec_data.t[self.tres_sel_tr_ind]
            v_proj_x = spec_data.x[self.tres_sel_tr_ind]
            v_proj_y = spec_data.cts[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
            gates = self.spec_data.softw_gates[self.tres_sel_tr_ind][self.tres_sel_sc_ind]
            if self.t_proj_plt is None:
                self.t_proj_plt = self.t_proj_plt_itm.plot(t_proj_x, t_proj_y)
                self.v_proj_plt = self.v_proj_plt_itm.plot(v_proj_x, v_proj_y)
                self.sum_proj_plt = Pg.create_plot_data_item(self.sum_x, self.sum_y)
                self.sum_proj_view_box.addItem(self.sum_proj_plt)
                # ??? lines for gates, color of sum?
                # self.sum_line_proj = MPLPlotter.line2d(self.sum_line.get_xdata(), self.sum_line.get_ydata(), 'blue')
                # self.tres_axes['t_proj'].add_line(self.t_proj_plt_itm)
                # self.tres_axes['v_proj'].add_line(self.v_proj_plt_itm)
                # self.tres_axes['sum_proj'].add_line(self.sum_line_proj)
                # gate_color = '0.75'
                # self.v_min_line = self.tres_axes['v_proj'].axvline(gates[0], color=gate_color)
                # self.v_max_line = self.tres_axes['v_proj'].axvline(gates[1], color=gate_color)
                # self.t_min_line = self.tres_axes['t_proj'].axhline(gates[2], color=gate_color)
                # self.t_max_line = self.tres_axes['t_proj'].axhline(gates[3], color=gate_color)
            else:
                # print('updating proj')
                self.t_proj_plt.setData(t_proj_x, t_proj_y)
                self.v_proj_plt.setData(v_proj_x, v_proj_y)
                self.sum_proj_plt.setData(self.sum_x, self.sum_y)  # does this work?
                # self.t_proj_plt_itm.set_xdata(t_proj_x)
                # self.v_proj_plt_itm.set_ydata(v_proj_y)
                # self.v_min_line.set_xdata([gates[0], gates[0]])
                # self.v_max_line.set_xdata([gates[1], gates[1]])
                # self.t_min_line.set_ydata([gates[2], gates[2]])
                # self.t_max_line.set_ydata([gates[3], gates[3]])
                # self.sum_proj_plt.set_data(self.sum_line.get_xdata(), self.sum_line.get_ydata())
                # for ax_key in [('t_proj', 'y'), ('v_proj', 'x'), ('sum_proj', 'x')]:
                #     self.tres_axes[ax_key[0]].relim()
                #     if ax_key[1] == 'y':
                #         self.tres_axes[ax_key[0]].set_xmargin(0.05)
                #         self.tres_axes[ax_key[0]].autoscale(enable=True, axis='x', tight=False)
                #         self.tres_axes[ax_key[0]].set_xlim(left=0)
                #     else:
                #         self.tres_axes[ax_key[0]].set_ymargin(0.05)
                #         self.tres_axes[ax_key[0]].autoscale(enable=True, axis='y', tight=False)
                #         self.tres_axes[ax_key[0]].set_ylim(bottom=0)
                # if draw:
                #     # self.tres_axes['t_proj'].draw_artist(self.t_proj_line)
                #     # self.tres_axes['v_proj'].draw_artist(self.v_proj_line)
                #     # self.tres_axes['sum_proj'].draw_artist(self.sum_line_proj)
                #     self.draw_trs()

        except Exception as e:
            print('error, while plotting projection, this happened: ', e)

    def draw_trs(self):
        # ??? delete?
        # self.tres_canv.draw()
        pass

    def reset_plots(self, clear_bool=False):
        # ??? delete
        # if clear_bool:
        #     for key, ax in self.tres_axes.items():
        #         ax.clear()
        # self.t_proj_plt_itm = None
        # self.tres_image = None
        pass

    ''' buttons, comboboxes and listwidgets: '''

    def sum_scaler_changed(self, index=None):
        """
        this will set the self.sum_scaler list to the values set in the gui.
        :param index: int, index of the element in the combobox
        """
        if index is None:
            index = self.comboBox_select_sum_for_pmts.currentIndex()
        if index == 0:
            if self.spec_data is not None:
                self.sum_scaler = self.spec_data.active_pmt_list[0]
                self.sum_track = -1
                self.lineEdit_arith_scaler_input.setText(str(self.sum_scaler))
            self.lineEdit_arith_scaler_input.setDisabled(True)
        elif index == 1:
            # self.sum_scaler = self.valid_scaler_input
            self.lineEdit_arith_scaler_input.setDisabled(False)

    def sum_scaler_lineedit_changed(self, text):
        """
        this will check if the input text in the line edit will result is a list
         and only contain valid scaler entries.
        :param text:str, in the form of type [+i, -j, +k], resulting in s[i]-s[j]+s[k]
        :return:
        """
        try:
            hopefully_list = ast.literal_eval(text)
            if isinstance(hopefully_list, list):
                isinteger = len(hopefully_list) > 0
                for scaler in hopefully_list:
                    isinteger = isinteger and isinstance(scaler, int) and abs(scaler) < self.spec_data.nrScalers[0]
                if isinteger:
                    self.sum_scaler = hopefully_list
                    self.label_arith_scaler_set.setText(str(hopefully_list))
                    self.update_sum_plot(self.spec_data)
                    self.update_projections(self.spec_data)
        except Exception as e:
            print(e)

    ''' gating: '''

    def gate_data(self, spec_data, plot_bool=True):
        spec_data = TiTs.gate_specdata(spec_data)
        self.write_all_gates_to_table(spec_data)
        # self.reset_table()
        # self.update_gates_list()
        if plot_bool:
            self.update_all_plots(self.spec_data)

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
            self.reset_plots(True)
            self.update_tres_data(self.spec_data, True)
            self.update_projections(self.spec_data, True)

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
                    if self.tres_sel_tr_name == tr_name[5:] and self.tres_sel_sc_ind == pmt_ind:
                        # print('this will be set to true: ', tr_name, pmt_ind)
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
        logging.debug('livePlotUi, row count after reset of table: ', self.tableWidget_gates.rowCount())

    ''' saving '''

    def save(self, pipedata_dict=None):
        im_path = self.full_file_path.split('.')[0] + '_' + str(self.tres_sel_tr_name) + '_' + str(
            self.tres_sel_sc_name) + '.png'
        self.tres_fig.savefig(im_path)

        if isinstance(pipedata_dict, bool):  # when pressing on save
            pipedata_dict = None
        if pipedata_dict is not None:
            self.pipedata_dict = pipedata_dict
        if self.pipedata_dict is not None:
            self.storage_data.softw_gates = self.extract_all_gates_from_gui()
            self.gate_data(self.storage_data, False)
            FileHandl.save_spec_data(self.storage_data, self.pipedata_dict)
        else:
            print('could not save data, because it was not save from the scan process yet.')

    ''' closing '''

    def closeEvent(self, *args, **kwargs):
        self.unsubscribe_from_main()
        if self.parent is not None:
            self.parent.close_live_plot_win()

    ''' subscription to main '''

    def subscribe_to_main(self):
        self.save_callback.connect(self.save)
        self.new_track_callback.connect(self.setup_new_track)
        self.new_data_callback.connect(self.new_data)
        self.progress_callback.connect(self.handle_progress_dict)
        Cfg._main_instance.gui_live_plot_subscribe(self.callbacks, self.progress_callback)

    def unsubscribe_from_main(self):
        Cfg._main_instance.gui_live_plot_unsubscribe()

    ''' rebinning '''

    def rebin_data(self, rebin_factor_ns=None):
        if rebin_factor_ns is None:
            if self.active_initial_scan_dict is not None:
                rebin_factor_ns = self.spec_data.softBinWidth_ns[self.tres_sel_tr_ind]
        rebin_factor_ns = rebin_factor_ns // 10 * 10
        self.spinBox.blockSignals(True)
        self.spinBox.setValue(rebin_factor_ns)
        self.spinBox.blockSignals(False)
        if self.storage_data is not None:
            logging.debug('rebinning data to bins of  %s' % rebin_factor_ns)
            rebin_track = -1 if self.checkBox.isChecked() else self.tres_sel_tr_ind
            # gates = deepcopy(self.spec_data.softw_gates)
            self.storage_data.softw_gates = self.extract_all_gates_from_gui()
            self.spec_data = Form.time_rebin_all_spec_data(self.storage_data, rebin_factor_ns, rebin_track)
            print('softw_binwidth of full data: %s, of rebinned data: %s '
                  % (self.storage_data.softBinWidth_ns, self.spec_data.softBinWidth_ns))
            try:
                self.gate_data(self.spec_data, False)
                self.reset_plots(True)
                self.update_all_plots(self.spec_data)

            except Exception as e:
                print('error while gating: ', e)
                # self.update_all_plots(self.spec_data)

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
        if self.active_file != progress_dict_from_main['activeFile']:
            self.active_file = progress_dict_from_main['activeFile']
            self.setWindowTitle('plot:  %s' % self.active_file)
        self.active_initial_scan_dict = Cfg._main_instance.scan_pars[self.active_iso]
        self.active_track_name = progress_dict_from_main['trackName']

# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     ui = TRSLivePlotWindowUi()
#     ui.show()
#     app.exec()
