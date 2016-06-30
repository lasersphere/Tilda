"""
Created on 

@author: simkaufm

Module Description:
"""
import sys
import ast
import logging
import numpy as np
import MPLPlotter
from copy import deepcopy
from PyQt5 import QtWidgets, Qt, QtCore
from Interface.LiveDataPlottingUi.Ui_LiveDataPlotting import Ui_MainWindow_LiveDataPlotting
import TildaTools as TiTs


class TRSLivePlotWindowUi(QtWidgets.QMainWindow, Ui_MainWindow_LiveDataPlotting):
    def __init__(self, full_file_path, parent_node):
        super(TRSLivePlotWindowUi, self).__init__()

        self.setupUi(self)
        self.show()
        self.setWindowTitle('plot win:     ' + full_file_path)
        self.parent_node = parent_node
        print('opened ', full_file_path)

        self.x_label_sum = 'DAC Volt'
        self.full_file_path = full_file_path

        self.spec_data = None

        ''' sum related '''
        self.add_sum_plot()

        self.sum_line = None
        self.sum_scaler = [0]   # list of scalers to add
        self.sum_track = -1  # int, for selecting the track which will be added. -1 for all

        self.sum_list = ['add all', 'manual']
        self.comboBox_select_sum_for_pmts.addItems(self.sum_list)
        self.comboBox_select_sum_for_pmts.currentIndexChanged.connect(self.sum_scaler_changed)
        self.comboBox_select_sum_for_pmts.currentIndexChanged.emit(0)

        self.lineEdit_arith_scaler_input.textChanged.connect(self.sum_scaler_lineedit_changed)

        ''' time resolved related: '''
        self.add_time_resolved_plot()
        self.tres_sel_tr = 0  # currently observed track in time resolved spectra
        self.tres_sel_sc = 0  # currently observed scaler in time resolved spectra

        self.tres_image = None
        self.t_proj_line = None
        self.resize(800, 600)

        self.tableWidget_gates.itemClicked.connect(self.handle_item_clicked)
        self.tableWidget_gates.itemChanged.connect(self.handle_gate_table_change)

    '''setting up the plots (no data etc. written) '''
    def add_sum_plot(self):
        self.sum_fig, self.sum_canv, self.sum_toolbar= MPLPlotter.create_figure_widget(self)
        self.sum_plot_layout = QtWidgets.QVBoxLayout()
        self.sum_plot_layout.addWidget(self.sum_toolbar)
        self.sum_plot_layout.addWidget(self.sum_canv)
        self.widget_inner_sum_plot.setLayout(self.sum_plot_layout)
        self.sum_ax = self.sum_fig.add_subplot(111)
        self.sum_ax.set_ylabel('counts')
        self.sum_ax.set_xlabel(self.x_label_sum)
        MPLPlotter.tight_layout()

    def add_time_resolved_plot(self):
        self.tres_v_layout = QtWidgets.QVBoxLayout()
        self.tres_fig, self.tres_axes, self.tres_canv, self.tres_toolbar = MPLPlotter.setup_image_widget(
            self.widget_tres_plot)
        self.tres_v_layout.addWidget(self.tres_toolbar)
        self.tres_v_layout.addWidget(self.tres_canv)
        self.pushButton_save_after_scan.clicked.connect(self.parent_node.save)
        self.widget_tres_plot.setLayout(self.tres_v_layout)

    def new_data(self, spec_data, reset_bool):
        """
        call this to pass a new dataset to the gui.
        """
        try:
            self.spec_data = deepcopy(spec_data)
            self.sum_scaler_changed(0)
            self.update_gates_list()
            self.gate_data(self.spec_data, plot_bool=False)
            self.update_all_plots(self.spec_data, reset=reset_bool)
        except Exception as e:
            print('error in liveplotterui while receiving new data: ', e)

    ''' updating the plots from specdata '''
    def update_all_plots(self, spec_data, draw=True, reset=False):
        if reset:
            self.reset_plots(True)
            self.reset_table()
        self.update_sum_plot(spec_data, draw)
        self.update_tres_data(spec_data, draw)
        self.update_projections(spec_data, draw)

    def update_sum_plot(self, spec_data, draw=True):
        x, y, err = spec_data.getArithSpec(self.sum_scaler, self.sum_track)
        if self.sum_line is None:
            self.sum_line = MPLPlotter.line2d(x, y, 'blue')
            self.sum_ax.add_line(self.sum_line)
        else:
            self.sum_line.set_ydata(y)
        self.sum_ax.relim()
        self.sum_ax.set_xmargin(0.01)
        self.sum_ax.set_ymargin(0.01)
        self.sum_ax.autoscale(enable=True, axis='both', tight=False)
        self.sum_ax.set_ylim(bottom=0)
        # self.update_projections(spec_data)
        if draw:
            self.sum_canv.draw()

    def update_tres_data(self, spec_data, draw=True):
        try:
            if self.tres_image is None:  # create image in first plot call.
                self.tres_image, self.tres_colorbar = MPLPlotter.configure_image_plot_widget(
                    self.tres_fig, self.tres_axes['image'], self.tres_axes['colorbar'],
                    spec_data.x[self.tres_sel_tr], spec_data.t[self.tres_sel_tr])
            self.tres_image.set_data(np.transpose(spec_data.time_res[self.tres_sel_tr][self.tres_sel_sc]))
            self.tres_colorbar.set_clim(0, np.nanmax(spec_data.time_res[self.tres_sel_tr][self.tres_sel_sc]))
            self.tres_colorbar.update_normal(self.tres_image)
            if draw:
                # self.tres_axes['image'].draw_artist(self.tres_image)
                self.draw_trs()
        except Exception as e:
            print('error, while plotting time resolved, this happened: ', e)

    def update_projections(self, spec_data, draw=True):
        try:
            t_proj_x = spec_data.t_proj[self.tres_sel_tr][self.tres_sel_sc]
            t_proj_y = spec_data.t[self.tres_sel_tr]
            v_proj_x = spec_data.x[self.tres_sel_tr]
            v_proj_y = spec_data.cts[self.tres_sel_tr][self.tres_sel_sc]
            if self.t_proj_line is None:
                self.v_proj_line, self.t_proj_line = MPLPlotter.setup_projection_widget(
                    self.tres_axes, t_proj_y, v_proj_x, x_label=self.x_label_sum)
                self.sum_line_proj = MPLPlotter.line2d(self.sum_line.get_xdata(), self.sum_line.get_ydata(), 'blue')
                self.tres_axes['t_proj'].add_line(self.t_proj_line)
                self.tres_axes['v_proj'].add_line(self.v_proj_line)
                self.tres_axes['sum_proj'].add_line(self.sum_line_proj)
            self.t_proj_line.set_xdata(t_proj_x)
            self.v_proj_line.set_ydata(v_proj_y)
            self.sum_line_proj.set_data(self.sum_line.get_xdata(), self.sum_line.get_ydata())
            for ax_key in [('t_proj', 'y'), ('v_proj', 'x'), ('sum_proj', 'x')]:
                self.tres_axes[ax_key[0]].relim()
                if ax_key[1] == 'y':
                    self.tres_axes[ax_key[0]].set_xmargin(0.05)
                    self.tres_axes[ax_key[0]].autoscale(enable=True, axis='x', tight=False)
                    self.tres_axes[ax_key[0]].set_xlim(left=0)
                else:
                    self.tres_axes[ax_key[0]].set_ymargin(0.05)
                    self.tres_axes[ax_key[0]].autoscale(enable=True, axis='y', tight=False)
                    self.tres_axes[ax_key[0]].set_ylim(bottom=0)
            if draw:
                # self.tres_axes['t_proj'].draw_artist(self.t_proj_line)
                # self.tres_axes['v_proj'].draw_artist(self.v_proj_line)
                # self.tres_axes['sum_proj'].draw_artist(self.sum_line_proj)
                self.draw_trs()

        except Exception as e:
            print('error, while plotting projection, this happened: ', e)

    def draw_trs(self):
        self.tres_canv.draw()

    def reset_plots(self, clear_bool=False):
        if clear_bool:
            for key, ax in self.tres_axes.items():
                ax.clear()
        self.t_proj_line = None
        self.tres_image = None

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
        self.spec_data = TiTs.gate_specdata(spec_data)
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
            currently_selected = self.find_one_scaler_track('track' + str(self.tres_sel_tr), self.tres_sel_sc)
            curr_checkb_item = self.tableWidget_gates.item(currently_selected.row(), 6)
            curr_checkb_item.setCheckState(QtCore.Qt.Unchecked)
            self.tres_sel_tr = int(self.tableWidget_gates.item(item.row(), 0).text()[5:])
            self.tres_sel_sc = int(self.tableWidget_gates.item(item.row(), 1).text())
            # print('new scaler, track: ', self.tres_sel_tr, self.tres_sel_sc)
            self.reset_plots(True)
            self.update_tres_data(self.spec_data, True)
            self.update_projections(self.spec_data, True)

    def update_gates_list(self):
        """
        read all software gate entries from the specdata and fill it to the displaying table.
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
                    if self.tres_sel_tr == int(tr_name[5:]) and self.tres_sel_sc == pmt_ind:
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

    def extract_one_gate_from_gui(self, tr, sc):
        item = self.find_one_scaler_track(tr, sc)
        gate_lis = []
        for i in range(2, self.tableWidget_gates.columnCount() - 1):
            gate_lis.append(float(self.tableWidget_gates.item(item.row(), i).text()))
        return gate_lis

    def insert_one_gate_to_gui(self, tr, sc, liste):
        item = self.find_one_scaler_track(tr, sc)
        for i in range(2, self.tableWidget_gates.columnCount() - 1):
            val = liste[i - 2]
            new_item = QtWidgets.QTableWidgetItem()
            new_item.setData(QtCore.Qt.EditRole, val)
            self.tableWidget_gates.setItem(item.row(), i, new_item)

    def find_one_scaler_track(self, tr, sc):
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
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     ui = TRSLivePlotWindowUi()
#     ui.show()
#     app.exec()
