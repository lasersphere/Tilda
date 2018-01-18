"""
Created on 

@author: simkaufm

Module Description:Interface for the running simple Counter, which will display the current counters and
the currently selected post acceleration device.
"""

import functools
import os

import numpy as np
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import Application.Config as Cfg
import PyQtGraphPlotter as Pg
from Interface.SimpleCounter.Ui_simpleCounterRunnning import Ui_SimpleCounterRunning


class SimpleCounterRunningUi(QtWidgets.QMainWindow, Ui_SimpleCounterRunning):
    simple_counter_call_back_signal = QtCore.pyqtSignal(tuple)

    def __init__(self, main_gui, act_pmts, datapoints):
        super(SimpleCounterRunningUi, self).__init__()

        work_dir_before_setup_ui = os.getcwd()
        os.chdir(os.path.dirname(__file__))  # necessary for the icons to appear
        self.setupUi(self)
        os.chdir(work_dir_before_setup_ui)  # change back

        self.main_gui = main_gui

        self.first_call = True

        self.sample_interval = 0.2  # seconds fpga samples at 200 ms

        self.act_pmts = act_pmts
        self.datapoints = datapoints
        self.names = []
        self.elements = []  # list of dicts.
        self.add_scalers_to_gridlayout(act_pmts)
        # self.x_data = np.array([np.arange(0, self.datapoints) for i in self.act_pmts])
        self.x_data = np.zeros((len(self.act_pmts), self.datapoints))
        self.y_data = np.zeros((len(self.act_pmts), self.datapoints))

        self.simple_counter_call_back_signal.connect(self.rcv)

        self.pushButton_stop.clicked.connect(self.stop)
        self.pushButton_refresh_post_acc_state.clicked.connect(self.refresh_post_acc_state)
        self.pushButton_reset_graphs.clicked.connect(self.reset_graphs)
        self.doubleSpinBox.valueChanged.connect(self.set_dac_volt)
        self.comboBox_post_acc_control.currentTextChanged.connect(self.set_post_acc_ctrl)

        Cfg._main_instance.start_simple_counter(act_pmts, datapoints,
                                                self.simple_counter_call_back_signal, self.sample_interval)

        self.show()

    def rcv(self, scaler_liste):
        """ handle new incoming data """
        if self.first_call:
            self.refresh_post_acc_state()
        self.first_call = False
        for i, j in enumerate(scaler_liste[0]):
            last_second_sum = np.sum(j)
            number_of_new_data_points = scaler_liste[1][i]
            if number_of_new_data_points:
                self.elements[i]['widg'].display(last_second_sum)
                self.y_data[i] = np.roll(self.y_data[i], -1)
                self.x_data[i] = np.roll(self.x_data[i], -1)
                self.y_data[i][-1] = last_second_sum
                self.x_data[i] += number_of_new_data_points * self.sample_interval
                self.x_data[i][-1] = 0  # always zero at time 0
                self.update_plot(i, self.x_data[i], self.y_data[i])

    def update_plot(self, indic, xdata, ydata):
        plt_data_item = self.elements[indic].get('plotDataItem', None)
        if plt_data_item is None:
            self.elements[indic]['plotDataItem'] = Pg.create_plot_data_item(xdata, ydata, pen='b')
            self.elements[indic]['pltItem'].addItem(self.elements[indic]['plotDataItem'])
            self.elements[indic]['pltItem'].vb.invertX(True)  # as requested by Christian
        else:
            self.elements[indic]['plotDataItem'].setData(xdata, ydata)

    def stop(self):
        self.close()

    def closeEvent(self, *args, **kwargs):
        Cfg._main_instance.stop_simple_counter()
        self.main_gui.close_simple_counter_win()

    def set_post_acc_ctrl(self, state_name):
        Cfg._main_instance.simple_counter_post_acc(state_name)
        self.refresh_post_acc_state()

    def refresh_post_acc_state(self):
        state_num, state_name = Cfg._main_instance.get_simple_counter_post_acc()
        self.label_post_acc_readback_state.setText(state_name)

    def reset_graphs(self):
        self.x_data = np.zeros((len(self.act_pmts), self.datapoints))
        self.y_data = np.zeros((len(self.act_pmts), self.datapoints))

    def set_dac_volt(self):
        volt_dbl = self.doubleSpinBox.value()
        Cfg._main_instance.simple_counter_set_dac_volt(volt_dbl)
        self.label_dac_set_volt.setText(str(volt_dbl))

    def splitter_was_moved(self, caller_ind, pos, ind):
        for pl_ind, pl_dict in enumerate(self.elements):
            if caller_ind != pl_ind:
                pl_dict['splitter'].blockSignals(True)
                pl_dict['splitter'].moveSplitter(pos, ind)
                pl_dict['splitter'].blockSignals(False)

    def add_scalers_to_gridlayout(self, scalers):
        for i, j in enumerate(scalers):
            try:
                name = 'pmt_' + str(j)
                label_name = 'label_pmt_' + str(j)
                splitter = QtWidgets.QSplitter(self.centralwidget)
                splitter.setOrientation(QtCore.Qt.Horizontal)
                label = QtWidgets.QLabel(splitter)
                label.setObjectName(label_name)
                widg = QtWidgets.QLCDNumber(splitter)
                widg.setDigitCount(6)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                                   QtWidgets.QSizePolicy.MinimumExpanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                widg.setMinimumWidth(self.width()/5)
                sizePolicy.setHeightForWidth(widg.sizePolicy().hasHeightForWidth())
                widg.setSizePolicy(sizePolicy)
                widg.setObjectName(name)

                # self.gridLayout_2.addWidget(label, i, 0, 1, 1)
                # self.gridLayout_2.addWidget(widg, i, 1, 1, 1)
                plt_widg, plt_item = Pg.create_x_y_widget(x_label='time [s]', y_label='cts')
                splitter.insertWidget(2, plt_widg)
                splitter.splitterMoved.connect(functools.partial(self.splitter_was_moved, i))
                self.gridLayout_2.addWidget(splitter, i, 0, 1, 1)
                _translate = QtCore.QCoreApplication.translate
                t = _translate('SimpleCounterRunning',
                               "<html><head/><body><p><span style=\" font-size:48pt;\">Ch" +
                               str(j) + "</span></p></body></html>")
                label.setText(t)
                widg.display(0)
                sc_dict = {'name': name, 'label': label, 'widg': widg,
                           'plotWid': plt_widg, 'pltItem': plt_item, 'splitter': splitter}
                self.elements.append(sc_dict)
            except Exception as e:
                print(e)