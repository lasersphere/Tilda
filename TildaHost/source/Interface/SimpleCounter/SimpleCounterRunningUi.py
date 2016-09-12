"""
Created on 

@author: simkaufm

Module Description:Interface for the running simple Counter, which will display the current counters and
the currently selected post acceleration device.
"""

import os

from PyQt5 import QtCore
from PyQt5 import QtWidgets

import Application.Config as Cfg
from Interface.SimpleCounter.Ui_simpleCounterRunnning import Ui_SimpleCounterRunning


class SimpleCounterRunningUi(QtWidgets.QMainWindow, Ui_SimpleCounterRunning):
    simple_counter_call_back_signal = QtCore.pyqtSignal(list)

    def __init__(self, main_gui, act_pmts, datapoints):
        super(SimpleCounterRunningUi, self).__init__()

        work_dir_before_setup_ui = os.getcwd()
        os.chdir(os.path.dirname(__file__))  # necessary for the icons to appear
        self.setupUi(self)
        os.chdir(work_dir_before_setup_ui)  # change back

        self.main_gui = main_gui

        self.first_call = True

        self.act_pmts = act_pmts
        self.datapoints = datapoints
        self.names = []
        self.add_scalers_to_gridlayout(act_pmts)

        self.simple_counter_call_back_signal.connect(self.rcv)

        self.pushButton_stop.clicked.connect(self.stop)
        self.pushButton_refresh_post_acc_state.clicked.connect(self.refresh_post_acc_state)
        self.doubleSpinBox.valueChanged.connect(self.set_dac_volt)
        self.comboBox_post_acc_control.currentTextChanged.connect(self.set_post_acc_ctrl)

        Cfg._main_instance.start_simple_counter(act_pmts, datapoints, self.simple_counter_call_back_signal)

        self.show()

    def rcv(self, scaler_liste):
        if self.first_call:
            self.refresh_post_acc_state()
        self.first_call = False
        for i, j in enumerate(scaler_liste):
            if i < len(self.names):
                getattr(self, self.names[i]).display(j)

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

    def set_dac_volt(self):
        volt_dbl = self.doubleSpinBox.value()
        Cfg._main_instance.simple_counter_set_dac_volt(volt_dbl)
        self.label_dac_set_volt.setText(str(volt_dbl))

    def add_scalers_to_gridlayout(self, scalers):
        for i, j in enumerate(scalers):
            try:
                name = 'pmt_' + str(j)
                self.names.append(name)
                label_name = 'label_pmt_' + str(j)
                setattr(self, name, QtWidgets.QLCDNumber(self.centralwidget))
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                sizePolicy.setHeightForWidth(getattr(self, name).sizePolicy().hasHeightForWidth())
                getattr(self, name).setSizePolicy(sizePolicy)
                getattr(self, name).setObjectName(name)
                self.gridLayout_2.addWidget(getattr(self, name), i, 1, 1, 1)
                setattr(self, label_name, QtWidgets.QLabel(self.centralwidget))
                getattr(self, label_name).setObjectName(label_name)
                self.gridLayout_2.addWidget(getattr(self, label_name), i, 0, 1, 1)
                _translate = QtCore.QCoreApplication.translate
                t = _translate('SimpleCounterRunning',
                               "<html><head/><body><p><span style=\" font-size:48pt;\">Ch" +
                               str(j) + "</span></p></body></html>")
                getattr(self, label_name).setText(t)
                getattr(self, name).display(0)
            except Exception as e:
                print(e)