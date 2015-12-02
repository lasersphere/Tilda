"""

Created on '07.05.2015'

@author:'simkaufm'

"""



from Interface.MainUi.Ui_Main import Ui_TildaMainWindow
from Interface.VersionUi.VersionUi import VersionUi
from Interface.TrackParUi.TrackUi import TrackUi
import Application.Config as Cfg

from copy import deepcopy
import threading
import time
import logging
import os
from PyQt5 import QtWidgets


class MainUi(QtWidgets.QMainWindow, Ui_TildaMainWindow):
    def __init__(self):
        super(MainUi, self).__init__()
        self.setupUi(self)

        self.actionWorking_directory.triggered.connect(self.choose_working_dir)
        self.actionVersion.triggered.connect(self.open_version_win)
        self.actionScan_Control.triggered.connect(self.open_scan_ctrl_win)
        self.actionVoltage_Measurement.triggered.connect(self.open_volt_meas_win)
        self.actionPost_acceleration_power_supply_control.triggered.connect(self.open_post_acc_win)
        self.actionSimple_Counter.triggered.connect(self.simple_counter)
        self.show()

    def choose_working_dir(self):
        """ will open a modal file dialog and set all workingdirectories of the pipeline to the chosen folder """
        workdir = Cfg._main_instance.open_work_dir_win()
        self.label_workdir_set.setText(str(workdir))

    def open_version_win(self):
        VersionUi()

    def open_scan_ctrl_win(self):
        Cfg._main_instance.open_scan_control_win()

    def open_volt_meas_win(self):
        Cfg._main_instance.open_volt_meas_win()

    def open_post_acc_win(self):
        Cfg._main_instance.open_post_acc_win()

    def simple_counter(self):
        Cfg._main_instance.start_simple_counter()

    def closeEvent(self, *args, **kwargs):
        Cfg._main_instance.close_main_win()