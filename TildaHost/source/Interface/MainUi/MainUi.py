"""

Created on '07.05.2015'

@author:'simkaufm'

"""

""" Guis: """
from Interface.MainUi.Ui_Main import Ui_TildaMainWindow
from Interface.VersionUi.VersionUi import VersionUi
from Interface.ScanControlUi.ScanControlUi import ScanControlUi
from Interface.VoltageMeasurementConfigUi.VoltMeasConfUi import VoltMeasConfUi
from Interface.PostAccControlUi.PostAccControlUi import PostAccControlUi
from Interface.SimpleCounter.SimpleCounterDialogUi import SimpleCounterDialogUi

import Application.Config as Cfg

import logging
import os
from PyQt5 import QtWidgets


class MainUi(QtWidgets.QMainWindow, Ui_TildaMainWindow):
    def __init__(self):
        super(MainUi, self).__init__()
        self.setupUi(self)

        self.act_scan_wins = []  # list of active scan windows
        self.post_acc_win = None  # only one active post acceleration window
        self.measure_voltage_win = None

        self.actionWorking_directory.triggered.connect(self.choose_working_dir)
        self.actionVersion.triggered.connect(self.open_version_win)
        self.actionScan_Control.triggered.connect(self.open_scan_ctrl_win)
        self.actionVoltage_Measurement.triggered.connect(self.open_volt_meas_win)
        self.actionPost_acceleration_power_supply_control.triggered.connect(self.open_post_acc_win)
        self.actionSimple_Counter.triggered.connect(self.simple_counter)
        self.show()

    def gui_cyclic(self):
        """
        cyclic function for the Gui
        """
        self.update_status()
        self.update_scan_wins()
        self.update_post_acc_win()

    def update_post_acc_win(self):
        """
        keep it short, will be called in cyclic()
        """
        if self.post_acc_win is not None:
            self.post_acc_win.post_acc_cyclic()

    def update_status(self):
        """
        keep it short, will be called in cyclic()
        """
        self.label_workdir_set.setText(str(Cfg._main_instance.working_directory))
        self.label_main_status.setText(Cfg._main_instance.m_state[0])
        self.label_database.setText(Cfg._main_instance.database)

    def update_scan_wins(self):
        """
        keep it short, will be called in cyclic()
        """
        for w in self.act_scan_wins:
            w.enable_go(Cfg._main_instance.m_state == 'idle')
            # disable go if not in idle state

    def choose_working_dir(self):
        """ will open a modal file dialog and set all workingdirectories of the pipeline to the chosen folder """
        workdir = QtWidgets.QFileDialog.getExistingDirectory(QtWidgets.QFileDialog(),
            'choose working directory', os.path.expanduser('~'))
        Cfg._main_instance.work_dir_changed(workdir)
        self.label_workdir_set.setText(str(workdir))

    def open_version_win(self):
        VersionUi()

    def open_scan_ctrl_win(self):
        if Cfg._main_instance.working_directory is None:
            self.choose_working_dir()
        self.act_scan_wins.append(ScanControlUi(self))

    def scan_control_win_closed(self, win_ref):
        self.act_scan_wins.remove(win_ref)

    def open_volt_meas_win(self):
        self.measure_voltage_win = VoltMeasConfUi(self.measure_voltage_pars)

    def close_volt_meas_win(self):
        self.measure_voltage_win = None

    def open_post_acc_win(self):
        self.post_acc_win = PostAccControlUi(self)

    def close_post_acc_win(self):
        self.post_acc_win = None

    def simple_counter(self):
        act_pmt_list, datapoints = self.open_simp_count_dial()
        Cfg._main_instance.start_simple_counter(act_pmt_list, datapoints)
        SimpleCounterDialogUi()  # blocking!

    def open_simp_count_dial(self):
        # open window here in the future, that returns the active pmt list and the plotpoints
        # needed before the pipeline is started
        # could also be realized by altering the simple counter dialog win
        return [0, 1], 600

    def closeEvent(self, *args, **kwargs):
        for win in self.act_scan_wins:
            logging.debug('will close: ' + str(win))
            try:
                win.close()
            except Exception as e:
                logging.error(str(e))
        try:
            if self.post_acc_win is not None:
                self.post_acc_win.close()
        except Exception as e:
            logging.error('error while closing post acceleration win:' + str(e))
        try:
            if self.measure_voltage_win is not None:
                self.measure_voltage_win.close()
        except Exception as e:
            logging.error('error while closing voltage measurement win:' + str(e))