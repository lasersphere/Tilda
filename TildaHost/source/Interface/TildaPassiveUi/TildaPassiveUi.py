"""
Created on 13/04/2016

@author: sikaufma

Module Description:

Control of the Userinterface which is used
 for controlling the Tilda Passive Mode.

"""

import datetime
import logging

import numpy
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import Application.Config as Cfg
import Service.FileOperations.FolderAndFileHandling as FileHandl
from Interface.TildaPassiveUi.Ui_TildaPassive import Ui_TildaPassiveMainWindow


class TildaPassiveUi(QtWidgets.QMainWindow, Ui_TildaPassiveMainWindow):
    tipa_raw_data_callback = QtCore.pyqtSignal(numpy.ndarray)
    tipa_status_callback = QtCore.pyqtSignal(int)
    tipa_steps_scans_callback = QtCore.pyqtSignal(dict)

    def __init__(self, main_ui):
        super(TildaPassiveUi, self).__init__()
        self.setupUi(self)
        self.settings_path = './Interface/TildaPassiveUi/TildaPassiveSettings.txt'
        self.setWindowTitle('Tilda Passive Control Window')

        self.main_ui = main_ui

        self.pushButton_save_settings.clicked.connect(self.save_settings)
        self.pushButton_start.clicked.connect(self.start_scan)
        self.pushButton_stop.clicked.connect(self.stop_scan)

        self.tipa_raw_data_callback.connect(self.rcv_raw_data)
        self.tipa_status_callback.connect(self.rcv_status)
        self.tipa_steps_scans_callback.connect(self.rcv_steps_scans)

        self.load_settings()
        self.rcv_status(-1)

        self.show()

    def rcv_raw_data(self, raw_data_list):
        """
        raw data coming from the pipeline will be received as a list of raw elements here.
        """
        rcv_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cOut.appendPlainText(rcv_time + '\t' + str(raw_data_list))

    def rcv_status(self, status_num):
        if status_num == -1:  # not initialized
            self.pushButton_stop.setDisabled(True)
            self.pushButton_start.setEnabled(True)
            self.doubleSpinBox_delay.setDisabled(False)
            self.doubleSpinBox_num_of_bins.setDisabled(False)
            self.label_status.setStyleSheet("QLabel {background-color : light gray; color : black;}")
            self.label_status.setText('not initialized, click Start')
        elif status_num == 0:  # idle = init
            self.pushButton_start.setDisabled(True)
            self.label_status.setStyleSheet("QLabel {background-color : light green; color : black;}")
            self.label_status.setText('idle')
            self.doubleSpinBox_delay.setDisabled(True)
            self.doubleSpinBox_num_of_bins.setDisabled(True)
        elif status_num == 1:  # scanning
            self.pushButton_start.setDisabled(True)
            self.pushButton_stop.setEnabled(True)
            self.doubleSpinBox_delay.setDisabled(True)
            self.doubleSpinBox_num_of_bins.setDisabled(True)
            self.label_status.setStyleSheet("QLabel {background-color : green; color : black;}")
            self.label_status.setText('scanning')
        elif status_num == 2:  # error
            self.pushButton_stop.setDisabled(True)
            self.pushButton_start.setEnabled(True)
            self.doubleSpinBox_delay.setDisabled(False)
            self.doubleSpinBox_num_of_bins.setDisabled(False)
            self.label_status.setStyleSheet("QLabel {background-color : red; color : white;}")
            self.label_status.setText('error, please restart')
        elif status_num == 3:  # mcp inactive?
            self.pushButton_start.setDisabled(True)
            self.pushButton_stop.setEnabled(True)
            self.doubleSpinBox_delay.setDisabled(True)
            self.doubleSpinBox_num_of_bins.setDisabled(True)
            self.label_status.setStyleSheet("QLabel {background-color : yellow; color : black;}")
            self.label_status.setText('scanning\n but no events since 5s\n MCP running? save?')

    def rcv_steps_scans(self, step_scans_dict):
        steps_str = str(step_scans_dict.get('nOfCompletedSteps', None))
        scans_str = str(step_scans_dict.get('nOfStartedScans', None))
        self.label_acq_steps_total.setText(steps_str)
        self.label_acq_scans_total.setText(scans_str)

    def load_settings(self):
        try:
            data = FileHandl.loadPickle(self.settings_path)
            self.doubleSpinBox_num_of_bins.setValue(data.get('dwell_mus', 0))
            self.doubleSpinBox_delay.setValue(data.get('delay_mus', 0))
            return data
        except Exception as e:
            logging.error('settings could not be loaded, because %s' % e)

    def save_settings(self):
        data = self.read_indicators()
        FileHandl.save_pickle_simple(self.settings_path, data)
        return self.settings_path

    def read_indicators(self):
        delay_mus = self.doubleSpinBox_delay.value()
        dwell_mus = self.doubleSpinBox_num_of_bins.value()
        return {'delay_mus': delay_mus, 'dwell_mus': dwell_mus}

    def start_scan(self):
        set_dict = self.read_indicators()
        num_of_bins = int(set_dict['dwell_mus'] * 100)
        delay_10ns = int(set_dict['delay_mus'] * 100)
        Cfg._main_instance.start_tilda_passive(num_of_bins, delay_10ns,
                                               self.tipa_raw_data_callback, self.tipa_status_callback,
                                               self.tipa_steps_scans_callback)

    def stop_scan(self):
        xml_path = Cfg._main_instance.scan_pars.get('Ni_tipa').get('pipeInternals').get('activeXmlFilePath')
        self.label_last_saved.setText(xml_path)
        self.label_last_saved.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        Cfg._main_instance.stop_tilda_passive()

    def closeEvent(self, *args, **kwargs):
        Cfg._main_instance.stop_tilda_passive(True)
        self.main_ui.close_tilda_passive()
