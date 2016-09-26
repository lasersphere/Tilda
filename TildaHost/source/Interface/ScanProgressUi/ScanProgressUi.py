"""
Created on 

@author: simkaufm

Module Description:
"""
import os

from PyQt5 import QtWidgets, QtCore

import Application.Config as Cfg
from Interface.ScanProgressUi.Ui_ScanProgress import Ui_ScanProgress


class ScanProgressUi(QtWidgets.QMainWindow, Ui_ScanProgress):
    scan_prog_callback_sig = QtCore.pyqtSignal(dict)

    def __init__(self, main_gui):
        """
        non modal scan progress window, which will be showing the progress of the scan and
        give the user the ability to abort or to halt the scan.
        """
        super(ScanProgressUi, self).__init__()
        self.setupUi(self)
        self.main_gui = main_gui

        Cfg._main_instance.subscribe_to_scan_prog(self.scan_prog_callback_sig)
        self.scan_prog_callback_sig.connect(self.update_progress)

        self.progressBar_track.setMaximum(100)
        self.progressBar_overall.setMaximum(100)

        self.progressBar_overall.setValue(0)
        self.progressBar_track.setValue(0)

        """ connect buttons etc. """
        self.pushButton_abort.clicked.connect(self.abort)
        self.checkBox.clicked.connect(self.halt)

        self.show()

    def abort(self):
        Cfg._main_instance.abort_scan = True

    def halt(self):
        Cfg._main_instance.halt_scan_func(self.checkBox.isChecked())

    def update_progress(self, progress_dict):
        """
        the dict contains the following keys:
        {'activeIso': str, 'overallProgr': float, 'timeleft': str, 'activeTrack': int, 'totalTracks': int,
        'trackProgr': float, 'activeScan': int, 'totalScans': int, 'activeStep': int, 'totalSteps': int,
        'trackName': str]
        """
        self.setWindowTitle('progress ' + os.path.split(progress_dict['activeFile'])[1])  #
        self.progressBar_overall.setValue(int(progress_dict['overallProgr']))
        self.label_timeleft_set.setText(progress_dict['timeleft'])
        self.label_act_track_num.setText(str(progress_dict['activeTrack']))
        self.label_total_track_num.setText(str(progress_dict['totalTracks']))  #
        self.progressBar_track.setValue(int(progress_dict['trackProgr']))
        self.label_act_scan_number.setText(str(progress_dict['activeScan']))
        self.label_max_scan_number.setText(str(progress_dict['totalScans']))  #
        self.label_act_completed_steps.setText(str(progress_dict['activeStep']))
        self.label_max_completed_steps.setText(str(progress_dict['totalSteps']))  #
        self.groupBox.setTitle(str(progress_dict['trackName']))

    def closeEvent(self, *args, **kwargs):
        self.abort()
        Cfg._main_instance.unsubscribe_from_scan_prog()
