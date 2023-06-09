"""
Created on 

@author: simkaufm

Module Description:
"""
import os
import logging

from PyQt5 import QtWidgets, QtCore

import Tilda.Application.Config as Cfg
from Tilda.Interface.ScanProgressUi.Ui_ScanProgress import Ui_ScanProgress


class ScanProgressUi(QtWidgets.QMainWindow, Ui_ScanProgress):
    scan_prog_callback_sig = QtCore.pyqtSignal(dict)

    def __init__(self, parent):
        """
        non modal scan progress window, which will be showing the progress of the scan and
        give the user the ability to abort or to halt the scan.
        """
        super(ScanProgressUi, self).__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # necessary for not keeping it in memory

        self.setupUi(self)

        self.scan_prog_callback_sig = Cfg._main_instance.subscribe_to_scan_prog()
        self.scan_prog_callback_sig.connect(self.update_progress)

        self.progressBar_track.setMaximum(100)
        self.progressBar_overall.setMaximum(100)

        self.progressBar_overall.setValue(0)
        self.progressBar_track.setValue(0)
        self.progressbar_color()

        """ connect buttons etc. """
        self.pushButton_abort.clicked.connect(self.abort)
        self.checkBox.clicked.connect(self.halt)
        self.pushButton_pause.clicked.connect(self.pause_scan)

        self.show()

    def reset(self):
        self.checkBox.setChecked(False)
        self.progressBar_overall.setValue(0)
        self.label_timeleft_set.setText('-1')
        self.label_act_track_num.setText('-1')
        self.label_total_track_num.setText('-1')  #
        self.progressBar_track.setValue(0)
        self.label_act_scan_number.setText('-1')
        self.label_max_scan_number.setText('-1')  #
        self.label_act_completed_steps.setText('-1')
        self.label_max_completed_steps.setText('-1')  #
        self.groupBox.setTitle('track...')
        self.progressbar_color()

    def abort(self):
        Cfg._main_instance.abort_scan = True
        self.progressbar_color('red')

    def progressbar_color(self, color='#1ED760'):
        # default set to green
        try:
            logging.debug('setting background color')
            # style = """QProgressBar::chunk { background-color: %s; }""" % color
            st = """
                QProgressBar::chunk { background-color: %s; }

                QProgressBar {
                border: 1px solid grey;
                border-radius: 2px;
                text-align: right;
                background: #eeeeee;
                }
                """ % color

            self.progressBar_overall.setStyleSheet(st)
            self.progressBar_track.setStyleSheet(st)
        except Exception as e:
            logging.error('error while setting palette: %s' % e, exc_info=True)

    def halt(self):
        Cfg._main_instance.halt_scan_func(self.checkBox.isChecked())

    def pause_scan(self):
        """
        This will pause the scan with a loop in the handshake.
        Use this, if the laser jumped or so and you want to continue on the data.
        """
        paused_bool = Cfg._main_instance.pause_scan()
        if paused_bool:  # this means the scan is currently paused
            self.pushButton_pause.setText('continue')
        else:
            self.pushButton_pause.setText('pause')
        self.pushButton_abort.setDisabled(paused_bool)
        self.checkBox.setDisabled(paused_bool)

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

    # def closeEvent(self, *args, **kwargs):
    #     self.abort()
