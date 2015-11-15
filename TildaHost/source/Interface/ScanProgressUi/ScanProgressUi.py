"""
Created on 

@author: simkaufm

Module Description:
"""
from PyQt5 import QtWidgets

from Interface.ScanProgressUi.Ui_ScanProgress import Ui_ScanProgress


class ScanProgressUi(QtWidgets.QMainWindow, Ui_ScanProgress):
    """
    non modal scan progress window, which will be showing the progress of the scan and
    give the user the ability to abort or to halt the scan.
    """
    def __init__(self, scan_dict):
        super(ScanProgressUi, self).__init__()
        self.setupUi(self)

        self.scan_dict = scan_dict
        self.n_o_total_tracks = 0
        self.n_o_compl_tracks = 0
        self.n_o_total_scans_in_track = 0
        self.n_o_compl_scans_in_track = 0
        self.n_o_total_steps_in_track = 0
        self.n_o_compl_steps_in_track = 0

        """ connect buttons etc. """
        self.pushButton_abort.clicked.connect(self.abort)
        self.pushButton_halt.clicked.connect(self.halt)

        self.show()

    def abort(self):
        pass

    def halt(self):
        pass

    def update_progressbar(self, bar, mini, maxi):
        if maxi != 0:
            percent = int(round(mini / maxi * 100))
            bar.setValue(percent)

    def update_track_progbar(self):
        mini = self.n_o_compl_steps_in_track * self.n_o_compl_scans_in_track
        maxi = self.n_o_total_steps_in_track * self.n_o_total_scans_in_track
        self.update_progressbar(self.progressBar_track, mini, maxi)

    def set_n_of_total_tracks(self, n_o_tracks):
        self.n_o_total_tracks = n_o_tracks
        self.label_total_track_num.setText(str(n_o_tracks))
        self.update_progressbar(self.progressBar_overall, self.n_o_compl_tracks, self.n_o_total_tracks)

    def set_n_of_compl_tracks(self, n_o_com_tracks):
        self.n_o_compl_tracks = n_o_com_tracks
        self.label_act_track_num.setText(str(n_o_com_tracks))
        self.update_progressbar(self.progressBar_overall, self.n_o_compl_tracks, self.n_o_total_tracks)

    def set_n_of_total_scans(self, n_o_total_scans):
        self.n_o_total_scans_in_track = n_o_total_scans
        self.label_max_scan_number.setText(str(n_o_total_scans))
        self.update_track_progbar()

    def set_n_of_compl_scans(self, n_o_compl_scans):
        self.n_o_compl_scans_in_track = n_o_compl_scans
        self.label_act_scan_number.setText(str(n_o_compl_scans))
        self.update_track_progbar()

    def set_n_of_total_steps_in_track(self, n_o_total_steps):
        self.n_o_total_steps_in_track = n_o_total_steps
        self.label_max_completed_steps.setText(str(n_o_total_steps))
        self.update_track_progbar()

    def set_n_of_compl_steps_in_track(self, n_o_compl_steps):
        self.n_o_compl_steps_in_track = n_o_compl_steps
        self.label_act_completed_steps.setText(str(n_o_compl_steps))
        self.update_track_progbar()

import sys

if __name__ == '__main__':
        app = QtWidgets.QApplication(sys.argv)
        ui = ScanProgressUi()
        ui.set_n_of_compl_tracks(1)
        ui.set_n_of_total_tracks(5)
        ui.set_n_of_compl_scans(1)
        ui.set_n_of_total_scans(10)
        ui.set_n_of_compl_steps_in_track(50)
        ui.set_n_of_total_steps_in_track(100)
        app.exec_()
