"""

Created on '29.10.2015'

@author:'simkaufm'

"""

import functools
import logging
from copy import copy

from PyQt5 import QtWidgets, QtCore

import Application.Config as Cfg
import Service.Scan.ScanDictionaryOperations as SdOp
from Interface.DmmUi.DmmUi import DmmLiveViewUi
from Interface.ScanControlUi.Ui_ScanControl import Ui_MainWindowScanControl
from Interface.SetupIsotopeUi.SetupIsotopeUi import SetupIsotopeUi
from Interface.TrackParUi.TrackUi import TrackUi


class ScanControlUi(QtWidgets.QMainWindow, Ui_MainWindowScanControl):
    def __init__(self, main_gui):
        """ Non-Modal Main window to control the Scan.
         All Isotope/track/sequencer settings are entered here. """
        super(ScanControlUi, self).__init__()
        self.setupUi(self)

        self.active_iso = None  # str, key for the self.scan_pars dict in Main
        self.win_title = None
        self.track_wins_dict = {}  # dict containing all open track windows, key is track_num
        self.dmm_win = None  # here the open dmm_win is stored.
        self.num_of_reps = 1  # how often this scan will be repeated. stored at begin of scan
        self.go_was_clicked_before = False  # variable to stroe if the user already clicked on 'Go'

        self.actionGo.triggered.connect(functools.partial(self.go, True))
        self.actionSetup_Isotope.triggered.connect(self.setup_iso)
        self.actionAdd_Track.triggered.connect(self.add_track)
        self.actionSave_settings_to_database.triggered.connect(self.save_to_db)
        self.action_remove_track.triggered.connect(self.remove_selected_track)
        self.actionConfigure_voltage_measurement.triggered.connect(self.open_dmm_win)
        self.listWidget.doubleClicked.connect(self.work_on_existing_track)

        self.main_gui = main_gui
        self.update_win_title()
        self.enable_go(False)

        self.show()

    def open_dmm_win(self):
        if self.dmm_win is None:
            new_name = 'DMM Settings for %s' % self.active_iso
            self.dmm_win = DmmLiveViewUi(self, new_name, enable_com=False, active_iso=self.active_iso)
        else:
            self.raise_win_to_front(self.dmm_win)

    def close_dmm_live_view_win(self):
        self.dmm_win = None

    def raise_win_to_front(self, window):
        # this will remove minimized status
        # and restore window with keeping maximized/normal state
        window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)

        # this will activate the window
        window.activateWindow()

    def enable_go(self, enable_bool):
        """
        wrapper for enabling the Go button, True-> enabled
        will be disabled via callback signal in MainUi when status in Main is not idle
        """
        enable = enable_bool and self.active_iso is not None
        # print('enabling Go? , ', enable, self.active_iso, bool)
        if not self.actionGo.isEnabled() and enable:  # one scan is done or isotope was selected
            if self.go_was_clicked_before:
                self.spinBox_num_of_reps.stepDown()
                if self.spinBox_num_of_reps.value() > 0:  # keep scanning if reps > 0
                    self.go(False)
                else:  # all scans are done here
                    self.spinBox_num_of_reps.setValue(self.num_of_reps)
                    self.go_was_clicked_before = False
        self.actionGo.setEnabled(enable)

    def go(self, read_spin_box=True):
        """
        will set the state in the main to go
        """
        if read_spin_box:
            self.go_was_clicked_before = True
            self.num_of_reps = self.spinBox_num_of_reps.value()
        self.main_gui.open_scan_progress_win()
        if Cfg._main_instance.scan_pars[self.active_iso]['isotopeData']['type'] in ['trs', 'trsdummy']:
            #  for now only open the window when using a time resolved scan.
            self.main_gui.open_live_plot_win()
        Cfg._main_instance.start_scan(self.active_iso)

    def add_track(self):
        """
        add a track either from loading of the database or by copying the last one in the Gui,
        depending on if there is still a track available in the db.
        Will not write to the database!
        """
        Cfg._main_instance.add_next_track_to_iso_in_scan_pars(self.active_iso)
        self.update_track_list()

    def remove_selected_track(self):
        """
        will remove the currently selected
        """
        try:
            Cfg._main_instance.remove_track_from_scan_pars(self.active_iso,
                                                           self.listWidget.currentItem().text())
            self.update_track_list()
        except Exception as e:
            logging.error('Error occurred while removing track from list: ' + str(e))

    def work_on_existing_track(self):
        """
        will open TrackUi which will actively write
        into self.buffer_scan_dict of ScanControlUi if ok is pressed.
        """
        track_name = self.listWidget.currentItem().text()
        track_number = int(track_name[5:])
        logging.debug('working on track' + str(track_number))
        try:
            self.track_wins_dict[str(track_number)] = TrackUi(self, track_number, self.active_iso, self.main_gui)
        except Exception as e:
            print(e)

    def track_win_closed(self, tracknum_int):
        self.track_wins_dict.pop(str(tracknum_int))

    def update_track_list(self):
        """
        updates the track list in the gui
        """
        self.listWidget.clear()
        if self.active_iso:
            scan_d = Cfg._main_instance.scan_pars.get(self.active_iso)
            t, track_num_lis = SdOp.get_number_of_tracks_in_scan_dict(scan_d)
            newitems = ['track' + str(tr) for tr in track_num_lis]
            self.listWidget.addItems(newitems)

    def update_win_title(self):
        """
        updates the window title with the active isotope name as in self.active_iso
        """
        win_title = 'please setup Isotope'
        if self.active_iso is not None:
            win_title = self.active_iso
        self.setWindowTitle(win_title)
        self.win_title = win_title

    def setup_iso(self):
        """
        opens a dialog for choosing the isotope. This Dialog is non Modal.
        -> Blocks other executions
        """
        if self.active_iso:
            Cfg._main_instance.remove_iso_from_scan_pars(self.active_iso)
            self.active_iso = None
        logging.debug('setting up isotope')
        SetupIsotopeUi(self)
        self.update_track_list()
        self.update_win_title()
        Cfg._main_instance.send_state()  # request state from main for enabling go

    def save_to_db(self):
        """
        save all settings of the given isotope to the database.
        """
        logging.debug('saving settings to database')
        Cfg._main_instance.save_scan_par_to_db(self.active_iso)

    def close_track_wins(self):
        """
        when closing, close all track windows
        """
        track_win_copy = copy(self.track_wins_dict)
        for key, val in track_win_copy.items():
            val.close()
        track_win_copy = None

    def closeEvent(self, event):
        """
        unsubscribe from parent gui when closed
        """
        if self.active_iso:
            Cfg._main_instance.remove_iso_from_scan_pars(self.active_iso)
        logging.info('closing scan win ' + str(self.win_title))
        self.close_track_wins()
        self.main_gui.scan_control_win_closed(self)
