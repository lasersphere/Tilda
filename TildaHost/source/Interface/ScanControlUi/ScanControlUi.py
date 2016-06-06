"""

Created on '29.10.2015'

@author:'simkaufm'

"""

from Interface.ScanControlUi.Ui_ScanControl import Ui_MainWindowScanControl
from Interface.TrackParUi.TrackUi import TrackUi
from Interface.SetupIsotopeUi.SetupIsotopeUi import SetupIsotopeUi
import Service.DatabaseOperations.DatabaseOperations as DbOp
import Service.Scan.ScanDictionaryOperations as SdOp
import Application.Config as Cfg

from PyQt5 import QtWidgets
import logging
from copy import copy


class ScanControlUi(QtWidgets.QMainWindow, Ui_MainWindowScanControl):
    def __init__(self, main_gui):
        """ Non-Modal Main window to control the Scan.
         All Isotope/track/sequencer settings are entered here. """
        super(ScanControlUi, self).__init__()
        self.setupUi(self)

        self.active_iso = None  # str, key for the self.scan_pars dict in Main
        self.win_title = None
        self.track_wins_dict = {}  # dict containing all open track windows, key is track_num

        self.actionGo.triggered.connect(self.go)
        self.actionSetup_Isotope.triggered.connect(self.setup_iso)
        self.actionAdd_Track.triggered.connect(self.add_track)
        self.actionSave_settings_to_database.triggered.connect(self.save_to_db)
        self.action_remove_track.triggered.connect(self.remove_selected_track)
        self.listWidget.doubleClicked.connect(self.work_on_existing_track)

        self.main_gui = main_gui
        self.update_win_title()
        self.enable_go(False)

        self.show()

    def enable_go(self, bool):
        """
        wrapper for enabling the Go button, True-> enabled
        will be disabled via callback signal in MainUi when status in Main is not idle
        """
        enable = bool and self.active_iso is not None
        # print('enabling Go? , ', enable, self.active_iso, bool)
        self.actionGo.setEnabled(enable)

    def go(self):
        """
        will set the state in the main to go
        """
        if Cfg._main_instance.start_scan(self.active_iso):
            self.main_gui.open_scan_progress_win()

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
            self.track_wins_dict[str(track_number)] = TrackUi(self, track_number, self.active_iso)
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
        for key, val in self.track_wins_dict.items():
            val.close()

    def closeEvent(self, event):
        """
        unsubscribe from parent gui when closed
        """
        if self.active_iso:
            Cfg._main_instance.remove_iso_from_scan_pars(self.active_iso)
        logging.info('closing scan win ' + str(self.win_title))
        self.close_track_wins()
        self.main_gui.scan_control_win_closed(self)
