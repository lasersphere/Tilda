"""

Created on '29.10.2015'

@author:'simkaufm'

"""

from Interface.ScanControlUi.Ui_ScanControl import Ui_MainWindowScanControl
from Interface.TrackParUi.TrackUi import TrackUi
from Interface.SetupIsotopeUi.SetupIsotopeUi import SetupIsotopeUi
import Service.DatabaseOperations.DatabaseOperations as DbOp
import Service.Scan.ScanDictionaryOperations as SdOp

from PyQt5 import QtWidgets

import logging
import sys
from copy import deepcopy


class ScanControlUi(QtWidgets.QMainWindow, Ui_MainWindowScanControl):
    def __init__(self, main):
        """ Non-Modal Main window to control the Scan.
         All Isotope/track/sequencer settings are entered here. """
        super(ScanControlUi, self).__init__()
        self.setupUi(self)

        self.buffer_scan_dict = {}
        self.track_win = None

        self.actionGo.triggered.connect(self.go)
        self.actionSetup_Isotope.triggered.connect(self.setup_iso)
        self.actionAdd_Track.triggered.connect(self.add_track)
        self.actionSave_settings_to_database.triggered.connect(self.save_to_db)
        self.action_remove_track.triggered.connect(self.remove_selected_track)
        self.listWidget.doubleClicked.connect(self.work_on_existing_track)

        self.show()

        self.main = main

    def go(self):
        # pss on the buffered scandict and let it run.
        logging.debug('starting measurement')

    def add_track(self):
        """
        add a track either from loading of the database or by copying the last one in the Gui,
        depending on if there is still a track available in the db.
        Will not write to the database!
        """
        logging.debug('adding track')
        sctype = self.buffer_scan_dict['isotopeData']['type']
        iso = self.buffer_scan_dict['isotopeData']['isotope']
        next_track_num, track_num_list = SdOp.get_available_tracknum(self.buffer_scan_dict)
        scand_from_db = DbOp.extract_track_dict_from_db(self.main.database, iso, sctype, next_track_num)
        if scand_from_db is not None:
            logging.debug('adding track' + str(next_track_num) + ' from database')
            self.buffer_scan_dict['track' + str(next_track_num)] = scand_from_db
        else:
            track_to_copy_from = 'track' + str(max(track_num_list))
            logging.debug('adding track' + str(next_track_num) + ' copying values from: ' + track_to_copy_from)
            self.buffer_scan_dict['track' + str(next_track_num)] = deepcopy(self.buffer_scan_dict[track_to_copy_from])
        self.update_track_list()

    def remove_selected_track(self):
        """
        will remove the currently sleectes
        """
        self.buffer_scan_dict.pop(self.listWidget.currentItem().text())
        self.update_track_list()

    def work_on_existing_track(self):
        """
        will open TrackUi which will actively write
        into self.buffer_scan_dict of ScanControlUi if ok is pressed.
        """
        track_name = self.listWidget.currentItem().text()
        track_number = int(track_name[-1])
        logging.debug('working on track' + str(track_number))
        self.track_win = TrackUi(self, track_number, self.buffer_scan_dict[track_name])

    def update_track_list(self):
        if self.buffer_scan_dict:
            self.listWidget.clear()
            newitems = sorted([key for key, val in self.buffer_scan_dict.items() if key[:-1] == 'track'])
            print(newitems)
            self.listWidget.addItems(newitems)

    def update_win_title(self):
        try:
            iso = str(self.buffer_scan_dict['isotopeData']['isotope'])
            sctype = str(self.buffer_scan_dict['isotopeData']['type'])
            win_title = iso + ' - ' + sctype
        except KeyError:
            win_title = 'please setup Isotope'
        self.setWindowTitle(win_title)

    def setup_iso(self):
        logging.debug('setting up isotope')
        iso_win = SetupIsotopeUi(self.main, {})
        self.buffer_scan_dict = deepcopy(iso_win.new_scan_dict)
        print(self.buffer_scan_dict)
        self.update_track_list()
        self.update_win_title()

    def save_to_db(self):
        logging.debug('saving settings to database')
        for i in range(SdOp.get_number_of_tracks_in_scan_dict(self.buffer_scan_dict)):
            DbOp.add_scan_dict_to_db(self.main.database, self.buffer_scan_dict, i, track_key='track' + str(i))



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    blub = ScanControlUi(None)
    app.exec_()