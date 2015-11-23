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
from copy import deepcopy, copy


class ScanControlUi(QtWidgets.QMainWindow, Ui_MainWindowScanControl):
    def __init__(self, main):
        """ Non-Modal Main window to control the Scan.
         All Isotope/track/sequencer settings are entered here. """
        super(ScanControlUi, self).__init__()
        self.setupUi(self)

        self.scanning = False
        self.buffer_scan_dict = {}
        self.track_wins_dict = {}
        self.win_title = None

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
        self.main.start_scan(self.buffer_scan_dict)
        # print(self.buffer_scan_dict)

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
        tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(self.buffer_scan_dict)
        self.buffer_scan_dict['isotopeData']['nOfTracks'] = tracks
        self.update_track_list()

    def remove_selected_track(self):
        """
        will remove the currently selected
        """
        try:
            self.buffer_scan_dict.pop(self.listWidget.currentItem().text())
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
        self.track_wins_dict[str(track_number)] = TrackUi(self, track_number, self.buffer_scan_dict[track_name])

    def track_win_closed(self, tracknum_int):
        self.track_wins_dict.pop(str(tracknum_int))

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
        self.win_title = win_title

    def setup_iso(self):
        logging.debug('setting up isotope')
        iso_win = SetupIsotopeUi(self.main, {})
        self.buffer_scan_dict = deepcopy(iso_win.new_scan_dict)
        print(self.buffer_scan_dict)
        self.update_track_list()
        self.update_win_title()

    def save_to_db(self):
        logging.debug('saving settings to database')
        trk_num, trk_lis = SdOp.get_number_of_tracks_in_scan_dict(self.buffer_scan_dict)
        for i in trk_lis:
            logging.debug('saving track ' + str(i) + ' dict is: ' +
                          str(self.buffer_scan_dict['track' + str(i)]))
            DbOp.add_scan_dict_to_db(self.main.database, self.buffer_scan_dict, i, track_key='track' + str(i))

    def close_track_wins(self):
        new_dict = copy(self.track_wins_dict)
        for key, val in new_dict.items():
            val.close()

    def closeEvent(self, event):
        if self.scanning:
            logging.info('will not exit, because a scan is ongoing.')
            event.ignore()
        else:
            logging.info('closing scan win' + str(self.win_title))
            event.accept()
            print(self.track_wins_dict)
            self.close_track_wins()
            self.main.scan_control_win_closed(self)
            print(self.track_wins_dict)


# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     blub = ScanControlUi(None)
#     app.exec_()