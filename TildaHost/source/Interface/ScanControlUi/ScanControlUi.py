"""

Created on '29.10.2015'

@author:'simkaufm'

"""

import functools
import glob
import logging
import os
from copy import copy

from PyQt5 import QtWidgets, QtCore, QtGui

import Application.Config as Cfg
import TildaTools
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes
from Interface.ScanControlUi.Ui_ScanControl import Ui_MainWindowScanControl
from Interface.SetupIsotopeUi.SetupIsotopeUi import SetupIsotopeUi
from Interface.TrackParUi.TrackUi import TrackUi


class ScanControlUi(QtWidgets.QMainWindow, Ui_MainWindowScanControl):
    def __init__(self, main_gui, job_stacker=None):
        """ Non-Modal Main window to control the Scan.
         All Isotope/track/sequencer settings are entered here. """
        super(ScanControlUi, self).__init__()
        self.setupUi(self)

        self.active_iso = None  # str, key for the self.scan_pars dict in Main
        self.win_title = None
        self.track_wins_dict = {}  # dict containing all open track windows, key is track_num
        self.num_of_reps = 1  # how often this scan will be repeated. stored at begin of scan
        self.go_was_clicked_before = False  # variable to store if the user already clicked on 'Go'
        self.last_scan_was_aborted_or_halted = False

        self.main_gui = main_gui
        self.job_stacker_gui = job_stacker
        self.update_win_title()
        self.enable_go(True)
        self.enable_config_actions(False)
        self.plot_window_was_opened_here = False

        self.actionErgo.triggered.connect(functools.partial(self.go, True, True))
        self.actionGo_on_file.triggered.connect(self.go_on_file)
        self.actionSetup_Isotope.triggered.connect(self.setup_iso)
        self.actionAdd_Track.triggered.connect(self.add_track)
        self.actionSave_settings_to_database.triggered.connect(self.save_to_db)
        self.action_remove_track.triggered.connect(self.remove_selected_track)
        self.listWidget.doubleClicked.connect(self.work_on_existing_track)
        self.actionRe_open_plot_win.triggered.connect(self.reopen_live_pl_win)

        self.actionRe_open_plot_win.setEnabled(False)

        ''' adding additional Keyboard shortcuts, some are already defined from the QT Designer File '''
        QtWidgets.QShortcut(QtGui.QKeySequence("ESC"), self, self.close)

        self.show()

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
        # print('enabling Go? , ', enable, enable_bool, self.active_iso, bool)
        if not self.actionErgo.isEnabled() and enable:  # one scan is done or isotope was selected
            if self.go_was_clicked_before:
                self.spinBox_num_of_reps.stepDown()
                if self.spinBox_num_of_reps.value() > 0 and not self.last_scan_was_aborted_or_halted:
                    # keep scanning if reps > 0 and no scan was aborted
                    if self.checkBox_reps_as_go.isChecked():
                        direc = os.path.join(Cfg._main_instance.working_directory, 'sums', '*.xml')
                        latest_file = max(glob.iglob(direc), key=os.path.getctime)
                        self.go_on_file(latest_file, read_num_reps_from_spinbox=False)
                    else:
                        self.go(False)
                else:  # all scans are done here or one scan was aborted
                    if self.job_stacker_gui is not None:
                        self.job_stacker_gui.wait_for_next_job = True  # gets feedback to job stacker
                    self.spinBox_num_of_reps.setValue(self.num_of_reps)
                    self.go_was_clicked_before = False
                    self.last_scan_was_aborted_or_halted = False  # reset abort variable when last scan was done
        self.actionErgo.setEnabled(enable)
        # go on file can also be done without selecting an isotope before:
        self.actionGo_on_file.setEnabled(enable_bool)
        # setup Isotope, add Track and remove Track should only be blocked for the scanning isotope
        enable_bool_scanning = enable_bool or Cfg._main_instance.scan_progress.get('activeIso', None) != self.active_iso
        for tracknums, tracks in self.track_wins_dict.items():
            tracks.enable_confirm(enable_bool_scanning)
        self.actionAdd_Track.setEnabled(enable_bool_scanning)
        self.action_remove_track.setEnabled(enable_bool_scanning)
        self.actionSetup_Isotope.setEnabled(enable_bool_scanning)

    def enable_config_actions(self, enable_bool):
        """ this will enable/disable the config elements """
        self.actionAdd_Track.setEnabled(enable_bool)
        self.actionSave_settings_to_database.setEnabled(enable_bool)
        self.action_remove_track.setEnabled(enable_bool)

    def go(self, read_spin_box=True, ergo=True):
        """
        will set the state in the main to go
        :param read_spin_box: bool, True for first "call"
                ergo: bool, False for go on file
        """
        self.last_scan_was_aborted_or_halted = False  # no matter what when clicking go, this will be set False
        if read_spin_box:
            self.go_was_clicked_before = True
            self.num_of_reps = self.spinBox_num_of_reps.value()
        # if Cfg._main_instance.scan_pars[self.active_iso]['isotopeData']['type'] in ['trs', 'trsdummy']:
            #  for now only open the window when using a time resolved scan.
        acq_on_file_in_dict = isinstance(
            Cfg._main_instance.scan_pars[self.active_iso]['isotopeData'].get('continuedAcquisitonOnFile', False), str)
        if ergo and acq_on_file_in_dict:
            # if its an ergo and continuedAcquisitonOnFile is already written to scandict, this must be deleted:
            Cfg._main_instance.scan_pars[self.active_iso]['isotopeData'].pop('continuedAcquisitonOnFile')
        if ergo:
            # if its an ergo we do not want any old readings of dmms or triton devs in the scandict
            Cfg._main_instance.remove_old_dmm_triton_from_scan_pars(self.active_iso)
        self.wrap_open_live_plot_win()
        Cfg._main_instance.start_scan(self.active_iso)

    def go_on_file(self, filename=None, read_num_reps_from_spinbox=True):
        """
        starts a measurement with scan parameters from an existing file which is selected via a pop up file dialog,
        adding up on the already accumulated data in this file.
        """
        parent = QtWidgets.QFileDialog(self)
        direc = os.path.join(Cfg._main_instance.working_directory, 'sums', '*.xml')
        if os.path.isdir(os.path.split(direc)[0]):
            # pre select the latest file
            if filename is None or filename is False:
                latest_file = max(glob.iglob(direc), key=os.path.getctime)
                filename, ok = QtWidgets.QFileDialog.getOpenFileName(
                    parent, 'select an existing .xml file', latest_file, '*.xml')
            if filename:
                logging.info('continuing acquisition on: %s' % filename)
                scan_dict, e_tree_ele = TildaTools.scan_dict_from_xml_file(filename)
                scan_dict['isotopeData']['continuedAcquisitonOnFile'] = os.path.split(filename)[1]
                for key, val in scan_dict.items():
                    if 'track' in key:
                        if 'trigger' in val:
                            if val['trigger'].get('type') is not None:
                                # old version trigger dict, without steptrigger etc.
                                trig_type_str = val['trigger']['type']
                                if 'TriggerTypes.' in trig_type_str:  # needed for older versions
                                    trig_type_str = trig_type_str.split('.')[1]
                                try:
                                    val['trigger']['type'] = getattr(TriggerTypes, trig_type_str)
                                except Exception as e:
                                    logging.error(
                                        'error: %s, could not do: getattr(TriggerTypes, %s) ' % (
                                        e, val['trigger']['type']))
                            else:
                                # new version trigger dict -> {trigger: {stepTrigger: {'type: ...} ... }
                                for trig_types, trig_dicts in val['trigger'].items():
                                    trig_type_str = trig_dicts['type']
                                    if 'TriggerTypes.' in trig_type_str:  # needed for older versions
                                        trig_type_str = trig_type_str.split('.')[1]
                                    try:
                                        trig_dicts['type'] = getattr(TriggerTypes, trig_type_str)
                                    except Exception as e:
                                        logging.error(
                                            'error: %s, could not do: getattr(TriggerTypes, %s) ' % (
                                                e, trig_dicts['type']))
                self.active_iso = Cfg._main_instance.add_iso_to_scan_pars_no_database(scan_dict)
                self.update_track_list()
                self.update_win_title()
                self.go(read_spin_box=read_num_reps_from_spinbox, ergo=False)

    def add_track(self):
        """
        add a track either from loading of the database or by copying the last one in the Gui,
        depending on if there is still a track available in the db.
        Will not write to the database!
        """
        if self.active_iso is not None:
            Cfg._main_instance.add_next_track_to_iso_in_scan_pars(self.active_iso)
            self.update_track_list()

    def remove_selected_track(self):
        """
        will remove the currently selected
        """
        if self.active_iso is not None and self.listWidget.currentItem() is not None:
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
            if self.track_wins_dict.get(str(track_number), None) is not None:
                self.raise_win_to_front(self.track_wins_dict[str(track_number)])
            else:
                self.track_wins_dict[str(track_number)] = TrackUi(self, track_number, self.active_iso, self.main_gui)
        except Exception as e:
            logging.error('error while opening track window, error is: %s' % e, exc_info=True)
        Cfg._main_instance.send_state()  # to check whether the opened track is scanning now.

    def track_win_closed(self, tracknum_int):
        self.track_wins_dict.pop(str(tracknum_int))

    def update_track_list(self):
        """
        updates the track list in the gui
        """
        self.listWidget.clear()
        if self.active_iso:
            scan_d = Cfg._main_instance.scan_pars.get(self.active_iso)
            t, track_num_lis = TildaTools.get_number_of_tracks_in_scan_dict(scan_d)
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

    def setup_iso(self, dont_open_setup_win=False, iso=None, seq=None):
        """
        opens a dialog for choosing the isotope. This Dialog is non Modal.
        -> Blocks other executions
        :param: open_setup_win bool: if true, the new isotope is setup using the SetupIsotopeUi.
                Otherwise the isotope and sequencer_type must be given
                iso str: isotope name. Must be given if open_setup_win is False
                seq str: sequencer type. Must be given if open_setup_win is False
        """
        if self.active_iso:  # first remove the before selected isotope from the scan pars
            self.enable_config_actions(False)
            Cfg._main_instance.remove_iso_from_scan_pars(self.active_iso)
            self.active_iso = None
        logging.debug('setting up isotope')
        if not dont_open_setup_win:
            SetupIsotopeUi(self)
        else:
            try:  # maybe iso or seq are incorrect, then this will fail.
                self.active_iso = Cfg._main_instance.add_iso_to_scan_pars(iso, seq)
            except:
                logging.error('Could not setup isotope. Maybe isotope name or sequencer type are invalid!')
        if self.active_iso:
            self.enable_config_actions(True)
        self.update_track_list()
        self.update_win_title()
        Cfg._main_instance.send_state()  # request state from main for enabling go

    def save_to_db(self):
        """
        save all settings of the given isotope to the database.
        """
        if self.active_iso is not None:
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

    def wrap_open_live_plot_win(self):
        self.plot_window_was_opened_here = True
        self.main_gui.open_live_plot_win()

    def enable_reopen_plot_win(self):
        """ if a live plot window was opened from within this scan control window, allow to reopen it. """
        self.actionRe_open_plot_win.setEnabled(self.plot_window_was_opened_here)

    def reopen_live_pl_win(self):
        self.main_gui.open_live_plot_win()
        self.actionRe_open_plot_win.setEnabled(False)

    def scan_was_aborted(self):
        """
        this can be called from outside, if the scan was aborted and
        corresponding reactions can be perforemd,
        e.g. do not proceed with the next repetition...
        :return:
        """
        logging.info('scan control window %s received: scan_aborted command' % self.win_title)
        self.last_scan_was_aborted_or_halted = True

    def scan_was_halted(self):
        """
        this can be called from outside, if the scan was halted and
        corresponding reactions can be perforemd,
        e.g. do not proceed with the next repetition...
        :return:
        """
        logging.info('scan control window %s received: scan_halted command' % self.win_title)
        self.last_scan_was_aborted_or_halted = True

    def closeEvent(self, event):
        """
        unsubscribe from parent gui when closed
        """
        if self.job_stacker_gui is not None:
            self.job_stacker_gui.scan_control_ui_closed(self.active_iso, self.spinBox_num_of_reps.value())
        if self.active_iso:
            Cfg._main_instance.abort_scan = True  # TODO: Is this really wanted???
            Cfg._main_instance.remove_iso_from_scan_pars(self.active_iso)
        logging.info('closing scan win ' + str(self.win_title))
        self.close_track_wins()
        self.main_gui.scan_control_win_closed(self)
