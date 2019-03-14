"""

Created on '14.02.2019'

@author:'fsommer'

"""

import ast
import functools
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime
from time import sleep

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import Service.FileOperations.FolderAndFileHandling as FileHandler

import Application.Config as Cfg

from Interface.JobStackerUi.Ui_JobStacker import Ui_JobStacker
from Interface.JobStackerUi.SelectRepetitionsUi import SelectRepetitionsUi
from Interface.ScanControlUi.ScanControlUi import ScanControlUi


class JobStackerUi(QtWidgets.QMainWindow, Ui_JobStacker):
    main_ui_status_call_back_signal = QtCore.pyqtSignal(dict)

    def __init__(self, main_gui):
        super(JobStackerUi, self).__init__()

        self.setupUi(self)
        self.stored_window_title = 'JobStacker'
        self.setWindowTitle(self.stored_window_title)
        self.main_gui = main_gui

        self.job_list_before_execution = None  # copy of job_list that is made when run is started
        self.scan_ctrl_win = None
        self.repetition_ctrl_win = None
        self.item_passed_to_scan_ctrl = None

        self.running = False
        self.aborted = False
        # self.reps_on_file_to_go = 0  # not necessary any more
        self.wait_for_next_job = False
        self.main_status_is_idle = True

        ''' push button functionality '''
        self.pb_add.clicked.connect(self.add_job)
        self.pb_del.clicked.connect(self.del_selected_job)
        self.pb_repetitions.clicked.connect(self.change_reps_of_selected)
        self.pb_load.clicked.connect(self.load_from_txt)
        self.pb_save.clicked.connect(self.save_to_txt)
        self.pb_run.clicked.connect(self.run_next_job)

        ''' job list related '''
        self.list_joblist.itemDoubleClicked.connect(self.dbl_clicked_item)

        self.show()

    ''' cmd list related: '''

    def cmd_list_to_gui(self, cmd_list):
        """ write a list of str job to the gui """
        # remove all items tht were already in the list.
        self.list_joblist.clear()
        self.list_joblist.addItems(cmd_list)

    def cmd_list_from_gui(self):
        """ return a list of all jobs in the gui """
        items = []
        num_of_items = self.list_joblist.count()
        for i in range(num_of_items):
            job_str = self.list_joblist.item(i).text()
            items.append(job_str)
        return items

    def add_job(self):
        self.list_joblist.addItems(['newjob | 1'])  # 1 is the standard number for repetitions, all other text filled by later
        added_item_id = self.list_joblist.count()
        self.item_passed_to_scan_ctrl = self.list_joblist.item(added_item_id-1)
        # open ScanControlUi window
        logging.info('Opening scan control window for new item {}')
        self.open_scan_ctrl_win()

    def del_selected_job(self):
        """ if an item is selected, remove this from the list. """
        items = self.list_joblist.selectedItems()
        [self.list_joblist.takeItem(self.list_joblist.row(each)) for each in items]

    def change_reps_of_selected(self):
        """ for selected items, set number of repetitions to user defined value """
        items = self.list_joblist.selectedItems()
        if items:
            self.setEnabled(False)
            self.repetition_ctrl_win = SelectRepetitionsUi(self)
            old_reps = items[0].text().split(' | ')[-1]
            self.repetition_ctrl_win.spinBox_number_reps.setValue(int(old_reps))
        else:
            logging.warning('Jobstacker: No item selected. Cannot change repetitions.')

    ''' saving and loading '''

    def load_from_txt(self):
        """ load the list of jobs from an existing text file """
        parent = QtWidgets.QFileDialog(self)
        if Cfg._main_instance is None:
            start_path = os.path.basename(__file__)
        else:
            start_path = Cfg._main_instance.working_directory
        txt_path, ok = QtWidgets.QFileDialog.getOpenFileName(
            parent, "select JobStacker .txt file", start_path, '*.txt')
        if txt_path:
            list_of_cmds = FileHandler.load_from_text_file(txt_path)
            self.cmd_list_to_gui(list_of_cmds)
            logging.debug('Loaded jobs from path: {}'.format(txt_path))
            return txt_path

    def save_to_txt(self):
        """ save the current settings to a text file """
        parent = QtWidgets.QFileDialog(self)
        if Cfg._main_instance is None:
            start_path = os.path.basename(__file__)
        else:
            start_path = Cfg._main_instance.working_directory
        path, ok = QtWidgets.QFileDialog.getSaveFileName(
            parent, "select job stacker .txt file", start_path, '*.txt')
        if path:
            FileHandler.save_txt_file_line_by_line(path, self.cmd_list_from_gui())
            return path

    ''' necessary for running '''

    def subscribe_to_main(self):
        """
        pass the call back signal to the main and connect to self.update_status
        """
        self.main_gui.main_ui_status_call_back_signal.connect(self.update_status)
        Cfg._main_instance.send_state()
        Cfg._main_instance.info_warning_string_main_signal.connect(self.info_from_main)

    def unsubscribe_from_main(self):
        """
        unsubscribe from main and disconnect signals
        """
        Cfg._main_instance.gui_status_unsubscribe()
        self.main_gui.main_ui_status_call_back_signal.disconnect()
        Cfg._main_instance.info_warning_string_main_signal.disconnect()

    def info_from_main(self, info_str):
        """ listens to info from main and changes states """
        # print('----------info from main: %s ---------------' % info_str)
        if info_str == 'scan_complete':
            logging.debug('job stacker received scan complete info.')
        elif info_str == 'starting_scan':
            pass
        elif info_str == 'pre_scan_timeout':
            pass
        elif info_str == 'scan_aborted':
            logging.debug('job stacker received scan aborted info')
            self.wait_for_next_job = True  # abort shouldn't change next job. Maybe user still wants to run it
            logging.info('Last job aborted in scan control window. Waiting for next job now.')
        elif info_str == 'scan_halted':
            logging.debug('job stacker received scan halted info')
            self.wait_for_next_job = True  # halt shouldn't change next job. Maybe user still wants to run it
            logging.info('Last job halted in scan control window. Waiting for next job now.')
        elif info_str == 'kepco_scan_timedout':
            # kepco scans are not tested yet. Should probably do the same as when scan is aborted here?
            pass

    def update_status(self, status_dict):
        """
        will be called when the Main changes its status
        status_dict keys: ['workdir', 'status', 'database', 'laserfreq', 'accvolt',
         'sequencer_status', 'fpga_status', 'dmm_status']
        """
        self.main_status_is_idle = status_dict.get('status', '') == 'idle'
        if self.main_status_is_idle and self.wait_for_next_job and Cfg._main_instance.jobs_to_do_when_idle_queue == []:
            logging.debug('Main is idle, starting next job now.')
            self.wait_for_next_job = False
            self.run_next_job()



    ''' joblist related '''

    def run_next_job(self):
        if self.running is False:  # only do this on first call
            self.aborted = False  # maybe previous run was aborted
            self.subscribe_to_main()  # need updates from scan process

            # save current joblist to file before executing
            # self.save_to_txt()
            # store original items of joblist. Jobs will be removed after execution to show progress...
            self.job_list_before_execution = self.cmd_list_from_gui()

            # lock user-interface and change run button to abort
            self.pb_save.setEnabled(False)  # lock all control buttons except run/abort
            self.pb_load.setEnabled(False)
            self.pb_repetitions.setEnabled(False)
            self.pb_del.setEnabled(False)
            self.pb_add.setEnabled(False)
            self.list_joblist.setEnabled(False)  # locks the job-list
            self.pb_run.setText('Abort')
            self.pb_run.clicked.disconnect()
            self.pb_run.clicked.connect(self.abort_run)

        # process next items
        item = self.list_joblist.item(0)

        if self.scan_ctrl_win is not None:
            self.scan_ctrl_win.close()
            self.scan_ctrl_win = None
        if item is not None and self.aborted is False:
            # reduce number of repetitions on file by 1
            item_props = item.text().split(' | ')
            self.label_infostring.setText('Current job: ' + item.text())
            repetitions_before = item_props.pop()
            new_repetitions = int(repetitions_before) - 1
            # store number or repetitions_on_file
            if new_repetitions > 0:
                item_props += [str(new_repetitions)]
                item.setText(' | '.join(item_props))
            else:
                self.list_joblist.takeItem(0)

            self.running = True
            next_item_text = item.text()
            next_item_info = next_item_text.split(' | ')
            self.open_scan_ctrl_win(item_info=next_item_info)
            logging.info('job stacker is starting next job now.')
            self.scan_ctrl_win.go()
        else:
            # all jobs are done, revert to normal
            logging.info('job stacker done, reactivating Ui.')
            self.running = False
            self.wait_for_next_job = False
            self.setWindowTitle('Job Stacker')
            self.label_infostring.setText('Not working on jobs. Click run to start.')
            # restore original joblist
            self.cmd_list_to_gui(self.job_list_before_execution)
            # unlock user-interface change abort back to run
            self.pb_save.setEnabled(True)  # unlock all control buttons except run/abort
            self.pb_load.setEnabled(True)
            self.pb_repetitions.setEnabled(True)
            self.pb_del.setEnabled(True)
            self.pb_add.setEnabled(True)
            self.list_joblist.setEnabled(True)  # unlocks the job-list
            self.pb_run.setText('Run')
            self.pb_run.clicked.disconnect()
            self.pb_run.clicked.connect(self.run_next_job)
            self.pb_run.setEnabled(True)  # in case it was disabled by an abort...

    def abort_run(self):
        self.aborted = True
        self.pb_run.setText('Execution will abort after next job!')
        self.pb_run.setEnabled(False)
        logging.info('User aborted execution of job stacker!')
        self.list_joblist.clear()

    def dbl_clicked_item(self):
        '''
        On doubleclick to one item, open a ScanControl window and load the specific isotope.
        Deactivate the JobStackerUi window in the meantime
        Also close all other open ScanControls before?
        :return:
        '''
        # get isotope and sequencer type from double-clicked item text
        self.item_passed_to_scan_ctrl = self.list_joblist.currentItem()
        cur_item_text = self.item_passed_to_scan_ctrl.text()
        cur_item_info = cur_item_text.split(' | ')
        # open ScanControlUi window and pass item info
        logging.info('Opening scan control window for item {}.'.format(cur_item_text))
        self.open_scan_ctrl_win(item_info=cur_item_info)

    ''' related windows '''

    def open_scan_ctrl_win(self, item_info=None):
        # disable JobStackerUi window
        if self.running is False:  # only lock completely during setup. Need abort button when running.
            self.setEnabled(False)  # while setting up isotope, no interactions should be made in the JobStackerUi
            self.setWindowTitle('Currently unavailable. ScanControlUi open!')
        else:
            self.setWindowTitle('Currently unavailable. Processing jobs!')
        self.stored_window_title = self.windowTitle()  # store current window title

        # open scan control window
        if Cfg._main_instance.working_directory is None:
            if self.main_gui.choose_working_dir() is None:
                return None
        self.scan_ctrl_win = ScanControlUi(self.main_gui, job_stacker=self)
        self.main_gui.act_scan_wins.append(self.scan_ctrl_win)

        if item_info:
            # pass isotope and sequencer type to new scan control window
            iso_str, seq_str, reps_as_go, reps_new_file = item_info
            self.scan_ctrl_win.setup_iso(dont_open_setup_win=True, iso=iso_str, seq=seq_str)
            # change repetitions in scan control window to desired value and set as go
            self.scan_ctrl_win.spinBox_num_of_reps.setValue(int(reps_as_go))
        # set reps_as_go checkbox to True and disable
        self.scan_ctrl_win.checkBox_reps_as_go.setChecked(True)
        self.scan_ctrl_win.checkBox_reps_as_go.setEnabled(False)

    def scan_control_ui_closed(self, active_iso, num_of_reps):
        if self.running is False:
            # get isotope and sequencer type from active iso
            if active_iso:
                iso_seq_naming = active_iso.split('_')
                seq_str = iso_seq_naming.pop(-1)
                iso_str = '_'.join(iso_seq_naming)  # isotopes are actually often named with underscores...
                reps_as_go = num_of_reps
                print(self.item_passed_to_scan_ctrl.text())
                num_repeat_job = self.item_passed_to_scan_ctrl.text().split(' | ')[-1]
                new_item_def = ' | '.join([iso_str, seq_str, str(reps_as_go), num_repeat_job])
                self.item_passed_to_scan_ctrl.setText(new_item_def)
            else:
                logging.info('No isotope chosen, new entry will not be created.')
                self.list_joblist.setCurrentItem(self.item_passed_to_scan_ctrl)
                indx = self.list_joblist.currentRow()
                self.list_joblist.takeItem(indx)
            # change number of reps_as_go#
            self.setWindowTitle(self.stored_window_title)
            self.setEnabled(True)
            self.item_passed_to_scan_ctrl = None

    def repetition_ctrl_closed(self, reps):
        if reps is not None:
            items = self.list_joblist.selectedItems()
            for each in items:
                item_props = each.text().split(' | ')
                item_props.pop()
                item_props += [str(reps)]
                each.setText(' | '.join(item_props))
        self.setEnabled(True)


    ''' window related '''

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        if self.main_gui is not None:
            # tell main window that this window is closed.
            self.main_gui.close_job_stacker_win()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = JobStackerUi(None)

    app.exec_()

