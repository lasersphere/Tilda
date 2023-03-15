"""

Created on '07.05.2015'

@author:'simkaufm'

"""

import logging
import os
import platform
import subprocess
from copy import deepcopy
import functools

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

import Tilda.Application.Config as Cfg
import Tilda.PolliFit.MPLPlotter as MPLPlotter
from Tilda.PolliFit.Gui.MainUi import MainUi as PolliMainUi
from Tilda.Interface.DialogsUi.ScanCompleteDialUi import ScanCompleteDialUi
from Tilda.Interface.DmmUi.DmmUi import DmmLiveViewUi
from Tilda.Interface.FrequencyUi.FreqUi import FreqUi
from Tilda.Interface.LiveDataPlottingUi.LiveDataPlottingUi import TRSLivePlotWindowUi
from Tilda.Interface.MainUi.Ui_Main import Ui_TildaMainWindow
from Tilda.Interface.OptionsUi.OptionsUi import OptionsUi
from Tilda.Interface.PostAccControlUi.PostAccControlUi import PostAccControlUi
from Tilda.Interface.PulsePatternUi.PulsePatternUi import PulsePatternUi
from Tilda.Interface.ScanControlUi.ScanControlUi import ScanControlUi
from Tilda.Interface.JobStackerUi.JobStackerUi import JobStackerUi
from Tilda.Interface.SimpleCounter.SimpleCounterDialogUi import SimpleCounterDialogUi
from Tilda.Interface.SimpleCounter.SimpleCounterRunningUi import SimpleCounterRunningUi
from Tilda.Interface.VersionUi.VersionUi import VersionUi
from Tilda.Scratch.Tetris import Tetris
from Tilda.Scratch.Snake import MyApp as Snake


class MainUi(QtWidgets.QMainWindow, Ui_TildaMainWindow):
    main_ui_status_call_back_signal = QtCore.pyqtSignal(dict)

    def __init__(self, application=None):
        QtCore.QLocale().setDefault(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        super(MainUi, self).__init__()
        work_dir_before_setup_ui = os.getcwd()
        os.chdir(os.path.dirname(__file__))  # necessary for the icons to appear
        self.setupUi(self)
        os.chdir(work_dir_before_setup_ui)  # change back
        # print('current working dir: %s' % os.getcwd())

        self.act_scan_wins = []  # list of active scan windows
        self.post_acc_win = None  # only one active post acceleration window
        self.scan_progress_win = None
        self.job_stacker_win = None
        self.freq_win = None
        self.simple_counter_gui = None
        self.dmm_live_view_win = None
        self.live_plot_win = None  # one active live plot window for displaying results from pipeline
        self.file_plot_wins = {}  # dict of active plot windows only for displaying from file.
        self.pollifit_win = None
        self.options_win = None
        self.tetris = None  # pssst dont tell
        self.snake = None
        self.pulse_pattern_win = None
        self.scan_complete_win = None
        # self.show_scan_compl_win = True  # TODO: Could use this for a session-long disable. Else just delete
        self.triton_listener_timedout_win = None

        self.application = application

        self.actionWorking_directory.triggered.connect(self.choose_working_dir)
        self.actionVersion.triggered.connect(self.open_version_win)
        self.actionScan_Control.triggered.connect(self.open_scan_ctrl_win)
        self.actionJob_Stacker.triggered.connect(self.open_job_stacker_win)  # TODO: define command and add actionItem
        self.actionPost_acceleration_power_supply_control.triggered.connect(self.open_post_acc_win)
        self.actionSimple_Counter.triggered.connect(self.open_simple_counter_win)
        self.actionoptions.triggered.connect(self.open_options_win)
        #self.actionSet_Laser_Frequency.triggered.connect(self.set_laser_freq)   # old version of frequency settings
        self.actionSet_Laser_Frequency.triggered.connect(self.open_freq_win)
        self.actionSet_acceleration_voltage.triggered.connect(self.set_acc_volt)
        self.actionLoad_spectra.triggered.connect(self.load_spectra)
        self.actionDigital_Multimeters.triggered.connect(self.open_dmm_live_view_win)
        self.actionPolliFit.triggered.connect(self.open_pollifit_win)
        self.actionPulse_pattern_generator.triggered.connect(self.open_pulse_pattern_win)


        """ connect double clicks on labels:"""
        self.label_workdir_set.mouseDoubleClickEvent = self.workdir_dbl_click
        self.label_workdir_set.setToolTip(self.workdir_dbl_click.__doc__)
        self.label_laser_freq_set.mouseDoubleClickEvent = self.laser_freq_dbl_click
        self.label_acc_volt_set.mouseDoubleClickEvent = self.acc_volt_dbl_click
        self.label_8.mouseDoubleClickEvent = self.dmm_setup_dbl_click
        self.label_triton_subscription.mouseDoubleClickEvent = self.triton_dbl_click

        """ connect buttons """
        self.pushButton_open_dir.clicked.connect(self.open_dir)
        self.pushButton_open_dir.setToolTip(self.open_dir.__doc__)

        """ add shortcuts """
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+T"), self, self.start_tetris)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+S"), self, self.start_snake)
        # QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+A"), self, self.open_pollifit_win)

        self.subscribe_to_main()
        self.show()

    ''' connected actions '''
    def workdir_dbl_click(self, event):
        """
        Doubleclick to open a file browser to select/change the TILDA working directory.
        """
        self.choose_working_dir()

    def laser_freq_dbl_click(self, event):
        #self.set_laser_freq()   # old version
        self.open_freq_win()

    def acc_volt_dbl_click(self, event):
        self.set_acc_volt()

    def dmm_setup_dbl_click(self, event):
        self.open_dmm_live_view_win()

    def triton_dbl_click(self, event):
        dial = QtWidgets.QMessageBox(self)
        ret = QtWidgets.QMessageBox.question(dial,
                                             'Triton Unsubscribe','Do you want to unsubscribe from all triton devs?'
                                             )
        if ret==QtWidgets.QMessageBox.Yes:
            Cfg._main_instance.triton_unsubscribe_all()

    def subscribe_to_main(self):
        """
        pass the call back signal to the main and connect to self.update_status
        """
        Cfg._main_instance.gui_status_subscribe(self.main_ui_status_call_back_signal)
        self.main_ui_status_call_back_signal.connect(self.update_status)
        Cfg._main_instance.send_state()
        Cfg._main_instance.info_warning_string_main_signal.connect(self.info_from_main)

    def info_from_main(self, info_str):
        """ handle info strings which are emitted from the main """
        # print('----------info from main: %s ---------------' % info_str)
        if info_str == 'scan_complete':
            self.open_scan_complete_win()
        elif info_str == 'starting_scan':
            if self.scan_complete_win is not None:
                self.scan_complete_win.close()
        elif info_str == 'pre_scan_timeout':
            # Realized this message should not interrupt the scan from continuing
            # info = QtWidgets.QMessageBox.information(
            #     self, 'Warning!', '-------- Warning -------\n '
            #                    'the pre scan measurment did not finish within the given time!\n'
            #                    'Press ok, to proceed with scan.')
            self.d = QtWidgets.QDialog()
            self.d.setWindowFlags(Qt.WindowStaysOnTopHint)
            layout = QtWidgets.QHBoxLayout(self.d)
            warning_text = QtWidgets.QLabel()
            warning_text.setAlignment(Qt.AlignCenter)
            warning_text.setText('-------- Warning -------\n '
                                 'The pre scan measurment did not finish within the given time!\n'
                                 'Will continue now')
            layout.addWidget(warning_text)
            self.d.setWindowTitle("Pre/Post Scan Timeout")
            self.d.show()
        elif info_str == 'scan_aborted':
            # tell all scan control windows that the scan was aborted
            for each in self.act_scan_wins:
                each.scan_was_aborted()
        elif info_str == 'scan_halted':
            # tell all scan control windows that the scan was halted
            for each in self.act_scan_wins:
                each.scan_was_halted()
        elif info_str == 'kepco_scan_timedout':
            info = QtWidgets.QMessageBox.information(
                self, 'Warning!',
                '-------- Warning -------\n '
                'the kepco scan finished ramping the voltage,\n'
                'but not all digital multimeters delivered a reading!\n'
                'Check your cabling and the timing of the trigger send to the digital multimeters!')
        elif info_str == 'triton_listener_timedout':
            if self.triton_listener_timedout_win is None and not Cfg._main_instance.scan_main.sequencer.pause_bool:
                self.triton_listener_timedout_win = True
                info = QtWidgets.QMessageBox.information(
                    self, 'Warning!',
                    '-------- Warning -------\n '
                    'Triton Listener did not receive values for some time!\n'
                    'Check subscriptions and if devices are still running! \n'
                    '-------- Warning -------\n ')
                self.triton_listener_timedout_win = None

    def unsubscribe_from_main(self):
        """
        unsubscribe from main and disconnect signals
        """
        Cfg._main_instance.gui_status_unsubscribe()
        self.main_ui_status_call_back_signal.disconnect()

    def update_status(self, status_dict):
        """
        will be called when the Main changes its status
        status_dict keys: ['workdir', 'status', 'database', 'laserfreq', 'accvolt',
         'sequencer_status', 'fpga_status', 'dmm_status']
        """
        self.label_workdir_set.setText(str(status_dict.get('workdir', '')))
        self.label_main_status.setText(str(status_dict.get('status', '')))
        self.label_database.setText(str(status_dict.get('database', '')))
        self.label_laser_freq_set.setText("{:.5f}".format(status_dict.get('laserfreq', '')))
        self.label_acc_volt_set.setText(str(status_dict.get('accvolt', '')))
        self.label_fpga_state_set.setText(str(status_dict.get('fpga_status', '')))
        self.label_sequencer_status_set.setText(str(status_dict.get('sequencer_status', '')))
        self.label_dmm_status.setText(self.make_dmm_status_nice(status_dict))
        self.label_triton_subscription.setText(status_dict.get('triton_status',''))
        stat_is_idle = status_dict.get('status', '') == 'idle'
        for w in self.act_scan_wins:
            w.enable_go(stat_is_idle)
        if self.dmm_live_view_win is not None:
            self.dmm_live_view_win.enable_communication(stat_is_idle)

    def choose_working_dir(self):
        """ will open a modal file dialog and set all workingdirectories of the pipeline to the chosen folder """
        start_path = os.path.expanduser('~')
        if Cfg._main_instance.working_directory:
            start_path = os.path.split(Cfg._main_instance.working_directory)[0]
        workdir = QtWidgets.QFileDialog.getExistingDirectory(QtWidgets.QFileDialog(),
            'choose working directory', start_path)
        return Cfg._main_instance.work_dir_changed(workdir)


    # def set_laser_freq(self):
    #     """
    #     not needed anymore, replaced by open freq win
    #     :return:
    #     """
    #     laser_freq, ok = QtWidgets.QInputDialog.getDouble(self, 'Laser', 'laser wavenumber [cm-1]',
    #                                                       0, 0, 9999999,
    #                                                       5)
    #     if ok:
    #         Cfg._main_instance.laser_freq_changed(laser_freq)

    def set_acc_volt(self):
        acc_volt, ok = QtWidgets.QInputDialog.getDouble(self, 'Acceleration Voltage', 'acceleration voltage [V]',
                                                          0, 0, 9999999,
                                                          2)
        if ok:
            Cfg._main_instance.acc_volt_changed(acc_volt)

    def load_spectra(self, file=None, loaded_spec=None, sum_sc_tr=None):
        if Cfg._main_instance.working_directory is None:
            if self.choose_working_dir() is None:
                return None
        if not isinstance(file, str):
            files = QtWidgets.QFileDialog.getOpenFileNames(
                self, 'choose an xml file', Cfg._main_instance.working_directory, '*.xml')[0]
        else:
            files = [file]
        if files:
            for file in files:
                if file not in self.file_plot_wins.keys():
                    self.open_file_plot_win(file, sum_sc_tr=sum_sc_tr)
                    Cfg._main_instance.load_spectra_to_main(file, self.file_plot_wins[file], loaded_spec=loaded_spec)
                else:
                    self.raise_win_to_front(self.file_plot_wins[file])

    ''' formatting '''

    def make_dmm_status_nice(self, status_dict):
        """
        will round teh readback of the dmms to 8 digits and return a nicely formatted status string
        :param status_dict: dict, keys as in self.update_status
        :return: str, linebreak for each new dmm.
        """
        ret = ''
        dmm_stat = status_dict.get('dmm_status', '')
        if dmm_stat:
            for dmm, dmm_dict in dmm_stat.items():
                ret += dmm
                stat = dmm_dict.get('status', '')
                ret += ', status: ' + stat
                read = dmm_dict.get('lastReadback', None)
                if read is not None:
                    ret += ', last reading: %.8f V | %s' % read
                ret += ' \n'
            return ret

    ''' configure '''
    def show_scan_finished_change(self, show_win_bool):
        """
        Activated from the ScanCompleteWin --> Change option
        """
        Cfg._main_instance.set_option('SCAN:show_scan_finished', show_win_bool)

    ''' open windows'''

    def open_dir(self):
        """
        Click to open the current TILDA working directory in the OS filesystem.
        """
        path = deepcopy(Cfg._main_instance.working_directory)
        if path:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        else:
            self.choose_working_dir()
            self.open_dir()

    def open_dmm_live_view_win(self):
        if self.dmm_live_view_win is None:
            self.dmm_live_view_win = DmmLiveViewUi(self)
        else:
            self.raise_win_to_front(self.dmm_live_view_win)

    def open_file_plot_win(self, file, sum_sc_tr=None):
        self.file_plot_wins[file] = TRSLivePlotWindowUi(full_file_path=file,
                                                        subscribe_as_live_plot=False,
                                                        sum_sc_tr=sum_sc_tr,
                                                        application=self.application)
        self.file_plot_wins[file].destroyed.connect(functools.partial(self.close_file_plot_win, file))

    def open_freq_win(self):
        if self.freq_win is None:
            self.freq_win = FreqUi(self)
        else:
            self.raise_win_to_front(self.freq_win)

    def open_live_plot_win(self):
        if self.live_plot_win is None:
            self.live_plot_win = TRSLivePlotWindowUi(application=self.application)
            self.live_plot_win.destroyed.connect(self.close_live_plot_win)
        else:
            self.raise_win_to_front(self.live_plot_win)
            self.live_plot_win.reset()

    def open_job_stacker_win(self):
        if Cfg._main_instance.working_directory is None:
            if self.choose_working_dir() is None:
                return None
        if self.job_stacker_win is None:
            self.job_stacker_win = JobStackerUi(self)
        else:
            self.raise_win_to_front(self.job_stacker_win)

    def open_options_win(self):
        if self.options_win is None:
            self.options_win = OptionsUi(self)
        else:
            self.raise_win_to_front(self.options_win)

    def open_pollifit_win(self):
        if self.pollifit_win is None:
            db = Cfg._main_instance.database
            if db is None:
                if self.choose_working_dir():
                    db = Cfg._main_instance.database
                else:
                    db = ''
            self.pollifit_win = PolliMainUi(db, self, overwrite_stdout=False)
        else:
            self.raise_win_to_front(self.pollifit_win)

    def open_post_acc_win(self):
        if self.post_acc_win is None:
            self.post_acc_win = PostAccControlUi(self)
        else:
            self.raise_win_to_front(self.post_acc_win)

    def open_pulse_pattern_win(self):
        if self.pulse_pattern_win is None:
            self.pulse_pattern_win = PulsePatternUi(None, '', self)
        else:
            self.raise_win_to_front(self.pulse_pattern_win)

    def open_scan_complete_win(self):
        if Cfg._main_instance.get_option('SCAN:show_scan_finished'):
            if self.scan_complete_win is None:
                play_sound = Cfg._main_instance.get_option('SOUND:is_on')
                sound_folder = Cfg._main_instance.get_option('SOUND:folder')
                self.scan_complete_win = ScanCompleteDialUi(self, play_sound, sound_folder)
            self.raise_win_to_front(self.scan_complete_win)

    def open_scan_ctrl_win(self):
        if Cfg._main_instance.working_directory is None:
            if self.choose_working_dir() is None:
                return None
        self.act_scan_wins.append(ScanControlUi(self))

    def open_simple_counter_win(self):
        sc_dial = SimpleCounterDialogUi()  # blocking!
        if sc_dial.start:
            self.simple_counter_gui = SimpleCounterRunningUi(self, sc_dial.act_pmts, sc_dial.datapoints)
            # Cfg._main_instance.start_simple_counter(sc_dial.act_pmts, sc_dial.datapoints)

    def open_version_win(self):
        VersionUi()

    def raise_win_to_front(self, window):
        # this will remove minimized status
        # and restore window with keeping maximized/normal state
        window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)

        # this will activate the window
        window.activateWindow()

    ''' close windows '''
    def scan_control_win_closed(self, win_ref):
        self.act_scan_wins.remove(win_ref)

    def close_freq_win(self):
        new_laser_freq = Cfg._main_instance.calc_freq()
        Cfg._main_instance.laser_freq_changed(new_laser_freq)
        self.freq_win = None

    def close_job_stacker_win(self):
        self.job_stacker_win = None

    def close_options_win(self):
        self.options_win = None

    def close_post_acc_win(self):
        self.post_acc_win = None

    def close_simple_counter_win(self):
        self.simple_counter_gui = None

    def close_dmm_live_view_win(self):
        self.dmm_live_view_win = None

    def close_live_plot_win(self):
        del self.live_plot_win
        self.live_plot_win = None
        #gc.collect()

        for scan_ctrl_win in self.act_scan_wins:
            scan_ctrl_win.enable_reopen_plot_win()
        logging.info('closed live plot window')

    def close_pollifit_win(self):
        self.pollifit_win = None

    def close_file_plot_win(self, file):
        logging.debug('removing file plot window from MainUi: %s' % file)
        if Cfg._main_instance is not None:
            Cfg._main_instance.close_spectra_in_main(file)

        del self.file_plot_wins[file]
        #gc.collect()
        logging.debug('remaining file plot wins are: ' + str(self.file_plot_wins))

    def close_pulse_pattern_win(self):
        if self.pulse_pattern_win:
            self.pulse_pattern_win = None

    def close_scan_complete_win(self):
        if self.scan_complete_win:
            self.scan_complete_win = None

    def closeEvent(self, *args, **kwargs):
        for win in self.act_scan_wins:
            logging.debug('will close: ' + str(win))
            try:
                win.close()
            except Exception as e:
                logging.error(str(e))
        try:
            if self.post_acc_win is not None:
                self.post_acc_win.close()
        except Exception as e:
            logging.error('error while closing post acceleration win:' + str(e))
        try:
            if self.scan_progress_win is not None:
                self.scan_progress_win.close()
        except Exception as e:
            logging.error('error while closing scan progress win:' + str(e))
        try:
            if self.freq_win is not None:
                self.freq_win.close()
        except Exception as e:
            logging.error('error while closing frequency win:' + str(e))
        try:
            if self.job_stacker_win is not None:
                self.job_stacker_win.close()
        except Exception as e:
            logging.error('error while closing job stacker win:' + str(e))
        try:
            if self.simple_counter_gui is not None:
                self.simple_counter_gui.close()
        except Exception as e:
            logging.error('error while closing simple counter win:' + str(e))
        try:
            MPLPlotter.close_all_figs()
        except Exception as e:
            logging.error('error while closing the plot window, exception is: ' + str(e))
        try:
            if self.dmm_live_view_win is not None:
                self.dmm_live_view_win.close()
        except Exception as e:
            logging.error('error while closing dmm_live_view_window, exception is: ' + str(e))
        try:
            if self.live_plot_win is not None:
                self.live_plot_win.close()
        except Exception as e:
            logging.error('error while closing dmm_live_view_window, exception is: ' + str(e))
        try:
            if self.pollifit_win is not None:
                self.pollifit_win.close()
        except Exception as e:
            logging.error('error while closing pollifit_win, exception is: ' + str(e))
        try:
            if self.pulse_pattern_win is not None:
                self.pulse_pattern_win.close()
        except Exception as e:
            logging.error('error while closing pulse_pattern_win, exception is: ' + str(e))
        list = [win for file, win in self.file_plot_wins.items()]
        for win in list:
            try:
                win.close()
            except Exception as e:
                logging.error(str(e))
        if self.tetris is not None:
            self.tetris.close()

        if self.snake is not None:
            self.snake.close()

    def start_tetris(self):
        self.tetris = Tetris()

    def start_snake(self):
        self.snake = Snake()
        self.snake.show()