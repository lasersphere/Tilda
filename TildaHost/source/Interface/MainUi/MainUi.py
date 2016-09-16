"""

Created on '07.05.2015'

@author:'simkaufm'

"""

""" Guis: """
import logging
import os
import platform
import subprocess
from copy import deepcopy

from PyQt5 import QtCore
from PyQt5 import QtWidgets

import Application.Config as Cfg
import MPLPlotter as MPlPlotter
from Gui.MainUi import MainUi as PolliMainUi
from Interface.DmmUi.DmmUi import DmmLiveViewUi
from Interface.LiveDataPlottingUi.LiveDataPlottingUi import TRSLivePlotWindowUi
from Interface.MainUi.Ui_Main import Ui_TildaMainWindow
from Interface.PostAccControlUi.PostAccControlUi import PostAccControlUi
from Interface.ScanControlUi.ScanControlUi import ScanControlUi
from Interface.ScanProgressUi.ScanProgressUi import ScanProgressUi
from Interface.SimpleCounter.SimpleCounterDialogUi import SimpleCounterDialogUi
from Interface.SimpleCounter.SimpleCounterRunningUi import SimpleCounterRunningUi
from Interface.TildaPassiveUi.TildaPassiveUi import TildaPassiveUi
from Interface.VersionUi.VersionUi import VersionUi


class MainUi(QtWidgets.QMainWindow, Ui_TildaMainWindow):
    main_ui_status_call_back_signal = QtCore.pyqtSignal(dict)

    def __init__(self):
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
        self.simple_counter_gui = None
        self.tilda_passive_gui = None
        self.dmm_live_view_win = None
        self.live_plot_win = None  # one active live plot window for displaying results from pipeline
        self.file_plot_wins = {}  # dict of active plot windows only for displaying from file.
        self.pollifit_win = None

        self.actionWorking_directory.triggered.connect(self.choose_working_dir)
        self.actionVersion.triggered.connect(self.open_version_win)
        self.actionScan_Control.triggered.connect(self.open_scan_ctrl_win)
        self.actionPost_acceleration_power_supply_control.triggered.connect(self.open_post_acc_win)
        self.actionSimple_Counter.triggered.connect(self.open_simple_counter_win)
        self.actionSet_Laser_Frequency.triggered.connect(self.set_laser_freq)
        self.actionSet_acceleration_voltage.triggered.connect(self.set_acc_volt)
        self.actionTilda_Passive.triggered.connect(self.start_tilda_passive_gui)
        self.actionLoad_spectra.triggered.connect(self.load_spectra)
        self.actionDigital_Multimeters.triggered.connect(self.open_dmm_live_view_win)
        self.actionPolliFit.triggered.connect(self.open_pollifit_win)

        """ connect double clicks on labels:"""
        self.label_workdir_set.mouseDoubleClickEvent = self.workdir_dbl_click
        self.label_laser_freq_set.mouseDoubleClickEvent = self.laser_freq_dbl_click
        self.label_acc_volt_set.mouseDoubleClickEvent = self.acc_volt_dbl_click
        self.label_8.mouseDoubleClickEvent = self.dmm_setup_dbl_click

        """ connect buttons """
        self.pushButton_open_dir.clicked.connect(self.open_dir)

        self.subscribe_to_main()
        self.show()


    ''' connected actions '''
    def workdir_dbl_click(self, event):
        self.choose_working_dir()

    def laser_freq_dbl_click(self, event):
        self.set_laser_freq()

    def acc_volt_dbl_click(self, event):
        self.set_acc_volt()

    def dmm_setup_dbl_click(self, event):
        self.open_dmm_live_view_win()

    def subscribe_to_main(self):
        """
        pass the call back signal to the main and connect to self.update_status
        """
        Cfg._main_instance.gui_status_subscribe(self.main_ui_status_call_back_signal)
        self.main_ui_status_call_back_signal.connect(self.update_status)
        Cfg._main_instance.send_state()

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
        self.label_laser_freq_set.setText(str(status_dict.get('laserfreq', '')))
        self.label_acc_volt_set.setText(str(status_dict.get('accvolt', '')))
        self.label_fpga_state_set.setText(str(status_dict.get('fpga_status', '')))
        self.label_sequencer_status_set.setText(str(status_dict.get('sequencer_status', '')))
        self.label_dmm_status.setText(self.make_dmm_status_nice(status_dict))
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

    def set_laser_freq(self):
        laser_freq, ok = QtWidgets.QInputDialog.getDouble(self, 'Laser', 'laser wavenumber [cm-1]',
                                                          0, 0, 9999999,
                                                          5)
        if ok:
            Cfg._main_instance.laser_freq_changed(laser_freq)

    def set_acc_volt(self):
        acc_volt, ok = QtWidgets.QInputDialog.getDouble(self, 'Acceleration Voltage', 'acceleration voltage [V]',
                                                          0, 0, 9999999,
                                                          2)
        if ok:
            Cfg._main_instance.acc_volt_changed(acc_volt)

    def load_spectra(self):
        if Cfg._main_instance.working_directory is None:
            if self.choose_working_dir() is None:
                return None
        file = QtWidgets.QFileDialog.getOpenFileName(
            self, 'choose an xml file', Cfg._main_instance.working_directory, '*.xml')[0]
        if file not in self.file_plot_wins.keys():
            self.open_file_plot_win(file)
            Cfg._main_instance.load_spectra_to_main(file, self.file_plot_wins[file])

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

    ''' open windows'''
    def open_version_win(self):
        VersionUi()

    def open_scan_ctrl_win(self):
        if Cfg._main_instance.working_directory is None:
            if self.choose_working_dir() is None:
                return None
        self.act_scan_wins.append(ScanControlUi(self))

    def open_scan_progress_win(self):
        try:
            geom = False
            print('opening scan progress window')
            if self.scan_progress_win is not None:
                geom = self.scan_progress_win.geometry()
            self.scan_progress_win = ScanProgressUi(self)
            if geom:
                self.scan_progress_win.setGeometry(geom)
        except Exception as e:
            print('erroror:', e)
        # pass

    def open_post_acc_win(self):
        self.post_acc_win = PostAccControlUi(self)

    def open_simple_counter_win(self):
        sc_dial = SimpleCounterDialogUi()  # blocking!
        if sc_dial.start:
            self.simple_counter_gui = SimpleCounterRunningUi(self, sc_dial.act_pmts, sc_dial.datapoints)
            # Cfg._main_instance.start_simple_counter(sc_dial.act_pmts, sc_dial.datapoints)

    def start_tilda_passive_gui(self):
            if Cfg._main_instance.working_directory is None:
                if self.choose_working_dir() is None:
                    return None
            if self.tilda_passive_gui is None:
                self.tilda_passive_gui = TildaPassiveUi(self)

    def open_dmm_live_view_win(self):
        if self.dmm_live_view_win is None:
            self.dmm_live_view_win = DmmLiveViewUi(self)
        else:
            self.raise_win_to_front(self.dmm_live_view_win)

    def open_live_plot_win(self):
        if self.live_plot_win is None:
            self.live_plot_win = TRSLivePlotWindowUi(parent=self)
        else:
            self.raise_win_to_front(self.live_plot_win)

    def open_file_plot_win(self, file):
        self.file_plot_wins[file] = TRSLivePlotWindowUi(full_file_path=file,
                                                        parent=self,
                                                        subscribe_as_live_plot=False)

    def open_pollifit_win(self):
        if self.pollifit_win is None:
            db = Cfg._main_instance.database
            if db is None:
                self.choose_working_dir()
                db = Cfg._main_instance.database
            self.pollifit_win = PolliMainUi(db, self, overwrite_stdout=False)
        else:
            self.raise_win_to_front(self.pollifit_win)

    def raise_win_to_front(self, window):
        # this will remove minimized status
        # and restore window with keeping maximized/normal state
        window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)

        # this will activate the window
        window.activateWindow()

    def open_dir(self):
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


    ''' close windows '''
    def scan_control_win_closed(self, win_ref):
        self.act_scan_wins.remove(win_ref)

    def close_scan_progress_win(self):
        self.scan_progress_win = None

    def close_post_acc_win(self):
        self.post_acc_win = None

    def close_simple_counter_win(self):
        self.simple_counter_gui = None

    def close_tilda_passive(self):
        self.tilda_passive_gui = None

    def close_dmm_live_view_win(self):
        self.dmm_live_view_win = None

    def close_live_plot_win(self):
        self.live_plot_win = None

    def close_pollifit_win(self):
        self.pollifit_win = None

    def close_file_plot_win(self, file):
        self.file_plot_wins.pop(file)

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
            if self.simple_counter_gui is not None:
                self.simple_counter_gui.close()
        except Exception as e:
            logging.error('error while closing simple counter win:' + str(e))
        try:
            MPlPlotter.close_all_figs()
        except Exception as e:
            logging.error('error while closing the plot window, exception is: ' + str(e))
        try:
            if self.tilda_passive_gui is not None:
                self.tilda_passive_gui.close()
        except Exception as e:
            logging.error('error while closing tilda passive GUi, exception is: ' + str(e))
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
        list = [win for file, win in self.file_plot_wins.items()]
        for win in list:
            try:
                win.close()
            except Exception as e:
                logging.error(str(e))


