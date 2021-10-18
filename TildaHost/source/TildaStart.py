"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import argparse
import logging
from logging.handlers import RotatingFileHandler
import sys
import os
import subprocess

import matplotlib

matplotlib.use('Qt5Agg')

sys.path.append('..\\..\\PolliFit\\source')

from Application.Main.Main import Main
import Application.Config as Cfg
from Service import LoggingUtil


_cyclic_interval_ms = 50


def main():
    """
    main loop of tilda
    """
    # Parser argument
    parser = argparse.ArgumentParser(description='Start Tilda')
    parser.add_argument('--log_level', '-l', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    args = parser.parse_args()

    # setup logging
    # logging.basicConfig(level=getattr(logging, args.log_level), format='%(message)s', stream=sys.stdout)
    # logging.info('Log level set to ' + args.log_level)

    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s %(funcName)s(%(lineno)d) %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    debug_file = './logs/debug'
    error_file = './logs/err'
    timing_file = './logs/timing'
    if not os.path.isdir(os.path.dirname(debug_file)):
        os.mkdir(os.path.dirname(debug_file))

    app_log = logging.getLogger()
    # app_log.setLevel(getattr(logging, args.log_level))
    app_log.setLevel(logging.NOTSET)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, args.log_level))
    # ch.setFormatter(formatter)
    ch.setFormatter(log_formatter)
    app_log.addHandler(ch)

    my_debug_handler = RotatingFileHandler(debug_file, mode='a', maxBytes=5 * 1024 * 1024,
                                     backupCount=10, encoding=None, delay=0)
    my_debug_handler.setFormatter(log_formatter)
    my_debug_handler.setLevel(logging.DEBUG)
    app_log.addHandler(my_debug_handler)

    my_err_handler = RotatingFileHandler(error_file, mode='a', maxBytes=5 * 1024 * 1024,
                                     backupCount=10, encoding=None, delay=0)
    my_err_handler.setFormatter(log_formatter)
    my_err_handler.setLevel(logging.ERROR)
    app_log.addHandler(my_err_handler)

    LoggingUtil.add_logging_level('TIMING', 5)  # create a new logging level for timing even below DEBUG(10)

    my_time_handler = RotatingFileHandler(timing_file, mode='a', maxBytes=5 * 1024 * 1024,
                                         backupCount=1, encoding=None, delay=0)
    my_time_handler.setFormatter(formatter)
    my_time_handler.setLevel(logging.TIMING)
    my_time_handler.addFilter(LoggingUtil.LevelFilter(5, 5))  # Filter to only show timings
    app_log.addHandler(my_time_handler)

    app_log.info('****************************** starting ******************************')
    app_log.info('Log level set to ' + args.log_level)

    # get details on current version
    try:
        # get the current branch
        branch = subprocess.check_output(['git', 'symbolic-ref', '--short', 'HEAD']).decode('utf-8').replace('\n', '')
        # get uniquely abbreviated commit object (or commit tag)
        commit = subprocess.check_output(['git', 'describe', '--always']).decode('utf-8').replace('\n', '')
        # update info in config
        Cfg.branch = branch
        Cfg.commit = commit
        # display info for user
        app_log.info('Detected branch: {}, commit: {}'.format(branch, commit))
    except Exception as e:
        app_log.warning('Could not detect git branch and commit. Error: {}'.format(e))

    # starting the main loop and storing the instance in Cfg.main_instance
    Cfg._main_instance = Main()

    start_gui()


def start_gui():
    """
    starts the gui for the main window.
    :parameter: main, instacne of the Tilda Main() module
    """
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import QTimer, Qt
    from Interface.MainUi.MainUi import MainUi

    app = QtWidgets.QApplication(sys.argv)
    ui = MainUi(app)
    Cfg._main_instance.application = app
    timer = QTimer()
    timer.setTimerType(Qt.PreciseTimer)
    timer.setInterval(_cyclic_interval_ms)
    # timer_call_back = functools.partial(cyclic, ui=ui)
    timer.timeout.connect(cyclic)
    timer.start()
    app.exec_()
    app.closeAllWindows()

    return ui


def cyclic():
    """
    periodic execution of these functions, when timer timesout, after _cyclic_interval_ms
    -> all calls should be brief, otherwise Gui is blocked
    """
    Cfg._main_instance.cyclic()


if __name__ == "__main__":
    main()
    Cfg._main_instance.close_main()
