"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import argparse
import functools
import logging
from logging.handlers import RotatingFileHandler
import sys
import os

import matplotlib

matplotlib.use('Qt5Agg')

sys.path.append('..\\..\\PolliFit\\source')

from Application.Main.Main import Main
import Application.Config as Cfg

_cyclic_interval_ms = 5000


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

    debug_file = './logs/debug'
    error_file = './logs/err'
    if not os.path.isdir(os.path.dirname(debug_file)):
        os.mkdir(os.path.dirname(debug_file))

    app_log = logging.getLogger()
    # app_log.setLevel(getattr(logging, args.log_level))
    app_log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, args.log_level))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # ch.setFormatter(log_formatter)
    app_log.addHandler(ch)

    my_handler = RotatingFileHandler(debug_file, mode='a', maxBytes=5 * 1024 * 1024,
                                     backupCount=2, encoding=None, delay=0)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.DEBUG)
    app_log.addHandler(my_handler)

    my_handler = RotatingFileHandler(error_file, mode='a', maxBytes=5 * 1024 * 1024,
                                     backupCount=2, encoding=None, delay=0)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.ERROR)
    app_log.addHandler(my_handler)

    app_log.info('****************************** starting ******************************')
    app_log.info('Log level set to ' + args.log_level)

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
    ui = MainUi()
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
