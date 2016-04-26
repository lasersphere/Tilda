"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import logging
import argparse
import sys
import functools

sys.path.append('..\\..\\PolliFit\\source')

from Application.Main.Main import Main
import Application.Config as Cfg

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
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(message)s', stream=sys.stdout)
    logging.info('Log level set to ' + args.log_level)

    # starting the main loop and storing the instance in Cfg.main_instance
    Cfg._main_instance = Main()

    start_gui()


def start_gui():
    """
    starts the gui for the main window.
    :parameter: main, instacne of the Tilda Main() module
    """
    from Interface.MainUi.MainUi import MainUi
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import QTimer, Qt

    app = QtWidgets.QApplication(sys.argv)
    ui = MainUi()
    timer = QTimer()
    timer.setTimerType(Qt.PreciseTimer)
    timer.setInterval(_cyclic_interval_ms)
    timer_call_back = functools.partial(cyclic, ui=ui)
    timer.timeout.connect(timer_call_back)
    timer.start()
    app.exec_()
    return ui


def cyclic(ui):
    """
    periodic execution of these functions, when timer timesout, after _cyclic_interval_ms
    -> all calls should be brief, otherwise Gui is blocked
    """
    Cfg._main_instance.cyclic()

if __name__ == "__main__":
    main()
    Cfg._main_instance.close_main()
