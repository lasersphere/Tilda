"""

Created on '07.05.2015'

@author:'simkaufm'

"""
import logging
import argparse
import sys

from Application.Main.Main import Main

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

    # starting the main loop
    main = Main()

    start_gui(main)


def start_gui(main):
    """
    starts the gui for the main window.
    :parameter: main, instacne of the Tilda Main() module
    """
    from Interface.MainUi.MainUi import MainUi
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    ui = MainUi(main)
    app.exec_()
    return ui


if __name__ == "__main__":
    main()
