"""

Created on '07.05.2015'

@author:'simkaufm'

"""
from PyQt5 import QtWidgets
import sys
import time

from Interface.MainUi.MainUi import MainUi

def main():
    """
    main loop of tilda
    """
    start_gui()


def start_gui():
    app = QtWidgets.QApplication(sys.argv)
    ui = MainUi()
    app.exec_()

if __name__ == "__main__":
    main()
