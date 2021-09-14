"""

Created on '10.09.2019'

@author:'fsommer'

"""


from PyQt5 import QtWidgets, QtCore, QtGui
import Service.FileOperations.FolderAndFileHandling as FileHandler

import Application.Config as Cfg

from Interface.OptionsUi.Ui_Options import Ui_Dialog_Options


class OptionsUi(QtWidgets.QWidget, Ui_Dialog_Options):

    def __init__(self, main_gui):
        super(OptionsUi, self).__init__()

        self.main_gui = main_gui
        self.setupUi(self)
        self.show()

    def accept(self):
        pass

    def reject(self):
        pass

    ''' window related '''

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        if self.main_gui is not None:
            # tell main window that this window is closed.
            self.main_gui.close_options_win()
