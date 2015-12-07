"""

Created on '25.11.2015'

@author:'simkaufm'

Description: Dialog to control the Simple Counter while this is running.
"""

from PyQt5 import QtWidgets, QtGui

from Interface.SimpleCounter.Ui_Simp_Count_Dial import Ui_Dialog_simpleCounterControl
import Application.Config as Cfg

class SimpleCounterDialogUi(QtWidgets.QDialog, Ui_Dialog_simpleCounterControl):
    def __init__(self):
        super(SimpleCounterDialogUi, self).__init__()

        self.setupUi(self)
        self.exec_()

        self.buttonBox.accepted.clicked.connect(self.closeEvent)
        self.buttonBox.rejected.clicked.connect(self.closeEvent)

    def closeEvent(self, QCloseEvent):
        Cfg._main_instance.stop_simple_counter()