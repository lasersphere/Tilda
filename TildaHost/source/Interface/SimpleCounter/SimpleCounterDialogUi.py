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

        self.buttonBox.accepted.connect(self.close)
        self.buttonBox.rejected.connect(self.close)
        # for now just stop the simple counter.
        # still there should be the option to set a voltage in the future.

        self.exec_()

    def closeEvent(self, QCloseEvent):
        Cfg._main_instance.set_state('stop_simple_counter')
        # Cfg._main_instance.stop_simple_counter()