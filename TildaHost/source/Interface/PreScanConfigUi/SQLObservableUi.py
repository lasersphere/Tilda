"""
Created on 2022-12-22

@author: Patrick Mueller

Module Description: UI class for SQL-Observables.
"""


from PyQt5 import QtWidgets

from Interface.PreScanConfigUi.Ui_SQLObservable import Ui_SQLObservable


class SQLObservableUi(QtWidgets.QWidget, Ui_SQLObservable):

    def __init__(self, parent):
        super(SQLObservableUi, self).__init__(parent)
        self.setupUi(self)
