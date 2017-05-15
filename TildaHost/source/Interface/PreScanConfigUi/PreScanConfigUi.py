"""

Created on '15.05.2017'

@author:'simkaufm'

description: Main gui for configuring the pre / post scan settings

"""

from PyQt5 import QtWidgets

from Interface.DmmUi.DmmUi import DmmLiveViewUi
from Interface.PreScanConfigUi.Ui_PreScanMain import Ui_PreScanMainWin


class PreScanConfigUi(QtWidgets.QMainWindow, Ui_PreScanMainWin):
    def __init__(self):
        super(PreScanConfigUi, self).__init__()
        self.setupUi(self)
        self.show()

        self.buttonBox.accepted.connect(self.confirm)
        self.buttonBox.rejected.connect(self.rejected)

        self.comboBox.addItems(['preScan', 'duringScan', 'afterScan'])

        self.voltage_mainwidget = DmmLiveViewUi(self)

    def confirm(self):
        print('confirmed')
        self.close()

    def rejected(self):
        print('rejected')
        self.close()

    ''' voltage related '''



if __name__=='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    gui = PreScanConfigUi()
    app.exec_()
