"""
Created on 

@author: simkaufm

Module Description: Sequencer Widget for the Kepco scan.
 This will be just a button in order to open the DMM-ctrl window.
"""
from PyQt5 import QtWidgets

from Interface.SequencerWidgets.BaseSequencerWidg import BaseSequencerWidgUi


class KepcoScanWidg(BaseSequencerWidgUi, QtWidgets.QWidget):
    def __init__(self, track_dict, main_gui):
        super(KepcoScanWidg, self).__init__(track_dict)
        self.main_gui = main_gui
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.button_for_open_config = QtWidgets.QPushButton('config dmm')
        self.button_for_open_config.setMaximumSize(100, 50)
        self.button_for_open_config.clicked.connect(self.open_dmm_win)
        self.layout.addWidget(self.button_for_open_config)

    def set_type(self):
        self.type = 'kepco'

    def connect_labels(self):
        # this is done already inside the DMM-Widget
        pass

    def open_dmm_win(self):
        self.main_gui.open_dmm_live_view_win()

    def set_vals_by_dict(self):
        pass


        # if __name__ == "__main__":
        #     app = QtWidgets.QApplication(sys.argv)
        #     ui = KepcoScanWidg({}, None)
        #     ui.show()
        #     app.exec()
