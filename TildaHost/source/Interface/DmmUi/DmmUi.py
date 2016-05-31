"""
Created on 

@author: simkaufm

Module Description: Main Ui for controlling the digital Multimeters connected to the PXI-Crate
"""

from PyQt5 import QtWidgets, QtCore

from Interface.DmmUi.Ui_DmmLiveView import Ui_MainWindow
from Interface.DmmUi.ChooseDmmWidget import ChooseDmmWidget
from Interface.DmmUi.DMMWidgets import get_wid_by_type
import Application.Config as Cfg


class DmmLiveViewUi(QtWidgets.QMainWindow, Ui_MainWindow):
    callback_from_choose_dmm = QtCore.pyqtSignal(tuple)

    def __init__(self, parent):
        super(DmmLiveViewUi, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('DMM Live View Window')
        self.parent_ui = parent
        self.dmm_types = Cfg._main_instance.scan_main.digital_multi_meter.types
        self.tabs = {'tab0': [self.tab_0, None]}
        self.tabWidget.setTabText(0, 'choose dmm')
        self.tabs['tab0'][1] = QtWidgets.QVBoxLayout(self.tabs['tab0'][0])
        self.choose_dmm_wid = ChooseDmmWidget(self.callback_from_choose_dmm, self.dmm_types)
        self.tabs['tab0'][1].addWidget(self.choose_dmm_wid)

        self.callback_from_choose_dmm.connect(self.initialize_dmm)
        self.show()

    def initialize_dmm(self, rcv_tpl):
        dev_type, dev_address = rcv_tpl
        tab_name = Cfg._main_instance.scan_main.prepare_dmm(dev_type, dev_address)
        if self.setup_new_tab_widget(tab_name, dev_type) is None:
            return None

    def setup_new_tab_widget(self, tab_name, dev_type):
        if tab_name in list(self.tabs.keys()) or tab_name is None:
            print('could not initialize: ', tab_name, ' ... already initialized?')
            return None
        self.tabs[tab_name] = [None, None]
        self.tabs[tab_name][0] = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tabs[tab_name][0], tab_name)
        self.tabs[tab_name][1] = QtWidgets.QVBoxLayout(self.tabs[tab_name][0])
        self.tabWidget.setCurrentWidget(self.tabs[tab_name][0])
        self.tabs[tab_name][1].addWidget(get_wid_by_type(dev_type, tab_name))
        return True

    def closeEvent(self, *args, **kwargs):
        self.parent_ui.close_dmm_live_view_win()

# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     ui = DmmLiveViewUi()
#     ui.show()
#     app.exec()
