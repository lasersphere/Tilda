"""
Created on 

@author: simkaufm

Module Description: Main Ui for controlling the digital Multimeters connected to the PXI-Crate
"""

from PyQt5 import QtWidgets, QtCore
import functools

from Interface.DmmUi.Ui_DmmLiveView import Ui_MainWindow
from Interface.DmmUi.ChooseDmmWidget import ChooseDmmWidget
from Interface.DmmUi.DMMWidgets import get_wid_by_type
import Application.Config as Cfg


class DmmLiveViewUi(QtWidgets.QMainWindow, Ui_MainWindow):
    # callback for the widget when choosing a new dmm
    # returns tuple of (dev_type, dev_adress) out of ChooseDmmWidget
    init_dmm_clicked_callback = QtCore.pyqtSignal(tuple)
    # callback to learn when the main is done with the init of the device.
    init_dmm_done_callback = QtCore.pyqtSignal(bool)

    def __init__(self, parent):
        super(DmmLiveViewUi, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('DMM Live View Window')
        self.parent_ui = parent
        self.dmm_types = Cfg._main_instance.scan_main.digital_multi_meter.types
        self.tabs = {'tab0': [self.tab_0, None]}
        self.tabWidget.setTabText(0, 'choose dmm')
        self.tabs['tab0'][1] = QtWidgets.QVBoxLayout(self.tabs['tab0'][0])
        self.init_dmm_clicked_callback.connect(self.initialize_dmm)

        self.choose_dmm_wid = ChooseDmmWidget(self.init_dmm_clicked_callback, self.dmm_types)
        self.tabs['tab0'][1].addWidget(self.choose_dmm_wid)

        self.show()

    def initialize_dmm(self, rcv_tpl):
        print('starting to initialize: ', rcv_tpl)
        dev_type, dev_address = rcv_tpl
        dmm_name = dev_type + '_' + dev_address
        if dmm_name in list(self.tabs.keys()) or dmm_name is None:
            print('could not initialize: ', dmm_name, ' ... already initialized?')
            return None  # break when not initialized
        Cfg._main_instance.init_dmm(dev_type, dev_address, self.init_dmm_done_callback)
        self.init_dmm_done_callback.connect(functools.partial(self.setup_new_tab_widget, (dmm_name, dev_type)))

    def setup_new_tab_widget(self, tpl):
        tab_name, dev_type = tpl
        print('done initializing: ', tab_name, dev_type)
        self.init_dmm_done_callback.disconnect()
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
