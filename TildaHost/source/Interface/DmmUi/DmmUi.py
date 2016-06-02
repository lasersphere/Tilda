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
    # callback to learn when the main is done with the init/deinit of the device.
    init_dmm_done_callback = QtCore.pyqtSignal(bool)
    deinit_dmm_done_callback = QtCore.pyqtSignal(bool)
    # callback for the voltage readings, done by the main when in idle state
    voltage_reading = QtCore.pyqtSignal(dict)

    def __init__(self, parent):
        super(DmmLiveViewUi, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('DMM Live View Window')
        self.parent_ui = parent
        self.dmm_types = Cfg._main_instance.scan_main.digital_multi_meter.types
        self.tabs = {
            'tab0': [self.tab_0, None, None]}  # dict for storing all tabs, key: [QWidget(), Layout, userWidget]
        self.tabWidget.setTabText(0, 'choose dmm')
        self.tabs['tab0'][1] = QtWidgets.QVBoxLayout(self.tabs['tab0'][0])
        self.init_dmm_clicked_callback.connect(self.initialize_dmm)
        self.check_for_already_active_dmms()

        self.choose_dmm_wid = ChooseDmmWidget(self.init_dmm_clicked_callback, self.dmm_types)
        self.tabs['tab0'][2] = self.tabs['tab0'][1].addWidget(self.choose_dmm_wid)

        Cfg._main_instance.dmm_gui_subscribe(self.voltage_reading)
        self.voltage_reading.connect(self.rcvd_voltage_dict)

        self.tabWidget.setTabsClosable(True)

        self.tabWidget.tabCloseRequested.connect(self.tab_wants_to_be_closed)

        self.show()

    def tab_wants_to_be_closed(self, *args):
        tab_ind = args[0]
        if tab_ind:
            dmm_name = self.tabWidget.tabText(tab_ind)
            self.deinit_dmm(dmm_name)
            self.tabs.pop(dmm_name)
            self.tabWidget.removeTab(tab_ind)

    def initialize_dmm(self, rcv_tpl):
        print('starting to initialize: ', rcv_tpl)
        dev_type, dev_address = rcv_tpl
        dmm_name = dev_type + '_' + dev_address
        if dmm_name in list(self.tabs.keys()) or dmm_name is None:
            print('could not initialize: ', dmm_name, ' ... already initialized?')
            return None  # break when not initialized
        Cfg._main_instance.init_dmm(dev_type, dev_address, self.init_dmm_done_callback)
        self.init_dmm_done_callback.connect(functools.partial(self.setup_new_tab_widget, (dmm_name, dev_type)))

    def deinit_dmm(self, dmm_name):
        Cfg._main_instance.deinit_dmm(dmm_name)

    def check_for_already_active_dmms(self):
        act_dmm_dict = Cfg._main_instance.get_active_dmms()
        for key, val in act_dmm_dict.items():
            dmm_type, dmm_addr, dmm_config = val
            self.setup_new_tab_widget((key, dmm_type), False)

    def setup_new_tab_widget(self, tpl, disconnect_signal=True):
        dmm_name, dev_type = tpl  # dmm_name = tab_name
        print('done initializing: ', dmm_name, dev_type)
        if disconnect_signal:
            self.init_dmm_done_callback.disconnect()
        self.tabs[dmm_name] = [None, None, None]
        self.tabs[dmm_name][0] = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tabs[dmm_name][0], dmm_name)
        self.tabs[dmm_name][1] = QtWidgets.QVBoxLayout(self.tabs[dmm_name][0])
        self.tabWidget.setCurrentWidget(self.tabs[dmm_name][0])
        self.tabs[dmm_name][2] = get_wid_by_type(dev_type, dmm_name)
        self.tabs[dmm_name][1].addWidget(self.tabs[dmm_name][2])
        return True

    def rcvd_voltage_dict(self, voltage_dict):
        """
        will be emitted by the main when reading a voltage during idle phase.
        :param voltage_dict: dict, {dmm_name: np.array(containing readbacks) or None}
        """
        for key, val in self.tabs.items():
            read = voltage_dict.get(key, None)
            if read is not None:
                self.tabs[key][2].new_voltage(read[-1])

    def closeEvent(self, *args, **kwargs):
        Cfg._main_instance.dmm_gui_unsubscribe()
        self.voltage_reading.disconnect()
        self.parent_ui.close_dmm_live_view_win()

# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     ui = DmmLiveViewUi()
#     ui.show()
#     app.exec()
