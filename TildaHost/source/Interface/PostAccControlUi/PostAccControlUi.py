"""
Created on 

@author: simkaufm

Module Description:
"""

from PyQt5 import QtWidgets
from copy import copy
from Interface.PostAccControlUi.Ui_PostAccControl import Ui_MainWindow_PostAcc


class PostAccControlUi(QtWidgets.QMainWindow, Ui_MainWindow_PostAcc):
    def __init__(self, main):
        super(PostAccControlUi, self).__init__()
        self.main = main
        self.setupUi(self)

        self.scan_main = main.scan_main
        self.post_acc_main = main.scan_main.post_acc_main

        self.update_power_sups_gui()
        # self.label_name1.setText('Test')
        # self.label_con1.text()
        # self.doubleSpinBox_set_volt1.value()
        for i in range(1, 4):
            getattr(self, 'pushButton_set_volt' + str(i)).connect(self.volt_set(*self.resol_power_sup(i)))
            getattr(self, 'pushButton_on_off' + str(i)).connect(self.set_outp(*self.resol_power_sup(i)))
        # self.label_last_set1
        # self.label_volt_read1
        # self.pushButton_on_off1
        self.pushButton_init_all.clicked.connect(self.init_pow_sups)
        self.pushButton_refresh.clicked.connect(self.update_power_sups_gui)
        # self.pushButton_all_on_off

        self.show()

    def init_pow_sups(self):
        self.scan_main.init_post_accel_pwr_supplies()
        self.update_power_sups_gui()

    def update_power_sups_gui(self):
        act_dict = self.post_acc_main.active_power_supplies
        for name, instance in act_dict.items():
            status_dict = self.scan_main.get_status_of_pwr_supply(name)
            self.update_single_power_sup(status_dict, name[-1:])

    def update_single_power_sup(self, status_dict, num_str):
        getattr(self, 'label_name' + num_str).setText(status_dict.get('name'))
        getattr(self, 'label_con' + num_str).setText(str(status_dict.get('com') + 1))
        getattr(self, 'label_last_set' + num_str).setText(status_dict.get('voltageSetTime'))
        getattr(self, 'label_volt_read' + num_str).setText(str(status_dict.get('readBackVolt')))
        getattr(self, 'label_volt_read' + num_str).setText(str(status_dict.get('readBackVolt')))
        self.button_color(getattr(self, 'pushButton_on_off' + num_str), status_dict.get('output'))

    def resol_power_sup(self, number):
        name = getattr(self, 'label_name' + str(number)).text()
        voltage = getattr(self, 'doubleSpinBox_set_volt' + str(number)).value()
        return name, voltage

    def volt_set(self, name, voltage):
        if name is not None:
            self.scan_main.set_Voltage(name, voltage)
            self.update_power_sups_gui()

    def set_outp(self, name, voltage):
        status_dict = self.scan_main.get_status_of_pwr_supply(name)
        cur_outp = status_dict.get('output', False)
        self.scan_main.set_post_accel_pwr_spply_output(name, not cur_outp)
        self.update_power_sups_gui()

    def button_color(self, butn, on_off):
        if on_off:
            getattr(self, butn).setStyleSheet("background-color: green")
        else:
            getattr(self, butn).setStyleSheet("background-color: red")

    def closeEvent(self, *args, **kwargs):
        self.main.closed_post_acc_win()
