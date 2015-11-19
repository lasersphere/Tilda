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

        self.update_power_sups()
        # self.label_name1
        # self.label_con1
        # self.doubleSpinBox_set_volt1
        # self.label_last_set1
        # self.label_volt_read1
        # self.pushButton_on_off1
        self.pushButton_init_all.clicked.connect(self.init_pow_sups)
        # self.pushButton_refresh
        # self.pushButton_all_on_off

        self.show()

    def init_pow_sups(self):
        self.scan_main.init_post_accel_pwr_supplies()
        print(self.scan_main.post_accel_pwr_supplies.active_power_supplies)

    def update_power_sups_gui(self):
        act_dict = self.post_acc_main.active_power_supplies
        for name, instance in act_dict:
            status_dict = self.post_acc_main.get_status_of_pwr_supply()
            self.update_single_power_sup(status_dict, name[-1:])

    def update_single_power_sup(self, status_dict, num_str):
        getattr(self, '.label_name.' + num_str + 'setText(' + status_dict['name'] + ')')

    def closeEvent(self, *args, **kwargs):
        self.main.closed_post_acc_win()
