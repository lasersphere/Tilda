"""
Created on 

@author: simkaufm

Module Description: Gui for a simple control of up to 3 post acceleration devices.
"""

import logging

from PyQt5 import QtWidgets, QtCore

import Application.Config as Cfg
from Interface.PostAccControlUi.Ui_PostAccControl import Ui_MainWindow_PostAcc


class PostAccControlUi(QtWidgets.QMainWindow, Ui_MainWindow_PostAcc):
    pacc_call_back_signal = QtCore.pyqtSignal(dict)

    def __init__(self, main_ui):
        super(PostAccControlUi, self).__init__()
        self.setupUi(self)

        self.subscription_name = 'PostAccControlUi'
        self.main_ui = main_ui
        self.scan_main = None
        self.post_acc_main = None
        self.last_status = {}

        self.pushButton_init_all.clicked.connect(self.init_pow_sups)
        self.pushButton_refresh.clicked.connect(self.get_status_of_power_sup)
        self.pushButton_all_on_off.clicked.connect(self.all_off)

        Cfg._main_instance.subscribe_to_power_sub_status(self.pacc_call_back_signal, self.subscription_name)
        self.pacc_call_back_signal.connect(self.rcvd_new_status_dict)

        self.show()

    def connect_buttons(self):
        """
        will connect the set and the on_off Pushbuttons with the corresponding functions.
        """
        for i in range(1, 4):
            try:
                name = self.resol_power_sup(i)
                if name != 'None':
                    getattr(self, 'pushButton_set_volt' + str(i)).clicked.connect(self.volt_set)
                    getattr(self, 'pushButton_on_off' + str(i)).clicked.connect(getattr(self, 'set_outp' + str(i)))
            except Exception as e:
                logging.error('while connecting the buttons for ' + name + ' the following exception occurred: ' +
                              ' \n\n' + str(e))

    def init_pow_sups(self):
        """
        updates the gui and calls the main to initialize all power supplies.
        """
        reset_labels = {}
        for i in range(1, 4):
            self.update_single_power_sup(reset_labels, str(i))
        Cfg._main_instance.init_power_sups(self.subscription_name)

    def rcvd_new_status_dict(self, status_dict_list):
        """
        this is called, whenever the call_back_signal is triggered.
        callback_signal should hold a list of status dictionaries.
        """
        for key, status_dict in status_dict_list.items():
            name = status_dict['name']
            self.update_single_power_sup(status_dict, name[-1:])
            self.last_status[name] = status_dict
        self.connect_buttons()

    def get_status_of_power_sup(self, button_val, name='all'):
        """
        request a status dict from the main.
        """
        Cfg._main_instance.power_supply_status(name, self.subscription_name)

    def update_single_power_sup(self, status_dict, num_str):
        """
        updates the Gui-labels for a single power supply.
        status_dict: dict containing all status infos.
        num_str: number of the device '1', '2', '3' possible
        """
        getattr(self, 'label_name' + num_str).setText(status_dict.get('name', 'None'))
        getattr(self, 'label_con' + num_str).setText(str(status_dict.get('com', -2) + 1))
        getattr(self, 'label_last_set' + num_str).setText(status_dict.get('voltageSetTime', 'None'))
        getattr(self, 'label_volt_read' + num_str).setText(str(status_dict.get('readBackVolt', 'None')))
        getattr(self, 'label_volt_read' + num_str).setText(str(status_dict.get('readBackVolt', 'None')))
        self.button_color('pushButton_on_off' + num_str, status_dict.get('output', None))

    def resol_power_sup(self, number):
        """
        by giving a number between 1 and 3, the function returns the text of the label,
        where the power supplies name is displayed.
        """
        name = getattr(self, 'label_name' + str(number)).text()
        return name

    def volt_set(self):
        sender = self.sender().text()
        name = self.resol_power_sup(int(sender[-1:]))
        if name is not None:
            voltage = getattr(self, 'doubleSpinBox_set_volt' + name[-1:]).value()
            if name is not None:
                Cfg._main_instance.set_power_supply_voltage(name, voltage, self.subscription_name)

    def set_outp(self, name, outp=None):
        if name is not None:
            status_dict = self.last_status.get(name)
            if status_dict is not None:
                cur_outp = status_dict.get('output', False)
                if outp is None:
                    new_outp = not cur_outp  # if not specified, just toggle the output status
                else:
                    new_outp = outp
                Cfg._main_instance.set_power_sup_outp(name, new_outp, self.subscription_name)

    def set_outp1(self):
        self.set_outp(self.resol_power_sup(1))

    def set_outp2(self):
        self.set_outp(self.resol_power_sup(2))

    def set_outp3(self):
        self.set_outp(self.resol_power_sup(3))

    def all_off(self):
        Cfg._main_instance.set_power_sup_outp('all', False, self.subscription_name)

    def button_color(self, butn, on_off):
        """
        this will change the background color of the output switch for each device.
        Green - On
        Red - Off
        readback of the output is not supported in Heiniznger Powersupplies,
        so this depends on the init of the device and storage of the output variable
        """
        if on_off is not None:
            if on_off:
                getattr(self, butn).setStyleSheet("background-color: green")
            else:
                getattr(self, butn).setStyleSheet("background-color: red")
        else:
            getattr(self, butn).setStyleSheet("background-color: light gray")

    def closeEvent(self, *args, **kwargs):
        """
        unsubscribe in the corresponding main
        """
        Cfg._main_instance.un_subscribe_to_power_sub_status(self.subscription_name)
        self.main_ui.close_post_acc_win()
