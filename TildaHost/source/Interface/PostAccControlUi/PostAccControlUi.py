"""
Created on 

@author: simkaufm

Module Description: Gui for a simple control of up to 3 post acceleration devices.
"""

from PyQt5 import QtWidgets
import logging


from Interface.PostAccControlUi.Ui_PostAccControl import Ui_MainWindow_PostAcc
import Application.Config as Cfg


class PostAccControlUi(QtWidgets.QMainWindow, Ui_MainWindow_PostAcc):
    def __init__(self, main_ui):
        super(PostAccControlUi, self).__init__()
        self.setupUi(self)

        self.main_ui = main_ui
        self.scan_main = Cfg._main_instance.scan_main
        self.post_acc_main = Cfg._main_instance.scan_main.post_acc_main

        self.update_power_sups_gui()
        self.pushButton_init_all.clicked.connect(self.init_pow_sups)
        self.pushButton_refresh.clicked.connect(self.update_power_sups_gui)
        self.pushButton_all_on_off.clicked.connect(self.all_off)

        self.show()

    def connect_buttons(self):
        for i in range(1, 4):
            try:
                name = self.resol_power_sup(i)
                getattr(self, 'pushButton_set_volt' + str(i)).clicked.connect(self.volt_set)
                getattr(self, 'pushButton_on_off' + str(i)).clicked.connect(getattr(self, 'set_outp' + str(i)))
            except Exception as e:
                logging.error('while connecting the buttons for ' + name + ' the following exception occurred: ' +
                              ' \n\n' + str(e))

    def init_pow_sups(self):
        self.scan_main.init_post_accel_pwr_supplies()
        self.update_power_sups_gui()
        self.connect_buttons()

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
        self.button_color('pushButton_on_off' + num_str, status_dict.get('output'))

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
                Cfg._main_instance.request_voltage_set(name, voltage)
                self.update_power_sups_gui()

    def set_outp(self, name, outp=None):
        if name is not None:
            status_dict = self.scan_main.get_status_of_pwr_supply(name)
            cur_outp = status_dict.get('output', False)
            if outp is None:
                new_outp = not cur_outp  # if not specified, just toggle the output status
            else:
                new_outp = outp
            self.scan_main.set_post_accel_pwr_spply_output(name, new_outp)
            self.update_power_sups_gui()

    def set_outp1(self):
        self.set_outp(self.resol_power_sup(1))

    def set_outp2(self):
        self.set_outp(self.resol_power_sup(2))

    def set_outp3(self):
        self.set_outp(self.resol_power_sup(3))

    def all_off(self):
        for i in range(1, 3):
            try:
                name = self.resol_power_sup(i)
                self.set_outp(name, outp=False)
            except Exception as e:
                logging.error('While turning all outputs off, the following error occurred: \n\n ' +
                              str(e) + '\n \n')

    def button_color(self, butn, on_off):
        """
        this will change the background color of the output switch for each device.
        Green - On
        Red - Off
        readback of the output is not supported in Heiniznger Powersupplies,
        so this depends on the init of the device and storage of the output variable
        """
        if on_off:
            getattr(self, butn).setStyleSheet("background-color: green")
        else:
            getattr(self, butn).setStyleSheet("background-color: red")

    def closeEvent(self, *args, **kwargs):
        """
        unsubscribe in the corresponding main
        """
        self.main_ui.close_post_acc_win()
