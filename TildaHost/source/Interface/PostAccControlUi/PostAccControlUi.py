"""
Created on 

@author: simkaufm

Module Description: Gui for a simple control of up to 3 post acceleration devices.
"""

from PyQt5 import QtWidgets, QtCore
import logging
import time


from Interface.PostAccControlUi.Ui_PostAccControl import Ui_MainWindow_PostAcc
import Application.Config as Cfg


class PostAccControlUi(QtWidgets.QMainWindow, Ui_MainWindow_PostAcc):

    post_acc_signal = QtCore.pyqtSignal([dict])

    def __init__(self, main_ui):
        super(PostAccControlUi, self).__init__()
        self.setupUi(self)

        self.main_ui = main_ui
        self.scan_main = None
        self.post_acc_main = None

        # self.update_power_sups_gui()
        self.pushButton_init_all.clicked.connect(self.init_pow_sups)
        self.pushButton_refresh.clicked.connect(self.update_power_sups_gui)
        self.pushButton_all_on_off.clicked.connect(self.all_off)

        self.post_acc_signal.connect(self.rcvd_new_status_dict)

        self.show()

    def post_acc_cyclic(self):
        pass

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
        Cfg._main_instance.init_power_sups()

    def rcvd_new_status_dict(self, status_dict_list):
        print(status_dict_list)

    def update_power_sups_gui(self):
        """
        update the gui by reading
        """
        act_dict = Cfg._main_instance.active_power_supplies
        for name, instance in act_dict.items():
            self.update_single_power_sup(status_dict, status_dict.get('name')[-1:])

    def get_status_of_power_sup(self, name):
        """
        request a status dict from the main.
        """
        Cfg._main_instance.power_supply_status(name)

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
                Cfg._main_instance.set_power_supply_voltage(name, voltage)
                self.update_power_sups_gui()

    def set_outp(self, name, outp=None):
        if name is not None:
            status_dict = self.get_status_of_power_sup(name)
            cur_outp = status_dict.get('output', False)
            if outp is None:
                new_outp = not cur_outp  # if not specified, just toggle the output status
            else:
                new_outp = outp
            Cfg._main_instance.set_power_sup_outp(name, new_outp)
            self.update_single_power_sup(self.get_status_of_power_sup(name), name[-1:])

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
