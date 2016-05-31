"""
Created on 

@author: simkaufm

Module Description:
"""

from PyQt5 import QtWidgets

from Interface.DmmUi.Ui_Ni4071Widget import Ui_form_layout
import Application.Config as Cfg


def get_wid_by_type(dmm_type, dmm_name):
    print('type of dmm:', dmm_type)
    if dmm_type == 'Ni4071':
        return Ni4071Widg(dmm_name)
    elif dmm_type == 'dummy':
        return Ni4071Widg(dmm_name)


class Ni4071Widg(QtWidgets.QWidget, Ui_form_layout):
    def __init__(self, dmm_name):
        super(Ni4071Widg, self).__init__()
        self.setupUi(self)
        self.dmm_name = dmm_name
        self.raw_config = Cfg._main_instance.scan_main.request_config_pars(self.dmm_name)
        self.formLayout.addWidget(QtWidgets.QLabel())
