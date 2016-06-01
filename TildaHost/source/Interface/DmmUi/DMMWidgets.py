"""
Created on 

@author: simkaufm

Module Description:
"""

from PyQt5 import QtWidgets, QtCore
from copy import deepcopy
import functools

from Interface.DmmUi.Ui_Ni4071Widget import Ui_form_layout
import Application.Config as Cfg


def get_wid_by_type(dmm_type, dmm_name):
    print('type of dmm:', dmm_type)
    if dmm_type == 'Ni4071':
        return Ni4071Widg(dmm_name)
    elif dmm_type == 'dummy':
        return Ni4071Widg(dmm_name)


class Ni4071Widg(QtWidgets.QWidget, Ui_form_layout):
    callback_from_Ni4071_widget = QtCore.pyqtSignal(dict)

    def __init__(self, dmm_name):
        super(Ni4071Widg, self).__init__()
        self.setupUi(self)
        self.dmm_name = dmm_name
        self.raw_config = None
        self.reset_button = QtWidgets.QPushButton('reset values')
        self.communicate_button = QtWidgets.QPushButton('setup device')
        self.reset_button.clicked.connect(self.reset_vals)
        self.communicate_button.clicked.connect(self.communicate_with_dmm)

        self.callback_from_Ni4071_widget.connect(self.setup_ui_from_conf_pars)
        self.request_conf_pars_from_dev()

    def request_conf_pars_from_dev(self):
        Cfg._main_instance.request_dmm_config_pars(self.dmm_name, self.callback_from_Ni4071_widget)

    def setup_ui_from_conf_pars(self, conf_dict):
        self.raw_config = conf_dict
        self.add_widgets_to_form_layout(self.raw_config, self.formLayout_config_values)
        self.formLayout_reading_and_buttons.addRow(self.reset_button, self.communicate_button)

    def add_widgets_to_form_layout(self, inp_dict, parent_layout):
        for key, val in sorted(inp_dict.items()):
            try:
                label = val[0]
                inp_type = val[1]
                vals = val[2]
                set_val = val[3]
                widget = None
                if inp_type == float:
                    widget = QtWidgets.QDoubleSpinBox()
                    widget.setMaximum(max(vals))
                    widget.setMinimum(min(vals))
                    widget.setKeyboardTracking(False)
                    widget.setValue(set_val)
                    widget.valueChanged.connect(functools.partial(self.calling, key))
                    widget.setSingleStep(0.1)
                elif inp_type == str:
                    widget = QtWidgets.QComboBox()
                    widget.addItems(vals)
                    widget.setCurrentText(set_val)
                elif inp_type == int:
                    widget = QtWidgets.QSpinBox()
                    widget.setMaximum(max(vals))
                    widget.setMinimum(min(vals))
                    widget.setKeyboardTracking(False)
                    widget.setValue(set_val)
                    widget.valueChanged.connect(functools.partial(self.calling, key))
                    widget.setSingleStep(1)
                elif inp_type == bool:
                    widget = QtWidgets.QCheckBox()
                    widget.setChecked(set_val)
                if widget is not None:
                    parent_layout.addRow(QtWidgets.QLabel(label), widget)
                    inp_dict[key] = [label, inp_type, vals, set_val, widget]
            except Exception as e:
                print(e)

    def coerce_val_tolist_val(self, val, myList):
        return min(myList, key=lambda x: abs(x - val))

    def calling(self, key, val):
        label, inp_type, vals, current_val, widget = self.raw_config[key]
        if inp_type is float or inp_type is int:
            new_val = self.coerce_val_tolist_val(val, vals)
            widget.setValue(new_val)
            self.raw_config[key][3] = new_val
        else:
            self.raw_config[key][3] = val

    def reset_vals(self):
        raw_config = Cfg._main_instance.scan_main.request_config_pars(self.dmm_name)
        for key, val in raw_config.items():
            try:
                label, inp_type, vals, current_val = val
                widget = self.raw_config[key][4]
                self.raw_config[key] = [label, inp_type, vals, current_val, widget]
                if inp_type is float or inp_type is int:
                    widget.setValue(current_val)
                elif inp_type is str:
                    widget.setCurrentText(current_val)
                elif inp_type is bool:
                    widget.setChecked(current_val)
            except Exception as e:
                print(e)

    def communicate_with_dmm(self):
        # should be via state in main
        # config values must only contain key: val
        config = {key: val[3] for key, val in self.raw_config.items()}
        Cfg._main_instance.config_and_arm_dmm(self.dmm_name, config, True)
        # Cfg._main_instance.scan_main.setup_dmm_and_arm(self.dmm_name, config, False)
