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
    print('initializing widget for type:', dmm_type)
    if dmm_type == 'Ni4071':
        return Ni4071Widg(dmm_name)
    elif dmm_type == 'dummy':
        return Ni4071Widg(dmm_name)
    else:
        print('could not find widget of type: ', dmm_type)


class Ni4071Widg(QtWidgets.QWidget, Ui_form_layout):

    def __init__(self, dmm_name):
        """
        this will create a widget for an already initialized dmm.
            -> initialized as it is an object in scan_main, dmm_dict
        :param dmm_name: str, name of the dmm
        """
        super(Ni4071Widg, self).__init__()
        self.setupUi(self)
        self.dmm_name = dmm_name
        self.raw_config = None
        self.reset_button = QtWidgets.QPushButton('reset values')
        self.communicate_button = QtWidgets.QPushButton('setup device')
        self.reset_button.clicked.connect(self.reset_vals)
        self.communicate_button.clicked.connect(self.communicate_with_dmm)
        self.poll_last_readback()  # therefore device must be initialized before!
        self.request_conf_pars_from_dev(True)  # therefore device must be initialized before!
        print('finished init of: ', self.dmm_name, ' widget')

    def request_conf_pars_from_dev(self, store_and_setup):
        """
        :param store_and_setup: bool, true if you want to setup the widget
         and store the dict to self.raw_config
        """
        raw_config = Cfg._main_instance.request_dmm_config_pars(self.dmm_name)
        if store_and_setup:
            self.setup_ui_from_conf_pars(raw_config)

    def setup_ui_from_conf_pars(self, conf_dict):
        """
        setup the more or less blank widget by adding input fields according to the conf_dict.
        :param conf_dict: dict, tuple (name_str, type_class, certain_value_list, actual_value_bool/int/str/float)
        :return:
        """
        print('rcvd config dict: ', conf_dict)
        self.raw_config = conf_dict
        self.add_widgets_to_form_layout(self.raw_config, self.formLayout_config_values)
        self.formLayout_reading_and_buttons.addRow(self.reset_button, self.communicate_button)

    def add_widgets_to_form_layout(self, inp_dict, parent_layout):
        """
        add input widgets to the parent layout.
        will connect each input widget to self.calling
        :param inp_dict: dict, tuple (name_str, type_class, certain_value_list, actual_value_bool/int/str/float)
        :param parent_layout: Layout
        :return: None, but chnages the items in the inp_dict to a list [label, inp_type, vals, set_val, widget]
        """
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
                    widget.valueChanged.connect(functools.partial(self.widget_value_changed, key))
                    widget.setSingleStep(0.1)
                elif inp_type == str:
                    widget = QtWidgets.QComboBox()
                    widget.addItems(vals)
                    widget.setCurrentText(set_val)
                    widget.currentTextChanged.connect(functools.partial(self.widget_value_changed, key))
                elif inp_type == int:
                    widget = QtWidgets.QSpinBox()
                    widget.setMaximum(max(vals))
                    widget.setMinimum(min(vals))
                    widget.setKeyboardTracking(False)
                    widget.setValue(set_val)
                    widget.valueChanged.connect(functools.partial(self.widget_value_changed, key))
                    widget.setSingleStep(1)
                elif inp_type == bool:
                    widget = QtWidgets.QCheckBox()
                    widget.setChecked(set_val)
                    widget.clicked.connect(functools.partial(self.widget_value_changed, key))
                if widget is not None:
                    parent_layout.addRow(QtWidgets.QLabel(label), widget)
                    inp_dict[key] = [label, inp_type, vals, set_val, widget]
            except Exception as e:
                print(e)

    def coerce_val_tolist_val(self, val, myList):
        """
        coerce the given val to an element from the list.
        """
        return min(myList, key=lambda x: abs(x - val))

    def widget_value_changed(self, key, val):
        """
        when a value of a generic created input is chnaged, this function will be called.
        :param key: str, name of the caller, which should be the key in the self.raw_config dict.
        :param val: bool/str/int/float
        :return:
        """
        label, inp_type, vals, current_val, widget = self.raw_config[key]
        if inp_type is float or inp_type is int:
            new_val = self.coerce_val_tolist_val(val, vals)
            widget.setValue(new_val)
            self.raw_config[key][3] = new_val
        else:
            self.raw_config[key][3] = val  # just set it for strings etc.

    def reset_vals(self):
        """
        resets the gui to the values currently stored in the dev
        """
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
        """ configures and arms teh device with the values currently stored in self.raw_config """
        # config values must only contain key: val
        config = {key: val[3] for key, val in self.raw_config.items()}
        Cfg._main_instance.config_and_arm_dmm(self.dmm_name, config, True)
        # Cfg._main_instance.scan_main.setup_dmm_and_arm(self.dmm_name, config, False)

    def new_voltage(self, val):
        """ will be called if a new voltage is received and will be displayed in the lcd """
        # print('gui read:', val)
        self.lcdNumber.display(round(val, 8))

    def poll_last_readback(self):
        """ this will get the last readback of this dmm as stored in the status of the main. """
        try:
            volt = Cfg._main_instance.dmm_status.get(
                self.dmm_name, {'lastReadback': None}).get('lastReadback', None)[0]
            # is tuple of (volt_float, time_str)
            if volt is not None:
                self.new_voltage(volt)
        except Exception as e:
            pass  # not yet initialized
