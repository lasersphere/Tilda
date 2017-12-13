"""
Created on 

@author: simkaufm

Module Description:
"""

import functools
import logging

from PyQt5 import QtWidgets

import Application.Config as Cfg
from Interface.DmmUi.Ui_Ni4071Widget import Ui_form_layout


def get_wid_by_type(dmm_type, dmm_name):
    """ same widget used by all type, each dmm will emit its parameters """
    logging.info('initializing widget for type: ' + dmm_type)
    return Ni4071Widg(dmm_name, dmm_type)


class Ni4071Widg(QtWidgets.QWidget, Ui_form_layout):

    def __init__(self, dmm_name, type):
        """
        this will create a widget for an already initialized dmm.
            -> initialized as it is an object in scan_main, dmm_dict
        :param dmm_name: str, name of the dmm
        """
        super(Ni4071Widg, self).__init__()
        self.pre_confs = None  # enum of the pre configured settings fot this device.
        self.setupUi(self)
        self.type = type  # maybe this widget can be kept so general, that no widget is neede for each dmm
        self.address = dmm_name.replace(self.type + '_', '')
        self.dmm_name = dmm_name
        self.raw_config = None
        self.reset_button = QtWidgets.QPushButton('reset values')
        self.communicate_button = QtWidgets.QPushButton('setup device')
        self.label_defaul_values = QtWidgets.QLabel('preconfigured settings:')
        self.comboBox_defaul_settings = QtWidgets.QComboBox()
        self.comboBox_defaul_settings.currentTextChanged.connect(self.handle_pre_conf_changed)
        self.reset_button.clicked.connect(self.reset_vals)
        self.communicate_button.clicked.connect(self.communicate_with_dmm)
        self.poll_last_readback()  # therefore device must be initialized before!
        self.request_conf_pars_from_dev(True)  # therefore device must be initialized before!
        logging.info('finished init of: ' + self.dmm_name + ' widget')

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
        logging.debug('rcvd config dict: ' + str(conf_dict))
        logging.info('setting up dmm widget from config dict for dmm ' + self.dmm_name)
        self.raw_config = conf_dict
        self.add_widgets_to_form_layout(self.raw_config, self.formLayout_config_values)
        self.formLayout_reading_and_buttons.addRow(self.reset_button, self.communicate_button)
        self.comboBox_defaul_settings.blockSignals(True)
        self.formLayout_reading_and_buttons.addRow(self.label_defaul_values, self.comboBox_defaul_settings)
        self.comboBox_defaul_settings.blockSignals(False)
        preconfname = self.raw_config['preConfName'][3]
        self.setup_default_val_comboBox(preconfname)

    def add_widgets_to_form_layout(self, inp_dict, parent_layout):
        """
        add input widgets to the parent layout.
        will connect each input widget to self.calling
        :param inp_dict: dict, tuple (name_str, indicator_or_control_bool, type_class,
         certain_value_list, actual_value_bool/int/str/float)
        :param parent_layout: Layout
        :return: None, but chnages the items in the inp_dict to a list [label, inp_type, vals, set_val, widget]
        """
        for key, val in sorted(inp_dict.items()):
            try:
                label = val[0]
                indicator_or_control_bool = val[1]
                inp_type = val[2]
                vals = val[3]
                set_val = val[4]
                widget = None
                if indicator_or_control_bool:  # it shold be a control
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
                else:
                    widget = QtWidgets.QLabel(str(set_val))
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

    def widget_value_changed(self, key, val, enabled=True, update_acc=True):
        """
        when a value of a generic created input is changed, this function will be called.
        :param key: str, name of the caller, which should be the key in the self.raw_config dict.
        :param val: bool/str/int/float
        :return:
        """
        label, inp_type, vals, current_val, widget = self.raw_config.get(key, (None, None, None, None))
        if label is None:
            return None
        if inp_type is float or inp_type is int:
            new_val = self.coerce_val_tolist_val(val, vals)
            widget.setValue(new_val)
            self.raw_config[key][3] = new_val
        else:
            if isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(val)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(val)
            elif isinstance(widget, QtWidgets.QLabel):
                widget.setText(str(val))
            self.raw_config[key][3] = val  # just set it for strings etc.
        widget.setEnabled(enabled)
        if update_acc:
            self.update_accuracy()

    def reset_vals(self):
        """
        resets the gui to the values currently stored in the dev
        """
        raw_config = Cfg._main_instance.scan_main.request_config_pars(self.dmm_name)
        for key, val in raw_config.items():
            try:
                label, ind_ctrl_bool, inp_type, vals, current_val = val
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
        print('will setup dmm to: ', config)
        Cfg._main_instance.config_and_arm_dmm(self.dmm_name, config, False)
        # Cfg._main_instance.scan_main.setup_dmm_and_arm(self.dmm_name, config, False)

    def new_voltage(self, val):
        """ will be called if a new voltage is received and will be displayed in the lcd """
        # print('gui read:', val)
        self.lcdNumber.display(round(val, 8))

    def poll_last_readback(self):
        """ this will get the last readback of this dmm as stored in the status of the main. """
        try:
            volt = Cfg._main_instance.get_active_dmms.get(
                self.dmm_name, ('type', 'addr', 'state', (None, 'time'), {}))[3][0]
            # is tuple of (volt_float, time_str)
            if volt is not None:
                self.new_voltage(volt)
        except Exception as e:
            pass  # not yet initialized

    def poll_accuracy(self):
        """
        get the accuracy for the current configuration
        which will be calculated from the device on the driver layer.
        :return: tuple, tuple of the accuracy
        """
        acc_tuple = Cfg._main_instance.dmm_get_accuracy(self.dmm_name, self.get_current_config())
        return acc_tuple

    def update_accuracy(self):
        """ update the accuracy tuple in the gui """
        if 'accuracy' in self.raw_config.keys():
            enabled = self.raw_config['accuracy'][-1].isEnabled()
            self.widget_value_changed('accuracy', self.poll_accuracy(), enabled, update_acc=False)

    def enable_communication(self, enable_bool):
        """
        this disables/enables all communication to the device.
        :param enable_bool: bool, True for enabling
        """
        self.communicate_button.setEnabled(enable_bool)
        # self.reset_button.setEnabled(enable_bool)

    def load_dict_to_gui(self, conf_dict):
        """
        this tries to sort all values in the dict to the corresponding keys/widgets
        :param conf_dict: dict, for a single dmm key is name of parameter
        """
        enable_widgets = True if conf_dict.get('preConfName', 'initial') == 'manual' else False
        for key, val in conf_dict.items():
            if key not in ['type', 'address']:
                try:
                    if key == 'preConfName':
                        self.comboBox_defaul_settings.blockSignals(True)
                        self.comboBox_defaul_settings.setCurrentText(val)
                        self.comboBox_defaul_settings.blockSignals(False)
                    # let the assignment widget be changeable
                    enable_wid = True if key in ['assignment'] else enable_widgets
                    self.widget_value_changed(key, val, enable_wid)
                except Exception as e:
                    # just print an error for now, maybe be more harsh here in the future.
                    logging.error(
                        'error: could not change value to: %s in key: %s, error is: %s' % (key, val, e), exc_info=True)

    def setup_default_val_comboBox(self, preconfname):
        """
        setup the combobox and disable other widgets if not manual
        """
        self.pre_confs = Cfg._main_instance.request_dmm_available_preconfigs(self.dmm_name)
        names = [each for each in self.pre_confs.__members__]
        names.append('manual')
        self.comboBox_defaul_settings.blockSignals(True)
        self.comboBox_defaul_settings.addItems(names)
        if preconfname != 'manual':
            act_conf = preconfname
            conf = self.pre_confs[act_conf].value
            self.load_dict_to_gui(conf)
        else:
            self.comboBox_defaul_settings.setCurrentText(preconfname)
        self.comboBox_defaul_settings.blockSignals(False)

    def handle_pre_conf_changed(self, text):
        conf = {}
        if text in self.pre_confs.__members__:
            conf = self.pre_confs[text].value
        elif text == 'manual':
            conf = self.get_current_config()
            conf['preConfName'] = 'manual'
        self.load_dict_to_gui(conf)

    def get_current_config(self):
        """
        call this to get a dictionary containing all values in the gui
        :return: dict, keys are parameter names.
        """
        ret = {key: val[3] for key, val in self.raw_config.items()}
        ret['type'] = self.type
        ret['address'] = self.address
        return ret
