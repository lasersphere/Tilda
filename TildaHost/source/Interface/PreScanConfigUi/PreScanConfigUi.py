"""

Created on '15.05.2017'

@author:'simkaufm'

description: Main gui for configuring the pre / during / post scan settings

"""

from PyQt5 import QtWidgets, QtCore
import logging
import functools
from copy import deepcopy

import Application.Config as Cfg
from Interface.PreScanConfigUi.Ui_PreScanMain import Ui_PreScanMainWin
from Interface.DmmUi.ChooseDmmWidget import ChooseDmmWidget
from Interface.DmmUi.DMMWidgets import Ni4071Widg


class PreScanConfigUi(QtWidgets.QMainWindow, Ui_PreScanMainWin):
    # callback for the widget when choosing a new dmm
    # returns tuple of (dev_type, dev_adress) out of ChooseDmmWidget
    init_dmm_clicked_callback = QtCore.pyqtSignal(tuple)
    # callback to learn when the main is done with the init/deinit of the device.
    init_dmm_done_callback = QtCore.pyqtSignal(bool)
    deinit_dmm_done_callback = QtCore.pyqtSignal(bool)
    # callback for the voltage readings, done by the main when in idle state
    voltage_reading = QtCore.pyqtSignal(dict)

    def __init__(self, parent, active_iso=''):
        super(PreScanConfigUi, self).__init__()
        self.setupUi(self)
        self.show()

        self.buttonBox.accepted.connect(self.confirm)
        self.buttonBox.rejected.connect(self.close)

        self.comboBox.addItems(['preScan', 'duringScan', 'postScan'])
        self.pre_or_during_scan_str = self.comboBox.currentText()
        self.comboBox.currentTextChanged.connect(self.pre_post_during_changed)
        self.checkBox_triton_measure.stateChanged.connect(self.triton_checkbox_changed)
        self.checkBox_voltage_measure.stateChanged.connect(self.voltage_checkbox_changed)

        self.parent_ui = parent
        self.active_iso = active_iso
        self.setWindowTitle('pre / during / post scan settings of %s' % self.active_iso)

        # digital multimeter related
        self.current_meas_volt_settings = {}  # storge for current settings
        self.comm_enabled = False  # from this window the dmms should not be controlled
        try:  # for starting without a main
            self.dmm_types = Cfg._main_instance.scan_main.digital_multi_meter.types
        except AttributeError:
            self.dmm_types = ['None']
        self.tabs = {
            'tab0': [self.tab_0, None, None]}  # dict for storing all tabs, key: [QWidget(), Layout, userWidget]
        self.tabWidget.setTabText(0, 'choose dmm')
        self.tabs['tab0'][1] = QtWidgets.QVBoxLayout(self.tabs['tab0'][0])
        self.init_dmm_clicked_callback.connect(self.initialize_dmm)
        self.check_for_already_active_dmms()

        self.choose_dmm_wid = ChooseDmmWidget(self.init_dmm_clicked_callback, self.dmm_types)
        self.tabs['tab0'][2] = self.tabs['tab0'][1].addWidget(self.choose_dmm_wid)
        try:
            Cfg._main_instance.dmm_gui_subscribe(self.voltage_reading)
        except AttributeError:  # no main available
            pass
        self.voltage_reading.connect(self.rcvd_voltage_dict)

        self.tabWidget.setTabsClosable(True)
        self.tabWidget.tabCloseRequested.connect(self.tab_wants_to_be_closed)

        self.setup_volt_meas_from_main()

    def confirm(self):
        """
        when ok is pressed, values are stored in the main, if an isotope has ben selected before.
        """
        if self.active_iso is not None:
            # check stuff from storage
            for pre_scan_key, meas_volt_dict in self.current_meas_volt_settings.items():
                Cfg._main_instance.scan_pars[self.active_iso]['measureVoltPars'][pre_scan_key] = deepcopy(meas_volt_dict)
            # overwrite with actual
            Cfg._main_instance.scan_pars[self.active_iso]['measureVoltPars'][
                self.pre_or_during_scan_str] = self.get_current_meas_volt_pars()
            print('set values to: ', Cfg._main_instance.scan_pars[self.active_iso]['measureVoltPars'])
            # TODO remove this:
            import json
            print(json.dumps(Cfg._main_instance.scan_pars[self.active_iso]['measureVoltPars'], sort_keys=True, indent=4))
        self.close()

    def closeEvent(self, event):
        """
        when closing this window, unsubscirbe from main and from voltage readback.
        Also tell parent gui that it is closed.
        """
        if self.parent_ui is not None:
            Cfg._main_instance.dmm_gui_unsubscribe()
            self.voltage_reading.disconnect()
            # TODO this should be removed
            self.parent_ui.close_pre_post_scan_win()

    def pre_post_during_changed(self, pre_post_during_str):
        """ whenever this is changed, load stuff from main. """
        # TODO when changing one, still needs to keep the settings for the other modi
        # TODO dmm: pre == post settings but also possible not to measure one of those.
        # -> pre != post, if user does not want a post measurement for example.

        # store values from current setting first!
        self.current_meas_volt_settings[self.pre_or_during_scan_str] = self.get_current_meas_volt_pars()
        self.pre_or_during_scan_str = pre_post_during_str
        existing_config = None if self.current_meas_volt_settings == {} else self.current_meas_volt_settings
        self.setup_volt_meas_from_main(existing_config)
        # TODO remove this:
        print('existing config:')
        import json
        print(json.dumps(existing_config, sort_keys=True, indent=4))

    def triton_checkbox_changed(self, state):
        """
        disable the main widget and leaves empty dict at:
             {'triton': {self.pre_or_during_scan_str: {} ... }
        """
        self.treeView_triton.setEnabled(state == 2)

    def voltage_checkbox_changed(self, state):
        """
        disable the main widget and leaves empty dict at:
             {'measureVoltPars': {self.pre_or_during_scan_str: {'dmms': {}, ...} ... }
        """
        self.tabWidget.setEnabled(state == 2)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setEnabled(state == 2)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setEnabled(state == 2)

    ''' voltage related, mostly copied from DmmUi.py '''

    def setup_volt_meas_from_main(self, meas_volt_pars_dict=None):
        """
        setup gui with the scan parameters stored in the main or from the meas_volt_pars_dict
        :param meas_volt_pars_dict: dict, scan_dict['measureVoltPars'] = {'preScan': {...}}
        :return:
        """
        if self.active_iso is not None and self.parent_ui is not None:
            if meas_volt_pars_dict is None:  # try to get it from main
                scan_dict = Cfg._main_instance.scan_pars[self.active_iso]
                meas_volt_pars_dict = scan_dict['measureVoltPars']
            meas_volt_dict = meas_volt_pars_dict.get(self.pre_or_during_scan_str, None)
            if meas_volt_dict is None:
                self.set_pulse_len_and_timeout({})
                self.checkBox_voltage_measure.setChecked(False)
                self.voltage_checkbox_changed(0)
                for key, val in self.tabs.items():
                    if key != 'tab0':
                        if self.pre_or_during_scan_str == 'preScan':
                            print('will set %s to preScan' % val[-1])
                            val[-1].handle_pre_conf_changed('pre_scan')
            else:
                #  copy values for measVoltPulseLength25ns and measVoltTimeout10ns from preScan
                meas_volt_dict['measVoltTimeout10ns'] = deepcopy(
                    meas_volt_pars_dict.get('preScan', {}).get('measVoltTimeout10ns', 0))
                meas_volt_dict['measVoltPulseLength25ns'] = deepcopy(
                    meas_volt_pars_dict.get('preScan', {}).get('measVoltPulseLength25ns', 0))
                meas_volt_dict['switchBoxSettleTimeS'] = deepcopy(
                    meas_volt_pars_dict.get('preScan', {}).get('switchBoxSettleTimeS', 5.0))
                self.load_config_dict(meas_volt_dict)
        else:
            self.checkBox_voltage_measure.setChecked(False)
            self.voltage_checkbox_changed(0)
            self.checkBox_triton_measure.setChecked(False)
            self.triton_checkbox_changed(0)

    def tab_wants_to_be_closed(self, *args):
        """
        when clicking on(X) on the tab this will be called and the dmm will be deinitialized.
        will be called by self.tabWidget.tabCloseRequested
        :param args: tuple, (ind,)
        """
        tab_ind = args[0]
        if tab_ind:  # not for tab 0 which is teh selector tab
            dmm_name = self.tabWidget.tabText(tab_ind)
            self.tabs.pop(dmm_name)
            self.tabWidget.removeTab(tab_ind)
            if self.comm_enabled:  # deinit only if comm is allowed.
                self.deinit_dmm(dmm_name)

    def initialize_dmm(self, dmm_tuple, startup_config=None):
        """
        will initialize the dmm of type and adress and store the instance in the scan_main.
        :param dmm_tuple: tuple, (dev_type_str, dev_addr_str)
        """
        print('starting to initialize: ', dmm_tuple)
        dev_type, dev_address = dmm_tuple
        dmm_name = dev_type + '_' + dev_address
        if dmm_name in list(self.tabs.keys()) or dmm_name is None:
            print('could not initialize: ', dmm_name, ' ... already initialized?')
            return None  # break when not initialized
        self.init_dmm_done_callback.connect(functools.partial(self.setup_new_tab_widget, (dmm_name, dev_type)))
        Cfg._main_instance.init_dmm(dev_type, dev_address, self.init_dmm_done_callback, startup_config)

    def deinit_dmm(self, dmm_name):
        """ deinitializes the dmm """
        Cfg._main_instance.deinit_dmm(dmm_name)

    def check_for_already_active_dmms(self):
        """ checks for already active dmms and opens tabs for them """
        try:
            act_dmm_dict = Cfg._main_instance.get_active_dmms()
            for key, val in act_dmm_dict.items():
                dmm_type, dmm_addr, state_str, last_readback, dmm_config = val
                self.setup_new_tab_widget((key, dmm_type), False)
        except AttributeError:  # no main instance available
            pass

    def setup_new_tab_widget(self, tpl, disconnect_signal=True):
        """
        setup a new tab inside the tab widget.
        with the get_wid_by_type function the right widget is initiated.
        """
        print('setting up tab: ', tpl)
        dmm_name, dev_type = tpl  # dmm_name = tab_name
        # print('done initializing: ', dmm_name, dev_type)
        if disconnect_signal:
            self.init_dmm_done_callback.disconnect()
        self.tabs[dmm_name] = [None, None, None]
        self.tabs[dmm_name][0] = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tabs[dmm_name][0], dmm_name)
        self.tabs[dmm_name][1] = QtWidgets.QVBoxLayout(self.tabs[dmm_name][0])
        self.tabWidget.setCurrentWidget(self.tabs[dmm_name][0])
        self.tabs[dmm_name][2] = Ni4071Widg(dmm_name, dev_type)
        self.tabs[dmm_name][2].enable_communication(self.comm_enabled)
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

    def set_pulse_len_and_timeout(self, meas_volt_dict):
        """
        sets the double spinboxes: doubleSpinBox_measVoltPulseLength_mu_s and doubleSpinBox_measVoltTimeout_mu_s_set
        to the value given in the meas_volt_dict or to the default value
        :param meas_volt_dict: dict, should contain 'measVoltPulseLength25ns' and 'measVoltTimeout10ns' keys
        :return: None
        """
        pulse_len_25ns = meas_volt_dict.get('measVoltPulseLength25ns', 0)
        if pulse_len_25ns is not None:
            if pulse_len_25ns == 0:
                pulse_len_25ns = 400  # set by default to 10µs
            pulse_len_mu_s = pulse_len_25ns * 25 / 1000
            self.doubleSpinBox_measVoltPulseLength_mu_s.setValue(pulse_len_mu_s)
        timeout_10_ns = meas_volt_dict.get('measVoltTimeout10ns', 0)
        if timeout_10_ns is not None:
            timeout_volt_meas_mu_s = timeout_10_ns / 100000
            if timeout_10_ns == 0:
                timeout_volt_meas_mu_s = 10000  # set to 10 s by default.
            self.doubleSpinBox_measVoltTimeout_mu_s_set.setValue(timeout_volt_meas_mu_s)
        settling_time_swb = meas_volt_dict.get('switchBoxSettleTimeS', 5.0)
        if settling_time_swb is None:
            settling_time_swb = 5.0
        self.doubleSpinBox_wait_after_switchbox.setValue(settling_time_swb)

    def load_config_dict(self, meas_volt_dict):
        """
        use this if you want to paste your config settings inside the dmm_conf_dict to the
        corresponding interfaces.
        Note, that this will not yet have any impact on the device itself.
        only by communicating with it than, it will be configured
        :param dmm_conf_dict: dict, keys are dmm_names, values are the config dicts for each dmm
        """
        print('loading config dict: ', meas_volt_dict)
        dmm_conf_dict = meas_volt_dict.get('dmms', {})
        self.checkBox_voltage_measure.setChecked(dmm_conf_dict != {})
        self.voltage_checkbox_changed(2 if dmm_conf_dict != {} else 0)
        self.set_pulse_len_and_timeout(meas_volt_dict)
        for key, val in dmm_conf_dict.items():
            try:
                self.tabs[key][-1].load_dict_to_gui(val)
            except Exception as e:
                logging.error(
                    'error: while loading the dmm vals to the gui of %s, this happened: %s' % (key, e))
                warning_ms = key + ' was not yet initialized\n \n Do you want to initialize it now?'
                try:
                    button_box_answer = QtWidgets.QMessageBox.question(
                        self, 'DMM not initialized', warning_ms,
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
                    if button_box_answer == QtWidgets.QMessageBox.Yes:
                        self.initialize_dmm((dmm_conf_dict[key].get('type', ''),
                                             dmm_conf_dict[key].get('address', '')),
                                            val.get('preConfName', 'initial'))
                except Exception as e:
                    print(e)

    def get_current_dmm_config(self):
        """
        this will return the current gui values of all active dmms
        :return: dict, key is dmm_name, val is dict, with config of all dmms
        """
        ret = {}
        for key, val in self.tabs.items():
            if key != 'tab0':
                try:
                    ret[key] = self.tabs[key][-1].get_current_config()
                except Exception as e:
                    logging.error(
                        'error: while reading the gui of %s, this happened: %s' % (key, e))
        return ret

    def get_current_meas_volt_pars(self):
        """
        this will return a complete measVoltPars dict, with information off all dmms and the pulselength and timeout.
        :return: dict, measVoltPars as stated as in draftScanparameters.py in the Service layer.
        """
        meas_volt_dict = {}
        # set 'dmms' to empty dict if this should not be measured.
        meas_volt_dict['dmms'] = self.get_current_dmm_config() if self.checkBox_voltage_measure.isChecked() else {}
        meas_volt_dict['measVoltPulseLength25ns'] = int(self.doubleSpinBox_measVoltPulseLength_mu_s.value() * 1000 / 25)
        meas_volt_dict['measVoltTimeout10ns'] = int(self.doubleSpinBox_measVoltTimeout_mu_s_set.value() * 1000000 / 10)
        meas_volt_dict['switchBoxSettleTimeS'] = self.doubleSpinBox_wait_after_switchbox.value()
        return meas_volt_dict


if __name__=='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    gui = PreScanConfigUi(None, '')
    app.exec_()
