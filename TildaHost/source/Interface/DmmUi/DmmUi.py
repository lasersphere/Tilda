"""
Created on 

@author: simkaufm

Module Description: Main Ui for controlling the digital Multimeters connected to the PXI-Crate
"""

import functools
import logging
from copy import deepcopy

from PyQt5 import QtWidgets, QtCore

import Application.Config as Cfg
from Interface.DmmUi.ChooseDmmWidget import ChooseDmmWidget
from Interface.DmmUi.DMMWidgets import get_wid_by_type
from Interface.DmmUi.Ui_DmmLiveView import Ui_MainWindow


class DmmLiveViewUi(QtWidgets.QMainWindow, Ui_MainWindow):
    # callback for the widget when choosing a new dmm
    # returns tuple of (dev_type, dev_adress) out of ChooseDmmWidget
    init_dmm_clicked_callback = QtCore.pyqtSignal(tuple)
    # callback to learn when the main is done with the init/deinit of the device.
    init_dmm_done_callback = QtCore.pyqtSignal(bool)
    deinit_dmm_done_callback = QtCore.pyqtSignal(bool)
    # callback for the voltage readings, done by the main when in idle state
    voltage_reading = QtCore.pyqtSignal(dict)

    def __init__(self, parent, window_name='DMM Live View Window',
                 enable_com=None, active_iso=None, pre_or_during_scan_str='', selected_track_name=None):
        """
        this will statup the GUI and check for already active dmm's
        :param parent: parent_gui, usually the main in order to unsubscribe from it etc.
        """
        super(DmmLiveViewUi, self).__init__()
        self.setupUi(self)
        self.setWindowTitle(window_name)
        self.parent_ui = parent
        try:
            self.dmm_types = Cfg._main_instance.scan_main.digital_multi_meter.types
        except AttributeError:
            self.dmm_types = ['None']

        self.comm_enabled = True

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

        self.pushButton_confirm.clicked.connect(self.confirm_settings)

        self.tabWidget.setTabsClosable(True)

        self.tabWidget.tabCloseRequested.connect(self.tab_wants_to_be_closed)

        self.show()

        self.comm_allow_overwrite_val = False
        if enable_com is not None:
            self.enable_communication(enable_com, True)

        self.active_iso = active_iso
        self.selected_track_name = selected_track_name
        self.pre_or_during_scan_str = pre_or_during_scan_str

        if self.active_iso is not None:
            scan_dict = Cfg._main_instance.scan_pars[self.active_iso]
            meas_volt_pars_dict = scan_dict[self.selected_track_name]['measureVoltPars']
            meas_volt_dict = meas_volt_pars_dict.get(self.pre_or_during_scan_str, None)
            if meas_volt_dict is None:
                self.set_pulse_len_and_timeout({})
                for key, val in self.tabs.items():
                    if key != 'tab0':
                        if self.pre_or_during_scan_str == 'preScan':
                            logging.info('will set %s to preScan' % val[-1])
                            val[-1].handle_pre_conf_changed('pre_scan')
                        else:
                            if scan_dict.get('isotopeData', {}).get('type', 'kepco') == 'kepco':
                                val[-1].handle_pre_conf_changed('kepco')
                            else:
                                val[-1].handle_pre_conf_changed('periodic')
                # maybe automatically get pres_scan config etc. here
                # self.load_config_dict()
            else:
                #  copy values for measVoltPulseLength25ns and measVoltTimeout10ns from preScan
                meas_volt_dict['measVoltTimeout10ns'] = deepcopy(meas_volt_pars_dict.get(
                    'preScan', {}).get('measVoltTimeout10ns', 0))
                meas_volt_dict['measVoltPulseLength25ns'] = deepcopy(meas_volt_pars_dict.get(
                    'preScan', {}).get('measVoltPulseLength25ns', 0))
                meas_volt_dict['switchBoxSettleTimeS'] = deepcopy(meas_volt_pars_dict.get(
                    'preScan', {}).get('switchBoxSettleTimeS', 5.0))
                self.load_config_dict(meas_volt_dict)
        else:  # delete unnecessary spinboxes when not configuring for track
            self.doubleSpinBox_measVoltTimeout_mu_s_set.setParent(None)
            self.doubleSpinBox_measVoltPulseLength_mu_s.setParent(None)
            self.doubleSpinBox_wait_after_switchbox.setParent(None)
            self.label_measVoltPulseLength_mu_s.setParent(None)
            self.label_measVoltTimeout_mu_s.setParent(None)
            self.label_wait_after_switchbox.setParent(None)
            self.verticalLayout.removeItem(self.formLayout_pulse_len_and_timeout)

    def enable_communication(self, comm_allow, overwrite=False):
        """
        allow or disable communication with a device.
        will be disabled from mainui, when status is not idle
        :param overwrite: bool, overwrite any following call with the comm_allow value now.
        :param comm_allow: bool, boolean if communication is allowed or not.
        """
        if overwrite:
            logging.debug('communication has ben permanently %s in %s'
                          % ('enabled' if comm_allow else 'disabled', str(self.windowTitle())))
            self.comm_allow_overwrite_val = True
            self._enable_communication(comm_allow)
        else:
            if self.comm_allow_overwrite_val:
                logging.debug('communication is blocked in %s' % str(self.windowTitle()))
                return None
            else:
                self._enable_communication(comm_allow)

    def _enable_communication(self, enable_bool):
        self.comm_enabled = enable_bool
        # self.choose_dmm_wid.pushButton_initialize.setEnabled(enable_bool)
        # self.tabWidget.setTabsClosable(enable_bool)
        for dmm_name, val_lists in self.tabs.items():
            if dmm_name != 'tab0':
                widget = val_lists[-1]
                widget.enable_communication(enable_bool)

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
        logging.info('starting to initialize dmm: %s' % str(dmm_tuple))
        dev_type, dev_address = dmm_tuple
        dmm_name = dev_type + '_' + dev_address
        if dmm_name in list(self.tabs.keys()) or dmm_name is None:
            logging.warning('could not initialize dmm: %s ... already initialized?' % dmm_name)
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
            logging.error('while checking for already active dmms an exception occured', exc_info=True)
            pass

    def setup_new_tab_widget(self, tpl, disconnect_signal=True):
        """
        setup a new tab inside the tab widget.
        with the get_wid_by_type function the right widget is initiated.
        """
        dmm_name, dev_type = tpl  # dmm_name = tab_name
        # print('done initializing: ', dmm_name, dev_type)
        if disconnect_signal:
            self.init_dmm_done_callback.disconnect()
        self.tabs[dmm_name] = [None, None, None]
        self.tabs[dmm_name][0] = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tabs[dmm_name][0], dmm_name)
        self.tabs[dmm_name][1] = QtWidgets.QVBoxLayout(self.tabs[dmm_name][0])
        self.tabWidget.setCurrentWidget(self.tabs[dmm_name][0])
        self.tabs[dmm_name][2] = get_wid_by_type(dev_type, dmm_name)
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

    def closeEvent(self, *args, **kwargs):
        """
        when closing this window, unsubscirbe from main and from voltage readback.
        Also tell parent gui that it is closed.
        """
        Cfg._main_instance.dmm_gui_unsubscribe()
        self.voltage_reading.disconnect()
        self.parent_ui.close_dmm_live_view_win()

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
        logging.info('loading voltage measurement config dict: %s' % str(meas_volt_dict))
        dmm_conf_dict = meas_volt_dict.get('dmms', {})
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
                    logging.error('error in %s while loading the dmm: %s with the config: %s\n'
                                  ' error is: %s ' % (self.name, key, str(val), str(e)), exc_info=True)

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
                        'error: while reading the gui of %s, this happened: %s' % (key, e), exc_info=True)
        return ret

    def get_current_meas_volt_pars(self):
        """
        this will return a complete measVoltPars dict, with information off all dmms and the pulselength and timeout.
        :return: dict, measVoltPars as stated as in draftScanparameters.py in the Service layer.
        """
        meas_volt_dict = {}
        meas_volt_dict['dmms'] = self.get_current_dmm_config()
        meas_volt_dict['measVoltPulseLength25ns'] = int(self.doubleSpinBox_measVoltPulseLength_mu_s.value() * 1000 / 25)
        meas_volt_dict['measVoltTimeout10ns'] = int(self.doubleSpinBox_measVoltTimeout_mu_s_set.value() * 1000000 / 10)
        meas_volt_dict['switchBoxSettleTimeS'] = self.doubleSpinBox_wait_after_switchbox.value()
        return meas_volt_dict

    def confirm_settings(self):
        """
        when ok is pressed, values are stored in the main, if an isotope has ben selected before.
        """
        if self.active_iso is not None:
            Cfg._main_instance.scan_pars[
                self.active_iso][self.selected_track_name][
                'measureVoltPars'][self.pre_or_during_scan_str] = self.get_current_meas_volt_pars()
            logging.debug('dmmUi has set values to: ' + str(
                Cfg._main_instance.scan_pars[self.active_iso][self.selected_track_name]['measureVoltPars']))
            self.parent_ui.pre_or_during_scan_index += 1
        self.close()





# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     ui = DmmLiveViewUi()
#     ui.show()
#     app.exec()
