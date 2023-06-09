"""

Created on '15.05.2017'

@author:'simkaufm'

description: Main gui for configuring the pre / during / post scan settings

"""

from PyQt5 import QtWidgets, QtCore, Qt
import logging
import functools
from copy import deepcopy
from datetime import timedelta

import Tilda.Application.Config as Cfg
from Tilda.Interface.PreScanConfigUi.Ui_PreScanMain import Ui_PreScanMainWin
from Tilda.Interface.DmmUi.ChooseDmmWidget import ChooseDmmWidget
from Tilda.Interface.DmmUi.DMMWidgets import Ni4071Widg


class PreScanConfigUi(QtWidgets.QMainWindow, Ui_PreScanMainWin):
    # callback for the widget when choosing a new dmm
    # returns tuple of (dev_type, dev_adress) out of ChooseDmmWidget
    init_dmm_clicked_callback = QtCore.pyqtSignal(tuple)
    # callback to learn when the main is done with the init/deinit of the device.
    init_dmm_done_callback = QtCore.pyqtSignal(bool)
    deinit_dmm_done_callback = QtCore.pyqtSignal(bool)
    # callback for the voltage readings, done by the main when in idle state
    voltage_reading = QtCore.pyqtSignal(dict)

    def __init__(self, parent, active_iso='', active_track_name=''):
        super(PreScanConfigUi, self).__init__()
        self.setupUi(self)
        self.show()

        self.buttonBox.accepted.connect(self.confirm)
        self.buttonBox.rejected.connect(self.close)

        self.comboBox.addItems(['preScan', 'duringScan', 'postScan'])
        self.pre_or_during_scan_str = self.comboBox.currentText()
        self.comboBox.currentTextChanged.connect(self.pre_post_during_changed)
        self.checkBox_voltage_measure.stateChanged.connect(self.voltage_checkbox_changed)

        self.parent_ui = parent
        self.active_iso = active_iso
        self.act_track_name = active_track_name
        self.setWindowTitle('pre / during / post scan settings of %s %s ' % (self.active_iso, self.act_track_name))

        # Triton related:
        self.cur_dev = None  # str, currently selected device name
        self.triton_scan_dict = self.get_triton_scan_pars()
        self.triton_scan_dict_backup = deepcopy(self.triton_scan_dict)  # to keep any data stored in the channels
        self.active_devices = None  # dict with active device will be updated on each self.setup_triton_devs () call
        self.setup_triton_devs()
        # logging.info('incoming triton dict: ')
        # import json
        # logging.debug(
        #     json.dumps(self.triton_scan_dict, sort_keys=True, indent=4))

        # self.listWidget_devices.currentItemChanged.connect(self.dev_selection_changed)
        self.listWidget_devices.itemClicked.connect(self.dev_selection_changed)
        self.tableWidget_channels.itemClicked.connect(self.check_any_ch_active)
        self.checkBox_triton_measure.stateChanged.connect(self.triton_measure_checkbox_changed)

        # SQL related
        self.observables_measure = []
        self.d_interval.setValue(Cfg._main_instance.sql_interval)
        self.sql_scan_dict = self.get_sql_scan_pars()
        self.sql_scan_dict_backup = deepcopy(self.sql_scan_dict)  # to keep any data stored in the channels
        self.active_channels = None
        self.setup_sql_channels()

        # self.b_add_measure.clicked.connect(self.add_observable)
        self.check_sql_measure.stateChanged.connect(self.sql_measure_checkbox_changed)
        self.table_sql.itemChanged[QtWidgets.QTableWidgetItem].connect(self.sql_process_row)

        # digital multimeter related
        self.current_meas_volt_settings = {}  # storage for current settings, this holds pre/post/during dicts
        self.comm_enabled = False  # from this window the dmms should not be controlled
        try:  # for starting without a main
            self.dmm_types = Cfg._main_instance.scan_main.digital_multi_meter.types
            self.timeout = Cfg._main_instance.pre_scan_measurement_timeout_s.seconds
        except AttributeError:
            self.dmm_types = ['None']
            self.timeout = timedelta(seconds=60).seconds
        self.doubleSpinBox_timeout_pre_scan_s.setValue(self.timeout)
        self.tabs = {
            'tab0': [self.tab_0, None, None]}  # dict for storing all tabs, key: [QWidget(), Layout, userWidget]
        self.tabWidget.setTabText(0, 'choose dmm')
        self.tabs['tab0'][1] = QtWidgets.QVBoxLayout(self.tabs['tab0'][0])
        self.init_dmm_clicked_callback.connect(self.initialize_dmm)
        # self.check_for_already_active_dmms()

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
        when ok is pressed, values are stored in the parent track ui.
        They are written to the main as soon as confirm is clicked in track gui.
        """
        try:
            self.pre_post_during_changed(self.comboBox.currentText(), closing_widget=True)
            logging.debug('confirmed meas volt settings')
            logging.debug('act iso: %s, main inst: %s ' % (str(self.active_iso), str(Cfg._main_instance)))
            if self.active_iso is not None and Cfg._main_instance is not None:
                logging.debug('accessing main now')
                # check stuff from storage
                logging.debug('current_meas_volt_settings:' + str(self.current_meas_volt_settings))
                for pre_scan_key, meas_volt_dict in self.current_meas_volt_settings.items():
                    self.parent_ui.buffer_pars['measureVoltPars'][pre_scan_key] = deepcopy(meas_volt_dict)
                # read latest triton settings from gui:
                self.triton_scan_dict[self.pre_or_during_scan_str] = self.get_current_triton_settings() \
                    if self.checkBox_triton_measure.isChecked() else {}
                self.sql_scan_dict[self.pre_or_during_scan_str] = self.get_current_sql_settings() \
                    if self.check_sql_measure.isChecked() else {}
                self.parent_ui.buffer_pars['triton'] = self.triton_scan_dict
                self.parent_ui.buffer_pars['sql'] = self.sql_scan_dict
                Cfg._main_instance.pre_scan_timeout_changed(self.doubleSpinBox_timeout_pre_scan_s.value())
                Cfg._main_instance.sql_set_interval(self.d_interval.value())
                # logging.info('set values to: ', Cfg._main_instance.scan_pars[self.active_iso]['measureVoltPars'])

            self.close()
        except Exception as e:
            logging.error('error while writing meas volt pars to main %s' % e, exc_info=True)

    def closeEvent(self, event):
        """
        when closing this window, unsubscirbe from main and from voltage readback.
        Also tell parent gui that it is closed.
        """
        if self.parent_ui is not None:
            Cfg._main_instance.dmm_gui_unsubscribe()
            self.voltage_reading.disconnect()
            self.parent_ui.close_pre_post_scan_win()

    def pre_post_during_changed(self, pre_post_during_str, closing_widget = False):
        """
        whenever this is changed, load stuff from main.
        :param: pre_post_during_str: the NEW tabs pre/post/during string
        """
        # store values from current setting first!
        self.current_meas_volt_settings[self.pre_or_during_scan_str] = self.get_current_meas_volt_pars()
        self.triton_scan_dict[self.pre_or_during_scan_str] = self.get_current_triton_settings()
        self.sql_scan_dict[self.pre_or_during_scan_str] = self.get_current_sql_settings()
        if not closing_widget: #Do this only when switching between tabs
            # remove all tabs to load them new from config
            while self.tabs.__len__() > 1:
                dmm_name = self.tabWidget.tabText(1)
                self.tabs.pop(dmm_name)
                self.tabWidget.removeTab(1)
            # for tab_ind in range(self.tabs.__len__()):
            #     if tab_ind != 0:
            #         dmm_name = self.tabWidget.tabText(tab_ind)
            #         self.tabs.pop(dmm_name)
            #         self.tabWidget.removeTab(tab_ind)
        # now load from main or from stored values for the newly selected pre/post/during thing
        self.pre_or_during_scan_str = pre_post_during_str
        existing_config = None
        existing_config = None if self.current_meas_volt_settings.get(self.pre_or_during_scan_str, {}) == {} \
            else self.current_meas_volt_settings
        self.setup_volt_meas_from_main(existing_config)
        self.setup_triton_devs()
        self.setup_sql_channels()

    def enable_triton_widgets(self, enable_bool):
        """
        disable the main widget and leaves empty dict at:
             {'triton': {self.pre_or_during_scan_str: {} ... }
        """
        self.listWidget_devices.setEnabled(enable_bool)
        self.tableWidget_channels.setEnabled(enable_bool)

    def triton_measure_checkbox_changed(self, state):
        """
        disable the main widget and leaves empty dict at:
             {'triton': {self.pre_or_during_scan_str: {} ... }
         if unchecked.
         If Checked it will load from backupt and force the gui to stay enabled
        """
        if state == 2:  # now checked
            self.triton_scan_dict[self.pre_or_during_scan_str] = deepcopy(self.triton_scan_dict_backup.get(
                self.pre_or_during_scan_str, {}))
        else:  # now unchecked
            self.check_any_ch_active()  # get current channel selection
            # store in backup
            self.triton_scan_dict_backup[self.pre_or_during_scan_str] = deepcopy(
                self.triton_scan_dict[self.pre_or_during_scan_str])
            #  clear current settins
            self.triton_scan_dict[self.pre_or_during_scan_str] = {}
        # setup gui according to the settings now.
        self.setup_triton_devs(state == 2)  # force measure if true

    def voltage_checkbox_changed(self, state):
        """
        disable the main widget and leaves empty dict at:
             {'measureVoltPars': {self.pre_or_during_scan_str: {'dmms': {}, ...} ... }
        """
        self.tabWidget.setEnabled(state == 2)
        self.doubleSpinBox_measVoltPulseLength_mu_s.setEnabled(state == 2)
        self.doubleSpinBox_measVoltTimeout_mu_s_set.setEnabled(state == 2)

    def sql_measure_checkbox_changed(self, state):
        """
        Enable/Disable scroll widget. Stop recording the selected data.
        disable the main widget and leaves empty dict at:
             {'triton': {self.pre_or_during_scan_str: {} ... }
         if unchecked.
         If Checked it will load from backupt and force the gui to stay enabled
        """
        if state == 2:  # now checked
            self.sql_scan_dict[self.pre_or_during_scan_str] = deepcopy(self.sql_scan_dict_backup.get(
                self.pre_or_during_scan_str, {}))
        else:  # now unchecked
            # self.check_any_ch_active()  # get current channel selection
            # store in backup
            self.sql_scan_dict_backup[self.pre_or_during_scan_str] = deepcopy(
                self.sql_scan_dict[self.pre_or_during_scan_str])
            #  clear current settins
            self.sql_scan_dict[self.pre_or_during_scan_str] = {}
        # setup gui according to the settings now.
        self.setup_sql_channels(state == 2)  # force measure if true

    ''' voltage related, mostly copied from DmmUi.py '''

    def setup_volt_meas_from_main(self, meas_volt_pars_dict=None):
        """
        setup gui with the scan parameters stored in the main or from the meas_volt_pars_dict
        :param meas_volt_pars_dict: dict, scan_dict['measureVoltPars'] = {'preScan': {...}}
        :return:
        """
        if self.active_iso is not None and self.parent_ui is not None:
            if meas_volt_pars_dict is None:
                # try to get it from main. Usually only the case when opening the window anew
                if self.parent_ui.buffer_pars is not None:
                    # First try to get it from buffer_pars of self.parent_ui!
                    meas_volt_pars_dict = self.parent_ui.buffer_pars['measureVoltPars']
                else:
                    # If for whatever reason buffer_pars are not available (should never be the case) try loading from main
                    scan_dict = Cfg._main_instance.scan_pars[self.active_iso]
                    meas_volt_pars_dict = scan_dict[self.act_track_name]['measureVoltPars']
            meas_volt_dict = meas_volt_pars_dict.get(self.pre_or_during_scan_str, None)
            if meas_volt_dict is None:  # there are no settings available yet for this pre/dur/post string
                self.set_pulse_len_and_timeout({})
                self.checkBox_voltage_measure.setChecked(False)
                self.voltage_checkbox_changed(0)
                for key, val in self.tabs.items():
                    if key != 'tab0':
                        if self.pre_or_during_scan_str == 'preScan':
                            logging.info('will set %s to preScan' % val[-1])
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
            self.checkBox_voltage_measure.setChecked(True)
            self.voltage_checkbox_changed(2)
            self.checkBox_triton_measure.setChecked(True)
            self.enable_triton_widgets(True)
            self.check_sql_measure.setChecked(True)
            self.enable_sql_widgets(True)

    def tab_wants_to_be_closed(self, *args):
        """
        when clicking on(X) on the tab this will be called and the dmm will be deinitialized.
        will be called by self.tabWidget.tabCloseRequested
        the user will be asked whether or not he wants to de-initialize the device
        :param args: tuple, (ind,)
        """
        tab_ind = args[0]
        if tab_ind:  # not for tab 0 which is teh selector tab
            dmm_name = self.tabWidget.tabText(tab_ind)
            de_init_bool = self.deinit_dialog(dmm_name) # Ask for deinit in a dialog!
            self.tabs.pop(dmm_name)
            self.tabWidget.removeTab(tab_ind)
            #if self.comm_enabled:  # deinit only if comm is allowed. This overrides a de-init wish of the user!
            if de_init_bool:
                logging.info('User requested deinitialization of %s' %dmm_name)
                self.deinit_dmm(dmm_name)

    def deinit_dialog(self, dmm_name):
        """
        Pops up a dialog to ask the user whether the chosen DMM should be de-initialized or kept active.
        :param dmm_name: Identifies the control
        :return: Bool: Whether the dmm was closed or not
        """
        msg = QtWidgets.QMessageBox()
        msg.setText('Do you also want to de-initialize %s?' % dmm_name)
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msg.setDetailedText(
            'If you want to continue using this DMM in any other pre- or post-measurements,'
            ' it should not be de-initialized here. If No is chosen, the DMM will only be removed'
            ' from the current pre/post/during measurement but not de-initialized.')
        retval = msg.exec_()  # 65536 for No and 16384 for Yes
        if retval == 65536:
            return False  # User does not want to de-initialize the dmm
        elif retval == 16384:
            return True  # User wants to de-initialize the dmm
        else:
            pass

    def initialize_dmm(self, dmm_tuple, startup_config=None):
        """
        will initialize the dmm of type and adress and store the instance in the scan_main.
        :param dmm_tuple: tuple, (dev_type_str, dev_addr_str)
        """
        logging.info('starting to initialize: %s %s ' % dmm_tuple)
        dev_type, dev_address = dmm_tuple
        dmm_name = dev_type + '_' + dev_address
        if dmm_name in list(self.tabs.keys()) or dmm_name is None:
            logging.warning('could not initialize: ', dmm_name, ' ... already initialized?')
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
        logging.info('setting up tab: ' + str(tpl))
        dmm_name, dev_type = tpl  # dmm_name = tab_name
        # logging.debug('done initializing: ', dmm_name, dev_type)
        if disconnect_signal:
            logging.debug('Main is done initializing: ' + str(tpl))
            self.init_dmm_done_callback.disconnect()
        self.tabs[dmm_name] = [None, None, None]
        self.tabs[dmm_name][0] = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tabs[dmm_name][0], dmm_name)
        self.tabs[dmm_name][1] = QtWidgets.QVBoxLayout(self.tabs[dmm_name][0])
        self.tabWidget.setCurrentWidget(self.tabs[dmm_name][0])
        self.tabs[dmm_name][2] = Ni4071Widg(dmm_name, dev_type)
        self.tabs[dmm_name][2].enable_communication(self.comm_enabled)
        self.tabs[dmm_name][1].addWidget(self.tabs[dmm_name][2])
        if disconnect_signal:
            # when done with initialising the device try to get the scan values
            # from the host ui / the main for this tab/dmm
            val = self.current_meas_volt_settings.get(self.pre_or_during_scan_str, {}).get(dmm_name, {})
            if val == {}:
                if self.parent_ui.buffer_pars is not None:
                    # First try to get it from buffer_pars of self.parent_ui!
                    meas_volt_pars_dict = self.parent_ui.buffer_pars['measureVoltPars']
                else:
                    # If for whatever reason buffer_pars are not available
                    #  (should never be the case) try loading from main
                    scan_dict = Cfg._main_instance.scan_pars[self.active_iso]
                    meas_volt_pars_dict = scan_dict[self.act_track_name]['measureVoltPars']
                meas_volt_dict = meas_volt_pars_dict.get(self.pre_or_during_scan_str, None)
                val = meas_volt_dict.get('dmms', {}).get(dmm_name, {})
            if val != {}:
                self.tabs[dmm_name][-1].load_dict_to_gui(val)
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
        timeout_10_ns = meas_volt_dict.get('measVoltTimeout10ns', -1)
        if timeout_10_ns is not None:
            timeout_volt_meas_mu_s = timeout_10_ns / 100000
            if timeout_10_ns == -1:
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
        :param meas_volt_dict: dict, keys are dmm_names, values are the config dicts for each dmm
        """
        logging.info('loading config dict: ' + str(meas_volt_dict))
        dmm_conf_dict = meas_volt_dict.get('dmms', {})
        self.checkBox_voltage_measure.setChecked(dmm_conf_dict != {})
        self.voltage_checkbox_changed(2 if dmm_conf_dict != {} else 0)
        self.set_pulse_len_and_timeout(meas_volt_dict)
        for key, val in dmm_conf_dict.items():  # key: dmm_name ;val: dmm configuration
            # first check whether the DMM is already intialized
            act_dmm_dict = Cfg._main_instance.get_active_dmms()
            if key in act_dmm_dict:
                # DMM is already initialized. Only load the info to the tab
                dmm_type, dmm_addr, state_str, last_readback, dmm_config = act_dmm_dict[key]
                self.setup_new_tab_widget((key, dmm_type), False)
                self.tabs[key][-1].load_dict_to_gui(val)
            else:
                # Not initialized yet.
                try:  # could possibly be removed. Was the previous standard call.
                    self.tabs[key][-1].load_dict_to_gui(val)
                except Exception as e:
                    logging.error(
                        'error: while loading the dmm vals to the gui of %s, this happened: %s' % (key, e))
                    warning_ms = key + ' was not yet initialized\n \n Do you want to initialize it now?'
                    try:
                        # Ask user whether he wants to initialize the DMM now
                        button_box_answer = QtWidgets.QMessageBox.question(
                            self, 'DMM not initialized', warning_ms,
                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
                        if button_box_answer == QtWidgets.QMessageBox.Yes:
                            # if user wants to initialize, do so
                            self.initialize_dmm((dmm_conf_dict[key].get('type', ''),
                                                 dmm_conf_dict[key].get('address', '')),
                                                val.get('preConfName', 'initial'))
                    except Exception as e:
                        logging.error(e)

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

    """ triton related """

    def setup_triton_devs(self, force_measure=False):
        """
        called on startup, fill list with devices
        and load settings from self.triton_scan_dict to gui.
        :return: dict, {dev: ['ch1', 'ch2' ...]}
        """
        self.listWidget_devices.clear()
        while self.tableWidget_channels.rowCount() > 0:
            self.tableWidget_channels.removeRow(0)
        self.cur_dev = None
        active_devices = self.get_channels_from_main()
        triton_pre_post_sc_dict = self.triton_scan_dict.get(self.pre_or_during_scan_str, {})
        measure = triton_pre_post_sc_dict != {} and triton_pre_post_sc_dict is not None or force_measure
        self.checkBox_triton_measure.blockSignals(True)
        self.checkBox_triton_measure.setChecked(measure)
        self.checkBox_triton_measure.blockSignals(False)
        self.enable_triton_widgets(measure)
        if measure:
            # list all active devices:
            for dev, ch_list in active_devices.items():
                dev_itm = QtWidgets.QListWidgetItem(dev, self.listWidget_devices)
                dev_itm.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                dev_itm.setCheckState(QtCore.Qt.Unchecked)
                if dev in triton_pre_post_sc_dict.keys():
                    dev_itm.setCheckState(QtCore.Qt.Checked)
                self.listWidget_devices.addItem(dev_itm)
            # also show not available devices which are still in the scan pars:
            for dev, ch_dict in triton_pre_post_sc_dict.items():
                if dev not in active_devices.keys():
                    dev_itm = QtWidgets.QListWidgetItem(dev, self.listWidget_devices)
                    dev_itm.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    dev_itm.setCheckState(QtCore.Qt.Checked)
                    dev_itm.setBackground(QtCore.Qt.red)
                    dev_itm.setToolTip('offline')
                    self.listWidget_devices.addItem(dev_itm)
        self.active_devices = active_devices
        if self.active_devices:
            self.dev_selection_changed(self.listWidget_devices.item(0))
        return active_devices

    def dev_selection_changed(self, cur):
        """
        when a device is clicked on, show the active channels of this device
        :param cur: QListWidgetItem, currently selected one.
        :return:
        """
        if cur is not None:
            cur_dev_selected = cur.checkState() == 2
            if self.cur_dev is not None:
                # this will store the settings in the gui of the self.cur_dev before changing to the next dev!
                self.triton_scan_dict[self.pre_or_during_scan_str] = self.get_current_triton_settings()
                # pass
            # now clear channels table and move on with next selected.
            while self.tableWidget_channels.rowCount() > 0:
                self.tableWidget_channels.removeRow(0)
            self.cur_dev = cur.text()
            logging.info('cur dev: %s checkstate: %s' %(self.cur_dev,cur_dev_selected))
            new_dev = False
            if self.cur_dev not in self.triton_scan_dict.get(self.pre_or_during_scan_str,
                                                             {}).keys() and cur_dev_selected:
                # device was not in current scan settings, will add now.
                # check for existing settings in backup:
                existing = deepcopy(
                    self.triton_scan_dict_backup.get(self.pre_or_during_scan_str, {}).get(self.cur_dev, {}))
                if existing == {}:  # was not in backup scan pars yet, will create new set
                    existing = {ch: {'required': 1, 'acquired': 0, 'data': []}
                                for ch in self.active_devices.get(self.cur_dev, {})}
                    new_dev = True
                if self.triton_scan_dict_backup.get(self.pre_or_during_scan_str, None) is None:
                    #  create an empty dict if this device was not in the scan pars yet
                    self.triton_scan_dict[self.pre_or_during_scan_str] = {}
                self.triton_scan_dict[self.pre_or_during_scan_str][self.cur_dev] = existing
            if not cur_dev_selected:
                # remove the device from the parameters if it is not selected!
                if self.cur_dev in self.triton_scan_dict[self.pre_or_during_scan_str].keys():
                    self.triton_scan_dict[self.pre_or_during_scan_str].pop(self.cur_dev)
            channels = self.active_devices.get(self.cur_dev, [])
            self.tableWidget_channels.setColumnCount(2)
            self.tableWidget_channels.setHorizontalHeaderLabels(['ch name', '# of samples'])
            self.tableWidget_channels.setRowCount(len(channels))

            for i, ch in enumerate(channels):
                ch_itm = QtWidgets.QTableWidgetItem(ch)
                ch_itm.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                ch_itm.setCheckState(QtCore.Qt.Checked if cur_dev_selected else QtCore.Qt.Unchecked)
                required_itm = QtWidgets.QTableWidgetItem('1')
                # required_itm.setFlags(QtCore.Qt.ItemIsEditable)
                # check all selected channels in the settings
                ch_selected_in_pars = ch in self.triton_scan_dict.get(
                    self.pre_or_during_scan_str, {}).get(self.cur_dev, {}).keys()
                check = QtCore.Qt.Checked if ch_selected_in_pars or new_dev else QtCore.Qt.Unchecked
                ch_itm.setCheckState(check)
                if ch_selected_in_pars:
                    required_itm.setText(
                        str(self.triton_scan_dict.get(self.pre_or_during_scan_str, {})[self.cur_dev][ch]['required']))
                self.tableWidget_channels.setItem(i, 0, ch_itm)
                self.tableWidget_channels.setItem(i, 1, required_itm)
            self.get_current_triton_settings()

    def get_selected_channels(self):
        """
        get the selected channels which are currently displayed.
        Note: This overrides the 'data' list, but one cannot access this gui for an go on a file, so this is ok.
        :return: dict, {'ch1': {'required': .. , 'acquired': 0, 'data': []}, ...}
        """
        ret = {}
        if self.cur_dev is not None:
            if self.tableWidget_channels.rowCount():
                for i in range(self.tableWidget_channels.rowCount()):
                    ch_itm = self.tableWidget_channels.item(i, 0)
                    if ch_itm.checkState() == 2:
                        req_itm = self.tableWidget_channels.item(i, 1)
                        sample_count = 0
                        try:
                            sample_count = int(req_itm.text())
                        except Exception as e:
                            logging.error('error in converting %s error is: %s' % (req_itm.text(), e))
                        if sample_count:
                            # TODO: in order to allow continous aquisition we need to allow negative
                            # and/or zero values here. Probably can remove the whole if statement
                            ret[ch_itm.text()] = {'required': sample_count,
                                                  'acquired': 0,
                                                  'data': []}
            else:
                return None
        else:
            return None
        return ret

    def check_any_ch_active(self):
        """
        check if any channel is selected in gui and update:

            self.triton_scan_dict[self.pre_or_during_scan_str][self.cur_dev]

        accordingly
        """
        if self.cur_dev is not None:
            logging.info('cur dev: %s' % self.cur_dev)
            ret = []
            for i in range(self.tableWidget_channels.rowCount()):
                ch_itm = self.tableWidget_channels.item(i, 0)
                ret.append(ch_itm.checkState() == 2)
            dev_itm = self.listWidget_devices.findItems(self.cur_dev, Qt.Qt.MatchExactly)[0]
            if any(ret):
                dev_itm.setCheckState(QtCore.Qt.Checked)
                channels = self.get_selected_channels()
                if channels is not None:
                    self.triton_scan_dict[self.pre_or_during_scan_str][self.cur_dev] = channels
            else:  # no ch active, can remove the dev
                if self.cur_dev in self.triton_scan_dict[self.pre_or_during_scan_str].keys():
                    self.triton_scan_dict[self.pre_or_during_scan_str].pop(self.cur_dev)
                dev_itm.setCheckState(QtCore.Qt.Unchecked)

    def get_channels_from_main(self):
        """
        get a dict of all online devices with a list of channels from the main
        :return: dict, {dev: ['ch1', 'ch2' ...]}
        """
        try:
            logging.info('getting active triton devs..')
            triton_dict = Cfg._main_instance.scan_main.get_available_triton()
            logging.info('active triton devs: %s' % str(triton_dict))
        except AttributeError:  # if no main available ( gui test etc.)
            triton_dict = {'no_main_dev': ['ch1', 'ch2'],
                           'no_main_dev2': ['ch1', 'ch2', 'ch3']}
        return triton_dict

    def get_triton_scan_pars(self):
        """
        get the triton part of the scan dict for one track in the main or return default
        :return: dict, for triton scan parameters, pre / during / post scan
        """
        default_ret = {'preScan': {'no_main_dev': {'ch2': {'required': 5,
                                                           'acquired': 0,
                                                           'data': []}},
                                   'no_main_dev2': {'ch2': {'required': 2,
                                                            'acquired': 0,
                                                            'data': []},
                                                    'ch3': {'required': 15,
                                                            'acquired': 0,
                                                            'data': []}
                                                    }
                                   },
                       'duringScan': {'no_main_dev': {'ch1': {'required': 5,
                                                              'acquired': 0,
                                                              'data': []}},
                                      'no_main_dev2': {'ch1': {'required': 2,
                                                               'acquired': 0,
                                                               'data': []},
                                                       'ch2': {'required': 15,
                                                               'acquired': 0,
                                                               'data': []}
                                                       }
                                      },
                       'postScan': {'no_main_dev3': {'ch2': {'required': 5,
                                                             'acquired': 0,
                                                             'data': []}},
                                    'no_main_dev4': {'ch2': {'required': 2,
                                                             'acquired': 0,
                                                             'data': []},
                                                     'ch3': {'required': 15,
                                                             'acquired': 0,
                                                             'data': []}
                                                     }
                                    }
                       }
        try:
            triton_dict = Cfg._main_instance.scan_pars[self.active_iso][self.act_track_name].get('triton', {})
        except AttributeError:  # if no main available ( gui test etc.)
            triton_dict = default_ret
        return triton_dict

    def get_current_triton_settings(self):
        """ return the current triton dict, which is updated constantly when user clicks on something. """
        # force update of self.triton_scan_dict[self.pre_or_during_scan_str][self.cur_dev] from gui:
        self.check_any_ch_active()
        # now check if this should be measured anyhow:
        triton_dict = self.triton_scan_dict[
            self.pre_or_during_scan_str] if self.checkBox_triton_measure.isChecked() else {}
        return triton_dict

    """ SQL related """

    # def add_observable(self):
    #     self.observables_measure.append(SQLObservableUi(self))
    #     self.scroll_widget_measure.layout().insertWidget(
    #         self.scroll_widget_measure.layout().count() - 2, self.observables_measure[-1])
    #     self.observables_measure[-1].show()

    def get_sql_channels_from_main(self):
        try:
            channels = Cfg._main_instance.scan_main.get_available_sql_channels()
        except AttributeError:  # if no main available ( gui test etc.)
            channels = ['dev0:ch0', 'dev0:ch1', 'dev1:ch0']
        return channels

    def enable_sql_widgets(self, enable_bool):
        self.table_sql.setEnabled(enable_bool)

    def setup_sql_channels(self, force_measure=False):
        self.table_sql.blockSignals(True)
        self.table_sql.setRowCount(0)
        active_channels = self.get_sql_channels_from_main()
        sql_pre_post_sc_dict = self.sql_scan_dict.get(self.pre_or_during_scan_str, {})
        measure = (sql_pre_post_sc_dict is not None and sql_pre_post_sc_dict != {}) or force_measure
        self.check_sql_measure.blockSignals(True)
        self.check_sql_measure.setChecked(measure)
        self.check_sql_measure.blockSignals(False)
        self.enable_sql_widgets(measure)
        if measure:
            # list all active devices:
            for i, ch in enumerate(active_channels):
                self.table_sql.insertRow(i)
                ch_item = QtWidgets.QTableWidgetItem(ch)
                ch_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                ch_item.setCheckState(QtCore.Qt.Unchecked)
                req_item = QtWidgets.QTableWidgetItem('1')
                # req_item.setFlags(QtCore.Qt.ItemIsEditable)
                if ch in sql_pre_post_sc_dict.keys():
                    ch_item.setCheckState(QtCore.Qt.Checked)
                    req_item.setText(str(sql_pre_post_sc_dict[ch]['required']))
                # check all selected channels in the settings
                self.table_sql.setItem(i, 0, ch_item)
                self.table_sql.setItem(i, 1, req_item)
            # also show not available devices which are still in the scan pars:
            i0 = self.table_sql.rowCount()
            for i, (ch, req_dict) in enumerate(sql_pre_post_sc_dict.items()):
                if ch not in active_channels:
                    self.table_sql.insertRow(i0 + i)
                    ch_item = QtWidgets.QTableWidgetItem(ch)
                    ch_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    ch_item.setCheckState(QtCore.Qt.Checked)
                    ch_item.setBackground(QtCore.Qt.red)
                    ch_item.setToolTip('offline')
                    ch_item.setCheckState(QtCore.Qt.Checked)
                    req_item = QtWidgets.QTableWidgetItem('1')
                    req_item.setBackground(QtCore.Qt.red)
                    req_item.setToolTip('offline')
                    # req_item.setFlags(QtCore.Qt.ItemIsEditable)
                    req_item.setText(str(sql_pre_post_sc_dict[ch]['required']))
                    self.table_sql.setItem(i0 + i, 0, ch_item)
                    self.table_sql.setItem(i0 + i, 1, req_item)
        self.active_channels = active_channels
        self.table_sql.blockSignals(False)
        return active_channels

    def get_sql_scan_pars(self):
        """
        Get the SQL part of the scan dict for one track in the main or return default.
        
        :returns: dict, for SQL scan parameters, pre / during / post scan.
        """
        default_ret = {'preScan': {'ch0': {'required': 10, 'acquired': 0, 'data': []}},
                       'duringScan': {},
                       'postScan': {}}
        try:
            sql_dict = Cfg._main_instance.scan_pars[self.active_iso][self.act_track_name].get('sql', {})
        except AttributeError:  # if no main available ( gui test etc.)
            sql_dict = default_ret
        return sql_dict

    def get_selected_sql_channels(self):
        """
        get the selected channels which are currently displayed.
        Note: This overrides the 'data' list, but one cannot access this gui for a go on a file, so this is ok.
        :return: dict, {'ch1': {'required': .. , 'acquired': 0, 'data': []}, ...}
        """
        ret = {}
        for i in range(self.table_sql.rowCount()):
            ch_item = self.table_sql.item(i, 0)
            req_item = self.table_sql.item(i, 1)
            if ch_item.checkState() == 2:
                ret[ch_item.text()] = {'required': int(req_item.text()), 'acquired': 0, 'data': []}
        return ret

    def sql_process_row(self, item):
        if item.column() == 0:
            self.sql_scan_dict[self.pre_or_during_scan_str][item.text()] = \
                {'required': int(self.table_sql.item(item.row(), 1).text()), 'acquired': 0, 'data': []}
        else:
            text = item.text()
            try:
                val = int(text)
            except ValueError:
                val = 1
            if val < 0 and self.pre_or_during_scan_str == 'duringScan':
                val = -1
            item.setText(str(val))

    def check_any_sql_channel_active(self):
        channels = self.get_selected_sql_channels()
        if channels:
            self.sql_scan_dict[self.pre_or_during_scan_str] = channels

    def get_current_sql_settings(self):
        """ return the current sql dict, which is updated constantly when user clicks on something. """
        # force update of self.sql_scan_dict[self.pre_or_during_scan_str] from gui:
        self.check_any_sql_channel_active()
        # now check if this should be measured anyhow:
        sql_dict = self.sql_scan_dict[self.pre_or_during_scan_str] if self.check_sql_measure.isChecked() else {}
        return sql_dict


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    gui = PreScanConfigUi(None, '', '')
    app.exec_()
