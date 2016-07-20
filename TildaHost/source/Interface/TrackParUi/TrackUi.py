"""

Created on '29.09.2015'

@author:'simkaufm'

"""
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import ast
import logging
from copy import deepcopy
import math

from Interface.TrackParUi.Ui_TrackPar import Ui_MainWindowTrackPars
from Interface.SetVoltageUi.SetVoltageUi import SetVoltageUi
import Interface.SequencerWidgets.FindDesiredSeqWidg as FindDesiredSeqWidg
import Service.Scan.ScanDictionaryOperations as SdOp
import Service.VoltageConversions.VoltageConversions as VCon
import Application.Config as Cfg
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs
import Interface.TriggerWidgets.FindDesiredTriggerWidg as FindDesiredTriggerWidg


class TrackUi(QtWidgets.QMainWindow, Ui_MainWindowTrackPars):
    track_ui_call_back_signal = QtCore.pyqtSignal(dict)

    def __init__(self, scan_ctrl_win, track_number, active_iso_name, main_gui):
        """
        Non modal Main window to determine the scanparameters for a single track of a given isotope.
        scan_ctrl_win is needed for writing the track dictionary to a given scan dictionary.
        track_number is the number of the track which will be worked on.
        default_track_dict is the default dictionary which will be deepcopied and then worked on.
        """
        super(TrackUi, self).__init__()

        self.track_name = 'track' + str(track_number)
        self.scan_ctrl_win = scan_ctrl_win
        self.active_iso = scan_ctrl_win.active_iso
        seq_type = self.active_iso.split('_')[-1]
        self.track_number = track_number
        self.main_gui = main_gui

        self.buffer_pars = deepcopy(Cfg._main_instance.scan_pars.get(active_iso_name).get(self.track_name))
        self.buffer_pars['dacStopRegister18Bit'] = self.calc_dac_stop_18bit()  # is needed to be able to fix stop
        self.dac_stop_bit_user = self.calc_dac_stop_18bit()

        self.track_ui_call_back_signal.connect(self.refresh_pow_sup_readback)
        self.set_volt_win = None

        self.setupUi(self)

        self.setWindowTitle(self.scan_ctrl_win.win_title + '_' + self.track_name)

        """ sequencer specific """
        self.sequencer_widget = FindDesiredSeqWidg.find_sequencer_widget(seq_type, self.buffer_pars, self.main_gui)
        self.verticalLayout.replaceWidget(self.specificSequencerSettings, self.sequencer_widget)

        """ Trigger related """
        self.checkBox.setDisabled(True)
        self.checkBox.setToolTip('not yet included')
        self.trigger_widget = None
        self.update_trigger_combob()
        self.trigger_widget = FindDesiredTriggerWidg.find_trigger_widget(self.buffer_pars.get('trigger', {}))
        self.trigger_vert_layout.replaceWidget(self.widget_trigger_place_holder, self.trigger_widget)
        self.comboBox_triggerSelect.currentTextChanged.connect(self.trigger_select)

        """DAC Settings"""
        self.doubleSpinBox_dacStartV.setRange(VCon.get_voltage_from_18bit(0), VCon.get_voltage_from_18bit(2 ** 18 - 1))
        self.doubleSpinBox_dacStartV.valueChanged.connect(self.dac_start_v_set)

        self.doubleSpinBox_dacStopV.setRange(VCon.get_voltage_from_18bit(0), VCon.get_voltage_from_18bit(2 ** 18 - 1))
        self.doubleSpinBox_dacStopV.valueChanged.connect(self.dac_stop_v_set)

        self.doubleSpinBox_dacStepSizeV.setRange(
            VCon.get_voltage_from_18bit(-(2 ** 18 - 1)), VCon.get_voltage_from_18bit(2 ** 18 - 1))
        self.doubleSpinBox_dacStepSizeV.valueChanged.connect(self.dac_step_size_set)

        self.spinBox_nOfSteps.setRange(2, 2 ** 18)
        self.spinBox_nOfSteps.valueChanged.connect(self.n_of_steps_set)

        self.spinBox_nOfScans.valueChanged.connect(self.n_of_scans_set)
        self.checkBox_invertScan.stateChanged.connect(self.invert_scan_set)

        """post acceleration controls:"""
        self.comboBox_postAccOffsetVoltControl.currentIndexChanged.connect(self.post_acc_offset_volt_control_set)
        self.doubleSpinBox_postAccOffsetVolt.valueChanged.connect(self.post_acc_offset_volt)

        """Scaler selection:"""
        self.lineEdit_activePmtList.textChanged.connect(self.active_pmt_list_set)

        """collinear/anticollinear"""
        self.checkBox_colDirTrue.stateChanged.connect(self.col_dir_true_set)

        """Buttons:"""
        self.pushButton_cancel.clicked.connect(self.cancel)
        self.pushButton_confirm.clicked.connect(self.confirm)
        self.pushButtonResetToDefault.clicked.connect(self.reset_to_default)
        self.pushButton_postAccOffsetVolt.clicked.connect(self.set_voltage)

        """Advanced Settings:"""
        self.doubleSpinBox_waitAfterReset_muS.valueChanged.connect(self.wait_after_reset_mu_sec_set)
        self.doubleSpinBox_waitForKepco_muS.valueChanged.connect(self.wait_for_kepco_mu_sec)

        self.set_labels_by_dict(self.buffer_pars)
        self.show()

    """functions:"""
    def set_labels_by_dict(self, track_dict):
        """" the values in the track_dict will be written to the corresponding spinboxes,
        which will call the connected functions.
        Each function is tried separately in order to give the next one a chance of execution,
        when default val is messed up.
        """
        cb_post_acc_ind_before_load = self.comboBox_postAccOffsetVoltControl.currentIndex()
        print('setting trackui labels by dict: ', track_dict)
        func_list = [
            # (self.doubleSpinBox_dwellTime_ms.setValue,
            #  self.check_for_none(track_dict.get('dwellTime10ns'), 0) * (10 ** -5)),
            (self.doubleSpinBox_dacStartV.setValue,
             VCon.get_voltage_from_18bit(self.check_for_none(track_dict.get('dacStartRegister18Bit'), 0))),
            (self.doubleSpinBox_dacStopV.setValue,
             VCon.get_voltage_from_18bit(self.check_for_none(track_dict.get('dacStopRegister18Bit'), 2 ** 18))),
            (self.spinBox_nOfSteps.setValue, self.check_for_none(track_dict.get('nOfSteps'), 0)),
            (self.doubleSpinBox_dacStepSizeV.setValue,
             VCon.get_stepsize_in_volt_from_18bit(self.check_for_none(track_dict.get('dacStepSize18Bit'), 0))),
            (self.spinBox_nOfScans.setValue, self.check_for_none(track_dict.get('nOfScans'), 0)),
            (self.checkBox_invertScan.setChecked, self.check_for_none(track_dict.get('invertScan'), False)),
            (self.invert_scan_set, self.check_for_none(track_dict.get('invertScan'), False)),
            (self.comboBox_postAccOffsetVoltControl.setCurrentIndex,
             int(self.check_for_none(track_dict.get('postAccOffsetVoltControl'), 0))),
            (self.doubleSpinBox_postAccOffsetVolt.setValue,
             self.check_for_none(track_dict.get('postAccOffsetVolt'), 0)),
            (self.lineEdit_activePmtList.setText,
             str(self.check_for_none(track_dict.get('activePmtList'), [0]))[1:-1]),
            (self.checkBox_colDirTrue.setChecked, self.check_for_none(track_dict.get('colDirTrue'), False)),
            (self.col_dir_true_set, self.check_for_none(track_dict.get('colDirTrue'), False)),
            (self.doubleSpinBox_waitAfterReset_muS.setValue,
             self.check_for_none(track_dict.get('waitAfterReset25nsTicks'), 0) * 25 * (10 ** -3)),
            (self.doubleSpinBox_waitForKepco_muS.setValue,
             self.check_for_none(track_dict.get('waitForKepco25nsTicks'), 0) * 25 * (10 ** -3))
        ]
        for func in func_list:
            try:
                func[0](func[1])
            except Exception as e:
               print('error while loading default track dictionary: ' + str(e))
        # self.comboBox_postAccOffsetVoltControl.currentIndexChanged.emit(self.comboBox_postAccOffsetVoltControl.currentIndex())
        print('setting trackui labels by dict is done postAccOffsetVoltControl is: ',
              self.buffer_pars['postAccOffsetVoltControl'])  #
        if cb_post_acc_ind_before_load == self.comboBox_postAccOffsetVoltControl.currentIndex():
            # force index change emit if the index was not changed by loading
            self.comboBox_postAccOffsetVoltControl.currentIndexChanged.emit(
                int(self.check_for_none(track_dict.get('postAccOffsetVoltControl'), 0)))

    def check_for_none(self, check, replace):
        """
        checks if param "check" is None and replaces it if it is None
        """
        if check is None:
            check = replace
        return check

    def update_trigger_combob(self, default_trig=TiTs.no_trigger):
        """
        updates the trigger combo box by looking up the members of the enum
        """
        self.comboBox_triggerSelect.addItems([tr.name for tr in TiTs])
        self.comboBox_triggerSelect.setCurrentIndex(self.buffer_pars.get(
            'trigger', {'type': default_trig}).get('type', default_trig.value).value)

    def trigger_select(self, trig_str):
        """
        finds the deisred trigger widget and sets it into self.trigger_widget
        """
        self.buffer_pars.get('trigger', {})['type'] = getattr(TiTs, trig_str)
        self.trigger_vert_layout.removeWidget(self.trigger_widget)
        self.trigger_widget.setParent(None)
        self.trigger_widget = FindDesiredTriggerWidg.find_trigger_widget(self.buffer_pars.get('trigger', {}))
        self.trigger_vert_layout.addWidget(self.trigger_widget)

    """ from lineedit/spinbox to set value """
    '''line voltage realted:'''
    def dac_start_v_set(self, start_volt):
        """ this will write the doublespinbox value to the working dict and set the label
        it will also call recalc_step_stop to adjust the stepsize and then fine tune the stop value """
        start_18bit = VCon.get_18bit_from_voltage(start_volt)
        start_volt = VCon.get_voltage_from_18bit(start_18bit)
        self.buffer_pars['dacStartRegister18Bit'] = start_18bit
        self.label_dacStartV_set.setText(str(round(start_volt, 8)) + ' | ' + str(start_18bit))
        self.label_kepco_start.setText(str(round(start_volt * 50, 2)))
        self.doubleSpinBox_dacStartV.blockSignals(True)
        self.doubleSpinBox_dacStartV.setValue(start_volt)
        self.doubleSpinBox_dacStartV.blockSignals(False)
        dis = self.buffer_pars['dacStopRegister18Bit'] - self.buffer_pars['dacStartRegister18Bit']
        self.buffer_pars['dacStepSize18Bit'] = math.copysign(self.buffer_pars['dacStepSize18Bit'], dis)
        self.recalc_n_of_steps_stop()

    def dac_stop_v_set(self, stop_volt):
        """ this will write the doublespinbox value to the working dict and set the label
        it will also call recalc_n_of_steps_stop to adjust the number of steps and then fine tune the stop value """
        self.buffer_pars['dacStopRegister18Bit'] = VCon.get_18bit_from_voltage(stop_volt)
        self.dac_stop_bit_user = VCon.get_18bit_from_voltage(stop_volt)  # only touch this when double spinbox is touched
        dis = self.buffer_pars['dacStopRegister18Bit'] - self.buffer_pars['dacStartRegister18Bit']
        self.buffer_pars['dacStepSize18Bit'] = math.copysign(self.buffer_pars['dacStepSize18Bit'], dis)
        self.recalc_n_of_steps_stop()

    def display_stop(self, stop_18bit):
        """ function only for displaying the stop value """
        setval = VCon.get_voltage_from_18bit(stop_18bit)
        self.buffer_pars['dacStopRegister18Bit'] = stop_18bit
        self.label_dacStopV_set.setText(str(round(setval, 8)) + ' | ' + str(stop_18bit))
        self.label_kepco_stop.setText(str(round(setval * 50, 2)))
        self.doubleSpinBox_dacStopV.blockSignals(True)
        self.doubleSpinBox_dacStopV.setValue(setval)
        self.doubleSpinBox_dacStopV.blockSignals(False)

    def recalc_n_of_steps_stop(self):
        """ start and stop should be more or less constant if the stepsize changes,
        the number of steps must be adjusted.
        after adjusting the number of steps, the stop voltage is fine tuned to the next possible value.
        """
        try:
            self.display_step_size(self.buffer_pars['dacStepSize18Bit'])
            self.display_n_of_steps(self.calc_n_of_steps())
            stop = self.calc_dac_stop_18bit()
            if stop < 0 or stop == self.buffer_pars['dacStartRegister18Bit'] or stop >= 2 ** 18:
                return False
            self.display_stop(stop)
            return True
        except Exception as e:
            logging.error('the following error occurred while calculating the number of steps:'
                          + str(e))

    def calc_dac_stop_18bit(self):
        """ calculates the dac stop voltage in 18bit: stop = start + step * steps """
        try:
            start = self.check_for_none(self.buffer_pars['dacStartRegister18Bit'], 0)
            step = self.check_for_none(self.buffer_pars['dacStepSize18Bit'], 1)
            steps = self.check_for_none(self.buffer_pars['nOfSteps'], 1)
            stop = VCon.calc_dac_stop_18bit(start, step, steps)
        except Exception as e:
            logging.error('following error occurred while calculating the stop voltage:' + str(e))
            stop = 0
        return stop

    def calc_step_size(self):
        """ calculates the stepsize: (stop - start) / nOfSteps  """
        try:
            start = self.check_for_none(self.buffer_pars.get('dacStartRegister18Bit'), 0)
            stop = self.dac_stop_bit_user
            steps = self.check_for_none(self.buffer_pars.get('nOfSteps'), 1)
            stepsize_18bit = VCon.calc_step_size(start, stop, steps)
        except Exception as e:
            logging.error('following error occurred while calculating the stepsize:' + str(e))
            stepsize_18bit = 0
        return stepsize_18bit

    def calc_n_of_steps(self):
        """ calculates the number of steps: abs((stop - start) / stepSize) """
        try:
            start = self.check_for_none(self.buffer_pars.get('dacStartRegister18Bit'), 0)
            stop = self.dac_stop_bit_user
            step = self.check_for_none(self.buffer_pars.get('dacStepSize18Bit'), 1)
            n_of_steps = VCon.calc_n_of_steps(start, stop, step)
        except Exception as e:
            logging.error('following error occurred while calculating the number of steps:' + str(e))
            n_of_steps = 0
        return n_of_steps  # sign should always be in the stepsize

    def dac_step_size_set(self, step_volt):
        """ if the stepsize is set, adjust the number of steps to keep start and stop constant"""
        last_step_18bit = self.buffer_pars['dacStepSize18Bit']
        step_18bit = VCon.get_18bit_stepsize(step_volt)
        self.display_step_size(step_18bit)
        if not self.recalc_n_of_steps_stop():  # for invalid stop value, return to last valid value
            self.display_step_size(last_step_18bit)
            self.recalc_n_of_steps_stop()

    def display_step_size(self, step_18bit):
        """ stores the stepSize to the working dictionary and displays them """
        self.buffer_pars['dacStepSize18Bit'] = step_18bit
        step_volt = VCon.get_stepsize_in_volt_from_18bit(step_18bit)
        self.label_dacStepSizeV_set.setText(str(round(step_volt, 8)) + ' | ' + str(step_18bit))
        self.label_kepco_step.setText(str(round(step_volt * 50, 2)))
        self.doubleSpinBox_dacStepSizeV.blockSignals(True)
        self.doubleSpinBox_dacStepSizeV.setValue(step_volt)
        self.doubleSpinBox_dacStepSizeV.blockSignals(False)

    def n_of_steps_set(self, steps):
        """ displays the number of steps that where set and recalculates the stepSize
         in order to keep start and stop more or less constant """
        self.display_n_of_steps(steps)
        self.display_step_size(self.calc_step_size())
        self.display_stop(self.calc_dac_stop_18bit())

    def display_n_of_steps(self, steps):
        """ write the number of steps to the working dictionary and display them """
        steps = int(steps)
        self.label_nOfSteps_set.setText(str(steps))
        self.buffer_pars['nOfSteps'] = steps
        self.spinBox_nOfSteps.blockSignals(True)
        self.spinBox_nOfSteps.setValue(steps)
        self.spinBox_nOfSteps.blockSignals(False)

    '''other scan pars:'''
    def n_of_scans_set(self, val):
        """ write the number of scans to the working dictionary and display them """
        self.label_nOfScans_set.setText(str(val))
        self.buffer_pars['nOfScans'] = val

    def invert_scan_set(self, state):
        """ write to the working dictionary and set the label """
        if state:
            boolstate = True
        else:
            boolstate = False
        self.label_invertScan_set.setText(str(boolstate))
        self.buffer_pars['invertScan'] = boolstate

    def active_pmt_list_set(self, lis):
        """ write to the working dictionary and set the label """
        if type(lis) == str:
            try:
                lis = str('[' + lis + ']')
                lis = ast.literal_eval(lis)
            except Exception as e:
                logging.debug('while converting the pmt list, this occurred: ' + str(e))
                lis = []
        self.label_activePmtList_set.setText(str(lis))
        self.buffer_pars['activePmtList'] = lis

    def col_dir_true_set(self, state):
        """ write to the working dictionary and set the label """
        if state:
            display = 'colinear'
            boolstate = True
        else:
            display = 'anti colinear'
            boolstate = False
        self.label_colDirTrue_set.setText(display)
        self.buffer_pars['colDirTrue'] = boolstate

    def wait_after_reset_mu_sec_set(self, time_mu_s):
        """ write to the working dictionary and set the label """
        time_25ns = int(round(time_mu_s / 25 * (10 ** 3)))
        self.buffer_pars['waitAfterReset25nsTicks'] = time_25ns
        setval = time_25ns * 25 * (10 ** -3)
        self.label_waitAfterReset_muS_set.setText(str(round(setval, 3)))

    def wait_for_kepco_mu_sec(self, time_mu_s):
        """ write to the working dictionary and set the label """
        time_25ns = int(round(time_mu_s / 25 * (10 ** 3)))
        self.buffer_pars['waitForKepco25nsTicks'] = time_25ns
        setval = time_25ns * 25 * (10 ** -3)
        self.label_waitForKepco_muS_set.setText(str(round(setval, 3)))

    """ set voltages """
    def post_acc_offset_volt_control_set(self, index):
        """ write to the working dictionary and set the label """
        val = self.comboBox_postAccOffsetVoltControl.currentText()
        if val != 'Kepco':
            Cfg._main_instance.power_supply_status(val, self.track_ui_call_back_signal)
        self.label_postAccOffsetVoltControl_set.setText(val)
        self.buffer_pars['postAccOffsetVoltControl'] = index

    def refresh_pow_sup_readback(self, stat_dict):
        """
        refresh the readback voltage whenever the signal is triggered.
        """
        name = self.comboBox_postAccOffsetVoltControl.currentText()
        try:
            self.label_postAccOffsetVoltControl_set.setText(
                str(stat_dict.get(name, {}).get('readBackVolt', 'not initalized')))
        except Exception as e:
            logging.error('while reading the status, this happened: ' + str(e))

    def post_acc_offset_volt(self, val):
        """ write to the working dictionary and set the label """
        self.label_postAccOffsetVolt_set.setText(str(val))
        self.buffer_pars['postAccOffsetVolt'] = val

    def set_voltage(self):
        """ this will connect to the corresponding Heinzinger and set the voltage.
         this helps the user to enter the voltage a while before the scan."""
        power_supply = self.comboBox_postAccOffsetVoltControl.currentText()
        if power_supply != 'Kepco':
            volt = self.buffer_pars['postAccOffsetVolt']
            Cfg._main_instance.set_power_supply_voltage(power_supply, volt, self.track_ui_call_back_signal)
            self.set_volt_win = SetVoltageUi(power_supply, volt, self.track_ui_call_back_signal)

    def cancel(self):
        """ closes the window without further actions """
        self.close()

    def confirm(self):
        """ closes the window and overwrites the corresponding track in the main """
        self.buffer_pars = SdOp.merge_dicts(self.buffer_pars, self.sequencer_widget.get_seq_pars())
        self.buffer_pars['trigger'] = self.trigger_widget.get_trig_pars()
        Cfg._main_instance.scan_pars[self.active_iso][self.track_name] = deepcopy(self.buffer_pars)
        self.close()

    def reset_to_default(self):
        """ will reset all spinboxes to the default value which is stored in teh main. """
        default_d = deepcopy(Cfg._main_instance.scan_pars[self.active_iso][self.track_name])
        default_d['dacStopRegister18Bit'] = self.calc_dac_stop_18bit()
        self.set_labels_by_dict(default_d)

    def closeEvent(self, event):
        """
        will remove the given track window from the dictionary in scan_ctrl_win
        """
        self.scan_ctrl_win.track_win_closed(self.track_number)
