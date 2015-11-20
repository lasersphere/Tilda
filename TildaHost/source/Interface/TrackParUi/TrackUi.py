"""

Created on '29.09.2015'

@author:'simkaufm'

"""
from PyQt5 import QtWidgets
import ast
import logging
from copy import deepcopy

from Interface.TrackParUi.Ui_TrackPar import Ui_MainWindowTrackPars
from Interface.SetVoltageUi.SetVoltageUi import SetVoltageUi
import Service.VoltageConversions.VoltageConversions as VCon
import Service.Scan.ScanDictionaryOperations as SdOp


class TrackUi(QtWidgets.QMainWindow, Ui_MainWindowTrackPars):
    def __init__(self, scan_ctrl_win, track_number, default_track_dict):
        """ Non.modal Main window to determine the scanparameters for a single track of a given isotope.
         scan_ctrl_win is needed for writing the track dictionary to a given scan dictionary.
         track_number is the number of the track which will be worked on.
         default_track_dict is the default dictionary which will be deepcopied and then worked on."""
        super(TrackUi, self).__init__()

        self.default_track_dict = deepcopy(default_track_dict)
        self.buffer_pars = deepcopy(default_track_dict)
        self.buffer_pars['dacStopRegister18Bit'] = self.calc_dac_stop_18bit()  # is needed to be able to fix stop
        self.default_track_dict['dacStopRegister18Bit'] = self.calc_dac_stop_18bit()

        self.scan_ctrl_win = scan_ctrl_win
        self.track_number = track_number

        self.setupUi(self)
        self.setWindowTitle(self.scan_ctrl_win.win_title + ' - track' + str(track_number))
        self.show()

        """Sequencer Settings:"""
        self.doubleSpinBox_dwellTime_ms.valueChanged.connect(self.dwelltime_set)

        """DAC Settings"""
        self.doubleSpinBox_dacStartV.valueChanged.connect(self.dac_start_v_set)
        self.doubleSpinBox_dacStopV.valueChanged.connect(self.dac_stop_v_set)
        self.doubleSpinBox_dacStepSizeV.valueChanged.connect(self.dac_step_size_set)
        self.spinBox_nOfSteps.valueChanged.connect(self.n_of_steps_set)
        self.spinBox_nOfScans.valueChanged.connect(self.n_of_scans_set)
        self.checkBox_invertScan.stateChanged.connect(self.invert_scan_set)

        """post acceleration controls:"""
        self.comboBox_postAccOffsetVoltControl.currentTextChanged.connect(self.post_acc_offset_volt_control_set)
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

        print(str(self.buffer_pars))
        self.set_labels_by_dict(self.buffer_pars)

    """functions:"""
    def set_labels_by_dict(self, track_dict):
        """" the values in the track_dict will be written to the corresponding spinboxes,
         which will call the connected functions """
        func_list = [
            (self.doubleSpinBox_dwellTime_ms.setValue,
             self.check_for_none(track_dict.get('dwellTime10ns'), 0) * (10 ** -5)),
            (self.doubleSpinBox_dacStartV.setValue,
             VCon.get_voltage_from_18bit(self.check_for_none(track_dict.get('dacStartRegister18Bit'), 0))),
            (self.doubleSpinBox_dacStopV.setValue,
             VCon.get_voltage_from_18bit(self.check_for_none(track_dict.get('dacStopRegister18Bit'), 2 ** 18))),
            (self.spinBox_nOfSteps.setValue, self.check_for_none(track_dict.get('nOfSteps'), 0)),
            (self.doubleSpinBox_dacStepSizeV.setValue,
             VCon.get_stepsize_in_volt_from_18bit(self.check_for_none(track_dict.get('dacStepSize18Bit'), 0))),
            (self.spinBox_nOfScans.setValue, self.check_for_none(track_dict.get('nOfScans'), 0)),
            (self.checkBox_invertScan.setChecked, self.check_for_none(track_dict.get('invertScan'), False)),
            (self.comboBox_postAccOffsetVoltControl.setCurrentIndex,
             int(self.check_for_none(track_dict.get('postAccOffsetVoltControl'), 0))),
            (
            self.doubleSpinBox_postAccOffsetVolt.setValue, self.check_for_none(track_dict.get('postAccOffsetVolt'), 0)),
            (self.lineEdit_activePmtList.setText, str(self.check_for_none(track_dict.get('activePmtList'), [0]))[1:-1]),
            (self.checkBox_colDirTrue.setChecked, self.check_for_none(track_dict.get('colDirTrue'), False)),
            (self.doubleSpinBox_waitAfterReset_muS.setValue,
             self.check_for_none(track_dict.get('waitAfterReset25nsTicks'), 0) * 25 * (10 ** -3)),
            (self.doubleSpinBox_waitForKepco_muS.setValue,
             self.check_for_none(track_dict.get('waitForKepco25nsTicks'), 0) * 25 * (10 ** -3))
        ]
        for func in func_list:
            try:
                func[0](func[1])
            except Exception as e:
                logging.error('error while loading default track dictionary: ' + str(e))

    def check_for_none(self, check, replace):
        if check is None:
            check = replace
        return check

    def dwelltime_set(self, val):
        """ this will write the doublespinbox value to the working dict and set the label """
        self.buffer_pars['dwellTime10ns'] = val * (10 ** 5)  # convert to units of 10ns
        self.label_dwellTime_ms_2.setText(str(val))

    def dac_start_v_set(self, start_volt):
        """ this will write the doublespinbox value to the working dict and set the label
        it will also call recalc_step_stop to adjust the stepsize and then fine tune the stop value """
        start_18bit = VCon.get_18bit_from_voltage(start_volt)
        start_volt = VCon.get_voltage_from_18bit(start_18bit)
        self.buffer_pars['dacStartRegister18Bit'] = start_18bit
        self.label_dacStartV_set.setText(str(start_volt) + ' | ' + str(format(start_18bit, '018b'))
                                         + ' | ' + str(start_18bit))
        self.label_kepco_start.setText(str(round(start_volt * 50, 2)))
        self.recalc_step_stop()

    def dac_stop_v_set(self, stop_volt):
        """ this will write the doublespinbox value to the working dict and set the label
        it will also call recalc_step_stop to adjust the stepsize and then fine tune the stop value """
        self.buffer_pars['dacStopRegister18Bit'] = VCon.get_18bit_from_voltage(stop_volt)
        self.recalc_step_stop()

    def display_stop(self, stop_18bit):
        """ function only for displaying the stop value """
        setval = VCon.get_voltage_from_18bit(stop_18bit)
        self.label_dacStopV_set.setText(str(setval) + ' | ' + str(format(stop_18bit, '018b'))
                                        + ' | ' + str(stop_18bit))
        self.label_kepco_stop.setText(str(round(setval * 50, 2)))

    def recalc_step_stop(self):
        """ start and stop should be more or less constant therefore in most cases the stepsize must be adjusted.
         after adjusting the step size, the stop voltage is fine tuned to the next possible value. """
        try:
            self.display_step_size(self.calc_step_size())
            self.display_stop(self.calc_dac_stop_18bit())
        except Exception as e:
            logging.error('the following error occurred while calculating the number of steps:'
                          + str(e))

    def recalc_n_of_steps_stop(self):
        """ start and stop should be more or less constant if the stepsize changes,
         the number of steps must be adjusted.
         after adjusting the nuumber of steps, the stop voltage is fine tuned to the next possible value. """
        try:
            self.display_n_of_steps(self.calc_n_of_steps())
            self.display_stop(self.calc_dac_stop_18bit())
        except Exception as e:
            logging.error('the following error occurred while calculating the number of steps:'
                          + str(e))

    def calc_dac_stop_18bit(self):
        """ calculates the dac stop voltage in 18bit: stop = start + step * steps """
        try:
            start = self.buffer_pars['dacStartRegister18Bit']
            step = self.buffer_pars['dacStepSize18Bit']
            steps = self.buffer_pars['nOfSteps']
            stop = start + step * steps
        except Exception as e:
            logging.error('following error occurred while calculating the stop voltage:' + str(e))
            stop = 0
        return stop

    def calc_step_size(self):
        """ calculates the stepsize: (stop - start) / nOfSteps  """
        try:
            start = self.buffer_pars['dacStartRegister18Bit']
            stop = self.buffer_pars['dacStopRegister18Bit']
            steps = self.buffer_pars['nOfSteps']
            dis = stop - start
            stepsize_18bit = int(round(dis / steps))
        except ZeroDivisionError:
            stepsize_18bit = 0
        except Exception as e:
            logging.error('following error occurred while calculating the stepsize:' + str(e))
            stepsize_18bit = 0
        return stepsize_18bit

    def calc_n_of_steps(self):
        """ calculates the number of steps: abs((stop - start) / stepSize) """
        try:
            start = self.buffer_pars['dacStartRegister18Bit']
            stop = self.buffer_pars['dacStopRegister18Bit']
            step = self.buffer_pars['dacStepSize18Bit']
            dis = abs(stop - start)
            n_of_steps = int(round(dis / step))
        except ZeroDivisionError:
            n_of_steps = 0
        except Exception as e:
            logging.error('following error occurred while calculating the number of steps:' + str(e))
            n_of_steps = 0
        return abs(n_of_steps)  # sign should always be in the stepsize

    def dac_step_size_set(self, step_volt):
        """ if the stepsize is set, adjust the number of steps to keep start and stop constant"""
        step_18bit = VCon.get_18bit_stepsize(step_volt)
        self.display_step_size(step_18bit)
        self.recalc_n_of_steps_stop()

    def display_step_size(self, step_18bit):
        """ stores the stepSize to the working dictionary and displays them """
        self.buffer_pars['dacStepSize18Bit'] = step_18bit
        step_volt = VCon.get_stepsize_in_volt_from_18bit(step_18bit)
        self.label_dacStepSizeV_set.setText(str(step_volt) + ' | ' + str(format(step_18bit, '018b'))
                                            + ' | ' + str(step_18bit))
        self.label_kepco_step.setText(str(round(step_volt * 50, 2)))

    def n_of_steps_set(self, steps):
        """ displays the number of steps that where set and recalculates the stepSize
         in order to keep start and stop more or less constant """
        self.display_n_of_steps(steps)
        self.recalc_step_stop()

    def display_n_of_steps(self, steps):
        """ write the number of steps to the working dictionary and display them """
        steps = int(round(steps))
        self.label_nOfSteps_set.setText(str(steps))
        self.buffer_pars['nOfSteps'] = steps

    def n_of_scans_set(self, val):
        """ write the number of scans to the working dictionary and display them """
        self.label_nOfScans_set.setText(str(val))
        self.buffer_pars['nOfScans'] = val

    def invert_scan_set(self, state):
        """ wirte to the working dictionars and set the label """
        boolstate = state == 2
        self.label_invertScan_set.setText(str(boolstate))
        self.buffer_pars['invertScan'] = boolstate

    def post_acc_offset_volt_control_set(self, val):
        """ write to the working dictionars and set the label """
        if val != 'Kepco':
            status = self.scan_ctrl_win.main.power_supply_status_request(val)
            if status is not None:
                val = str(status.get('readBackVolt'))
        self.label_postAccOffsetVoltControl_set.setText(val)
        self.buffer_pars['postAccOffsetVoltControl'] = self.comboBox_postAccOffsetVoltControl.currentIndex()

    def post_acc_offset_volt(self, val):
        """ write to the working dictionars and set the label """
        self.label_postAccOffsetVolt_set.setText(str(val))
        self.buffer_pars['postAccOffsetVolt'] = val

    def active_pmt_list_set(self, lis):
        """ write to the working dictionars and set the label """
        if type(lis) == str:
            try:
                lis = str('[' + lis + ']')
                lis = ast.literal_eval(lis)
            except:
                lis = []
        self.label_activePmtList_set.setText(str(lis))
        self.buffer_pars['activePmtList'] = lis

    def col_dir_true_set(self, state):
        """ write to the working dictionars and set the label """
        boolstate = state == 2
        if boolstate:
            display = 'colinear'
        else:
            display = 'anti colinear'
        self.label_colDirTrue_set.setText(display)
        self.buffer_pars['colDirTrue'] = boolstate
        # pass

    def wait_after_reset_mu_sec_set(self, time_mu_s):
        """ write to the working dictionars and set the label """
        time_25ns = int(round(time_mu_s / 25 * (10 ** 3)))
        self.buffer_pars['waitAfterReset25nsTicks'] = time_25ns
        setval = time_25ns * 25 * (10 ** -3)
        self.label_waitAfterReset_muS_set.setText(str(round(setval, 3)))

    def wait_for_kepco_mu_sec(self, time_mu_s):
        """ write to the working dictionars and set the label """
        time_25ns = int(round(time_mu_s / 25 * (10 ** 3)))
        self.buffer_pars['waitForKepco25nsTicks'] = time_25ns
        setval = time_25ns * 25 * (10 ** -3)
        self.label_waitForKepco_muS_set.setText(str(round(setval, 3)))

    def set_voltage(self):
        """ this will connect to the corresponding Heinzinger and set the voltage.
         this helps the user to enter the voltage a while before the scan."""
        power_supply = self.comboBox_postAccOffsetVoltControl.currentText()
        if power_supply != 'Kepco':
            volt = self.buffer_pars['postAccOffsetVolt']
            self.scan_ctrl_win.main.set_power_supply_voltage(power_supply, volt)
            setvoltui = SetVoltageUi(power_supply, volt, self.scan_ctrl_win.main)
            self.label_postAccOffsetVoltControl_set.setText(setvoltui.readback)

    def cancel(self):
        """ closes the window without further actions """
        self.close()

    def confirm(self):
        """ closes the window and merges the buffer_scan_dict
        of the scan control window with the working dictionary """
        self.scan_ctrl_win.buffer_scan_dict['track' + str(self.track_number)] = \
            SdOp.merge_dicts(self.scan_ctrl_win.buffer_scan_dict['track' + str(self.track_number)],
                             self.buffer_pars)
        self.close()

    def reset_to_default(self):
        """ will reset all spinboxes to the default value which was passed in the beginning. """
        self.set_labels_by_dict(self.default_track_dict)

    def closeEvent(self, event):
        """
        will remove the given track window from the dictionary in scan_ctrl_win
        """
        self.scan_ctrl_win.track_win_closed(self.track_number)
