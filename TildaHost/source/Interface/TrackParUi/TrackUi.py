"""

Created on '29.09.2015'

@author:'simkaufm'

"""
from PyQt5 import QtWidgets, QtCore
import ast
import sys
from copy import deepcopy

from Interface.TrackParUi.Ui_TrackPar import Ui_MainWindowTrackPars
import Service.Scan.draftScanParameters as Dft
import Service.Formating as form
import Service.VoltageConversions.VoltageConversions as VCon


class TrackUi(QtWidgets.QMainWindow, Ui_MainWindowTrackPars):
    def __init__(self, main, track_number, default_track_dict):
        super(TrackUi, self).__init__()

        self.default_track_dict = deepcopy(default_track_dict)
        self.buffer_pars = default_track_dict
        self.buffer_pars['dacStopRegister18Bit'] = self.calc_dac_stop_18bit()
        self.default_track_dict['dacStopRegister18Bit'] = self.calc_dac_stop_18bit()

        self.main = main
        self.track_number = track_number

        self.setupUi(self)
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

        """post accerleration controls:"""
        self.comboBox_postAccOffsetVoltControl.currentTextChanged.connect(self.post_acc_offset_volt_control_set)
        self.doubleSpinBox_postAccOffsetVolt.valueChanged.connect(self.post_acc_offset_volt)

        """Scaler selection:"""
        self.lineEdit_activePmtList.textChanged.connect(self.active_pmt_list_set)

        """colinear/anticolinear"""
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

    def wait_after_reset_mu_sec_set(self, time_mu_s):
        time_25ns = int(round(time_mu_s / 25 * (10 ** 3)))
        self.buffer_pars['waitAfterReset25nsTicks'] = time_25ns
        setval = time_25ns * 25 * (10 ** -3)
        self.label_waitAfterReset_muS_set.setText(str(round(setval, 3)))

    def wait_for_kepco_mu_sec(self, time_mu_s):
        time_25ns = int(round(time_mu_s / 25 * (10 ** 3)))
        self.buffer_pars['waitForKepco25nsTicks'] = time_25ns
        setval = time_25ns * 25 * (10 ** -3)
        self.label_waitForKepco_muS_set.setText(str(round(setval, 3)))

    def set_labels_by_dict(self, track_dict):
        # function to set all labels by setting the values to the corresponding spinboxes etc.
        self.doubleSpinBox_dwellTime_ms.setValue(track_dict['dwellTime10ns'] * (10 ** -5))
        self.doubleSpinBox_dacStartV.setValue(VCon.get_voltage_from_18bit(track_dict['dacStartRegister18Bit']))
        self.doubleSpinBox_dacStopV.setValue(VCon.get_voltage_from_18bit(track_dict['dacStopRegister18Bit']))
        self.spinBox_nOfSteps.setValue(track_dict['nOfSteps'])
        self.doubleSpinBox_dacStepSizeV.setValue(VCon.get_stepsize_in_volt_from_18bit(track_dict['dacStepSize18Bit']))
        self.spinBox_nOfScans.setValue(track_dict['nOfScans'])
        self.checkBox_invertScan.setChecked(track_dict['invertScan'])
        self.comboBox_postAccOffsetVoltControl.setCurrentIndex(int(track_dict['postAccOffsetVoltControl']))
        self.doubleSpinBox_postAccOffsetVolt.setValue(track_dict['postAccOffsetVolt'])
        self.lineEdit_activePmtList.setText(str(track_dict['activePmtList']))
        self.checkBox_colDirTrue.setChecked(track_dict['colDirTrue'])
        self.doubleSpinBox_waitAfterReset_muS.setValue(track_dict['waitAfterReset25nsTicks'] * 25 * (10 ** -3))
        self.doubleSpinBox_waitForKepco_muS.setValue(track_dict['waitForKepco25nsTicks'] * 25 * (10 ** -3))

    def dwelltime_set(self, val):
        self.buffer_pars['dwellTime10ns'] = val * (10 ** 5)  # convert to units of 10ns
        self.label_dwellTime_ms_2.setText(str(val))

    def dac_start_v_set(self, start_volt):
        start_18bit = VCon.get_18bit_from_voltage(start_volt)
        start_volt = VCon.get_voltage_from_18bit(start_18bit)
        self.buffer_pars['dacStartRegister18Bit'] = start_18bit
        self.label_dacStartV_set.setText(str(start_volt) + ' | ' + str(format(start_18bit, '018b'))
                                         + ' | ' + str(start_18bit))
        self.label_kepco_start.setText(str(round(start_volt * 50, 2)))
        self.recalc_step_stop()

    def dac_stop_v_set(self, stop_volt):
        self.buffer_pars['dacStopRegister18Bit'] = VCon.get_18bit_from_voltage(stop_volt)
        self.recalc_step_stop()

    def display_stop(self, stop_18bit):
        # self.buffer_pars['dacStopRegister18Bit'] = stop_18bit
        setval = VCon.get_voltage_from_18bit(stop_18bit)
        self.label_dacStopV_set.setText(str(setval) + ' | ' + str(format(stop_18bit, '018b'))
                                        + ' | ' + str(stop_18bit))
        self.label_kepco_stop.setText(str(round(setval * 50, 2)))

    def recalc_step_stop(self):
        # start and stop should be more or less constant therefore in most cases the stepsize must be adjusted.
        # after adjusting the stop voltage is fine tuned to the next possible value.
        self.display_step_size(self.calc_step_size())
        self.display_stop(self.calc_dac_stop_18bit())

    def recalc_n_of_steps_stop(self):
        self.display_n_of_steps(self.calc_n_of_steps())
        self.display_stop(self.calc_dac_stop_18bit())

    def calc_dac_stop_18bit(self):
        start = self.buffer_pars['dacStartRegister18Bit']
        step = self.buffer_pars['dacStepSize18Bit']
        steps = self.buffer_pars['nOfSteps']
        stop = start + step * steps
        return stop

    def calc_step_size(self):
        start = self.buffer_pars['dacStartRegister18Bit']
        stop = self.buffer_pars['dacStopRegister18Bit']
        steps = self.buffer_pars['nOfSteps']
        dis = abs(stop - start)
        try:
            stepsize_18bit = int(round(dis / steps))
        except ZeroDivisionError:
            stepsize_18bit = 0
        return stepsize_18bit

    def calc_n_of_steps(self):
        start = self.buffer_pars['dacStartRegister18Bit']
        stop = self.buffer_pars['dacStopRegister18Bit']
        step = self.buffer_pars['dacStepSize18Bit']
        dis = abs(stop - start)
        try:
            n_of_steps = int(round(dis / step))
        except ZeroDivisionError:
            n_of_steps = 0
        return n_of_steps

    def dac_step_size_set(self, step_volt):
        step_18bit = VCon.get_18bit_stepsize(step_volt)
        self.display_step_size(step_18bit)
        self.recalc_n_of_steps_stop()

    def display_step_size(self, step_18bit):
        self.buffer_pars['dacStepSize18Bit'] = step_18bit
        step_volt = VCon.get_stepsize_in_volt_from_18bit(step_18bit)
        self.label_dacStepSizeV_set.setText(str(step_volt) + ' | ' + str(format(step_18bit, '018b'))
                                            + ' | ' + str(step_18bit))
        self.label_kepco_step.setText(str(round(step_volt * 50, 2)))

    def n_of_steps_set(self, steps):
        self.display_n_of_steps(steps)
        self.recalc_step_stop()

    def display_n_of_steps(self, steps):
        steps = int(round(steps))
        self.label_nOfSteps_set.setText(str(steps))
        self.buffer_pars['nOfSteps'] = steps

    def n_of_scans_set(self, val):
        self.label_nOfScans_set.setText(str(val))
        self.buffer_pars['nOfScans'] = val

    def invert_scan_set(self, state):
        boolstate = state == 2
        self.label_invertScan_set.setText(str(boolstate))
        self.buffer_pars['invertScan'] = boolstate
        # pass

    def post_acc_offset_volt_control_set(self, val):
        self.label_postAccOffsetVoltControl_set.setText(str(val))
        self.buffer_pars['postAccOffsetVoltControl'] = self.comboBox_postAccOffsetVoltControl.currentIndex()

    def post_acc_offset_volt(self, val):
        self.label_postAccOffsetVolt_set.setText(str(val))
        self.buffer_pars['postAccOffsetVolt'] = val
        # pass

    def active_pmt_list_set(self, lis):
        if type(lis) == str:
            try:
                lis = ast.literal_eval(lis)
            except:
                lis = []
        self.label_activePmtList_set.setText(str(lis))
        self.buffer_pars['activePmtList'] = lis
        # pass

    def col_dir_true_set(self, state):
        boolstate = state == 2
        if boolstate:
            display = 'colinear'
        else:
            display = 'anti colinear'
        self.label_colDirTrue_set.setText(display)
        self.buffer_pars['colDirTrue'] = boolstate
        # pass

    def set_voltage(self):
        pass

    def cancel(self):
        self.destroy()

    def confirm(self):
        print(self.buffer_pars)
        # try:
        # self.main.scanpars[self.track_number] = self.buffer_pars
        # except IndexError:
        # self.main.scanpars.append(self.buffer_pars)
        self.destroy()
        pass

    def reset_to_default(self):
        self.set_labels_by_dict(self.default_track_dict)

    def nothing(self):
        pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dicti = Dft.draftTrackPars
    dicti['dwellTime10ns'] = 2000000
    ui = TrackUi(None, 0, dicti)
    app.exec_()