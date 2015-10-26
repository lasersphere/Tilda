"""

Created on '29.09.2015'

@author:'simkaufm'

"""
from PyQt5 import QtWidgets, QtCore
import ast
import sys

from Interface.TrackParUi.Ui_TrackPar import Ui_MainWindowTrackPars
import Service.Scan.draftScanParameters as Dft
import Service.Formating as form


class TrackUi(QtWidgets.QMainWindow, Ui_MainWindowTrackPars):
    def __init__(self, main, track_number, default_track_dict):
        super(TrackUi, self).__init__()

        self.default_track_dict = default_track_dict
        self.buffer_pars = default_track_dict

        self.main = main
        self.track_number = track_number

        self.setupUi(self)
        self.show()

        self.doubleSpinBox_dwellTime_ms.valueChanged.connect(self.dwelltime_set)

        self.doubleSpinBox_dacStartV.valueChanged.connect(self.dac_start_v_set)
        self.doubleSpinBox_dacStopV.valueChanged.connect(self.dac_stop_v_set)
        self.doubleSpinBox_dacStepSizeV.valueChanged.connect(self.dac_step_size_set)
        self.spinBox_nOfSteps.valueChanged.connect(self.n_of_steps_set)
        self.spinBox_nOfScans.valueChanged.connect(self.n_of_scans_set)
        self.checkBox_invertScan.clicked.connect(self.invert_scan_set)

        self.comboBox_postAccOffsetVoltControl.currentTextChanged.connect(self.post_acc_offset_volt_control_set)
        self.doubleSpinBox_postAccOffsetVolt.valueChanged.connect(self.post_acc_offset_volt)

        self.lineEdit_activePmtList.textChanged.connect(self.active_pmt_list_set)
        self.checkBox_colDirTrue.clicked.connect(self.col_dir_true_set)

        self.pushButton_cancel.clicked.connect(self.cancel)
        self.pushButton_confirm.clicked.connect(self.confirm)
        self.pushButtonResetToDefault.clicked.connect(self.reset_to_default)
        self.pushButton_postAccOffsetVolt.clicked.connect(self.set_voltage)
        self.set_labels_by_dict(self.default_track_dict)

    def set_labels_by_dict(self, track_dict):
        self.dwelltime_set(track_dict['dwellTime10ns'] * (10 ** -5))
        self.dac_start_v_set(form.getVoltageFrom18Bit(track_dict['dacStartRegister18Bit']))
        self.dac_stop_v_set(form.getVoltageFrom18Bit(
            track_dict['dacStartRegister18Bit'] + track_dict['dacStepSize18Bit'] * track_dict['nOfSteps']))
        self.dac_step_size_set(form.getVoltageFrom18Bit(track_dict['dacStepSize18Bit']) - form.getVoltageFrom18Bit(0))
        self.n_of_steps_set(track_dict['nOfSteps'])
        self.n_of_scans_set(track_dict['nOfScans'])
        self.invert_scan_set(track_dict['invertScan'])
        # self.post_acc_offset_volt_control_set(track_dict['postAccOffsetVoltControl'])
        self.post_acc_offset_volt(track_dict['postAccOffsetVolt'])
        # self.active_pmt_list_set(track_dict['activePmtList'])
        # self.col_dir_true_set(track_dict['colDirTrue'])

    def dwelltime_set(self, val):
        self.buffer_pars['dwellTime10ns'] = val * (10 ** 5)  # convert to units of 10ns
        self.label_dwellTime_ms_2.setText(str(val))

    def dac_start_v_set(self, val):
        bit = form.get18BitInputForVoltage(val)
        setval = form.getVoltageFrom18Bit(bit)
        self.buffer_pars['dacStartRegister18Bit'] = bit
        self.label_dacStartV_set.setText(str(setval) + ' | ' + str(format(bit, '018b'))
                                         + ' | ' + str(bit))
        self.label_kepco_start.setText(str(round(setval * 50, 2)))

    def dac_stop_v_set(self, val):
        bit = form.get18BitInputForVoltage(val)
        setval = form.getVoltageFrom18Bit(bit)
        self.label_dacStopV_set.setText(str(setval) + ' | ' + str(format(bit, '018b'))
                                         + ' | ' + str(bit))
        self.label_kepco_stop.setText(str(round(setval * 50, 2)))
        try:
            if self.buffer_pars['dacStepSize18Bit'] and self.buffer_pars['dacStartRegister18Bit']:
                startv = self.buffer_pars['dacStartRegister18Bit']
                div = abs(val - form.getVoltageFrom18Bit(startv))
                stepsize = form.getVoltageFrom18Bit(self.buffer_pars['dacStepSize18Bit']) + 10
                nofsteps = div / stepsize
                self.n_of_steps_set(nofsteps)
        except KeyError:
            pass

    def dac_step_size_set(self, val):
        bit = form.get18BitStepSize(val)
        setval = form.getVoltageFrom18Bit(bit) + 10
        self.label_dacStepSizeV_set.setText(str(setval) + ' | ' + str(format(bit, '018b'))
                                            + ' | ' + str(bit))
        self.label_kepco_step.setText(str(round(setval * 50, 2)))
        self.buffer_pars['dacStepSize18Bit'] = bit

    def n_of_steps_set(self, val):
        self.label_nOfSteps_set.setText(str(val))
        self.buffer_pars['nOfSteps'] = val

    def n_of_scans_set(self, val):
        self.label_nOfScans_set.setText(str(val))
        self.buffer_pars['nOfScans'] = val

    def invert_scan_set(self, bool):
        self.label_invertScan_set.setText(str(bool))
        self.buffer_pars['invertScan'] = bool
        # pass

    def post_acc_offset_volt_control_set(self, val):
        self.label_postAccOffsetVoltControl_set.setText(str(val))
        self.buffer_pars['postAccOffsetVoltControl'] = val
        # pass

    def post_acc_offset_volt(self, val):
        self.label_postAccOffsetVolt_set.setText(str(val))
        self.buffer_pars['postAccOffsetVolt'] = val
        # pass

    def active_pmt_list_set(self, val):
        strlist = str('[' + val + ']')
        try:
            lis = ast.literal_eval(strlist)
        except:
            lis = []
        self.label_activePmtList_set.setText(str(lis))
        self.buffer_pars['activePmtList'] = lis
        # pass

    def col_dir_true_set(self, bool):
        self.label_colDirTrue_set.setText(str(bool))
        self.buffer_pars['colDirTrue'] = bool
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
        pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dicti = Dft.draftTrackPars
    dicti['dwellTime10ns'] = 2000000
    ui = TrackUi(None, 0, dicti)
    app.exec_()