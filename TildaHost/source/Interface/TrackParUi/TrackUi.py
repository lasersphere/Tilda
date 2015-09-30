"""

Created on '29.09.2015'

@author:'simkaufm'

"""
from PyQt5 import QtWidgets, QtCore


from Interface.TrackParUi.Ui_TrackPar import Ui_MainWindowTrackPars

class TrackUi(QtWidgets.QMainWindow, Ui_MainWindowTrackPars):
    def __init__(self):
        super(TrackUi, self).__init__()

        self.buffer_pars = {}

        self.setupUi(self)
        self.show()

        self.doubleSpinBox_dwellTime_ms.valueChanged.connect(self.dwelltime_set)

        self.doubleSpinBox_dacStartV.valueChanged.connect(self.dac_start_v_set)
        self.doubleSpinBox_dacStopV.valueChanged.connect(self.dac_stop_v_set)
        self.doubleSpinBox_dacStepSizeV.valueChanged.connect(self.dac_step_size_set)
        self.spinBox_nOfSteps.valueChanged.connect(self.n_of_steps_set)
        self.spinBox_nOfScans.valueChanged.connect(self.n_of_scans_set)
        self.checkBox_invertScan.clicked.connect(self.invert_scan_set)

        self.spinBox_postAccOffsetVoltControl.valueChanged.connect(self.post_acc_offset_volt_control_set)
        self.doubleSpinBox_postAccOffsetVolt.valueChanged.connect(self.post_acc_offset_volt)

        self.lineEdit_activePmtList.editingFinished.connect(self.active_pmt_list_set)
        self.checkBox_colDirTrue.clicked.connect(self.col_dir_true_set)

        self.pushButton_advancedSettings.clicked.connect(self.adv_set)
        self.pushButton_cancel.clicked.connect(self.cancel)
        self.pushButton_confirm.clicked.connect(self.confirm)

    def dwelltime_set(self, val):
        self.buffer_pars['dwellTime10ns'] = val * (10 ** 5)
        self.label_dwellTime_ms_2.setText(str(val))

    def dac_start_v_set(self):
        pass

    def dac_stop_v_set(self):
        pass

    def dac_step_size_set(self):
        pass

    def n_of_steps_set(self):
        pass

    def n_of_scans_set(self):
        pass

    def invert_scan_set(self):
        pass

    def post_acc_offset_volt_control_set(self):
        pass

    def post_acc_offset_volt(self):
        pass

    def active_pmt_list_set(self):
        pass

    def col_dir_true_set(self):
        pass

    def adv_set(self):
        pass

    def cancel(self):
        pass

    def confirm(self):
        pass