"""

Created on '25.11.2015'

@author:'simkaufm'

Description: Dialog to control the Simple Counter while this is running.
"""

from PyQt5 import QtWidgets

from Tilda.Interface.MQTTSimpleCounter.Ui_Simp_Count_Dial import Ui_Dialog_simpleCounterControl
import ast


class SimpleCounterDialogUi(QtWidgets.QDialog, Ui_Dialog_simpleCounterControl):
    def __init__(self):
        super(SimpleCounterDialogUi, self).__init__()

        self.start = False
        self.act_pmts = []
        self.datapoints = 0

        self.setupUi(self)

        self.buttonBox.accepted.connect(self.ok)
        self.buttonBox.rejected.connect(self.close)
        self.lineEdit_act_pmts.textChanged.connect(self.set_act_pmts)
        self.spinBox_plotpoints.valueChanged.connect(self.set_datapoints)

        ''' default vals: '''
        self.lineEdit_act_pmts.setText('0, 1, 2, 3')
        self.spinBox_plotpoints.setValue(600)

        self.exec_()

    def ok(self):
        self.start = True

    def cancel(self):
        self.start = False

    def set_act_pmts(self, lis_str):
        """ convert input line to list and set the label """
        if type(lis_str) == str:
            try:
                lis_str = str('[' + lis_str + ']')
                lis = ast.literal_eval(lis_str)
            except Exception as e:
                lis = []
        self.label_act_pmts_set.setText(str(lis))
        self.act_pmts = lis

    def set_datapoints(self, val):
        self.datapoints = val
        self.label_plotpoints_set.setText(str(val))


