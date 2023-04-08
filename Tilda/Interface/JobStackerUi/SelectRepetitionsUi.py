"""

Created on '14.02.2019'

@author:'fsommer'

"""

import sys

from PyQt5 import QtWidgets

from Tilda.Interface.JobStackerUi.Ui_SelectRepetitions import Ui_Dialog_JobRepetitions


class SelectRepetitionsUi(QtWidgets.QDialog, Ui_Dialog_JobRepetitions):

    def __init__(self, job_stacker):
        super(SelectRepetitionsUi, self).__init__()

        self.setupUi(self)
        self.job_stacker_ui = job_stacker

        self.spinBox_number_reps.setMinimum(1)
        self.spinBox_number_reps.setMaximum(99)  # could be adapted if really wanted^^

        self.show()

    ''' window related '''

    def accept(self):
        """ overwrite the accept event """
        self.job_stacker_ui.repetition_ctrl_closed(self.spinBox_number_reps.value())
        self.close()

    def reject(self):
        """ overwrite the reject event """
        self.job_stacker_ui.repetition_ctrl_closed(None)
        self.close()

    def close_and_confirm(self):
        """
        close the window
        send spinBox value to mother
        """
        self.job_stacker_ui.repetition_ctrl_closed(self.spinBox_number_reps.value())
        self.close()

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        self.job_stacker_ui.repetition_ctrl_closed(None)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = SelectRepetitionsUi(None)

    app.exec_()

