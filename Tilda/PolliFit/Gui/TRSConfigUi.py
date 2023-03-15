"""
Created on 05.03.2022

@author: Patrick Mueller
"""


from copy import deepcopy
from PyQt5 import QtWidgets, QtCore
from Tilda.PolliFit.Gui.Ui_TRSConfig import Ui_TRSConfig


class TRSConfigUi(QtWidgets.QWidget, Ui_TRSConfig):
    gate_signal = QtCore.pyqtSignal()

    def __init__(self, softw_gates):
        super(TRSConfigUi, self).__init__()
        self.setupUi(self)
        self.widgets = []
        self.old_softw_gates = None
        self.softw_gates = None
        self.set_softw_gates(softw_gates)

        self.b_ok.clicked.connect(self.close)
        self.b_cancel.clicked.connect(self.revert_and_close)

    def _gen_input_box(self, val):
        w = QtWidgets.QDoubleSpinBox(self)
        w.setDecimals(2)
        w.setMinimum(0)
        w.setMaximum(10000)
        w.setSingleStep(0.01)
        w.setValue(val)
        return w

    def _gen_input_frame(self, t_min, t_max):
        self.widgets[-1].append([self._gen_input_box(t_min), self._gen_input_box(t_max)])
        hor_input = QtWidgets.QHBoxLayout(self)
        hor_input.addWidget(self.widgets[-1][-1][0])
        hor_input.addWidget(self.widgets[-1][-1][1])
        hor_input.setContentsMargins(3, 3, 3, 3)
        frame_input = QtWidgets.QFrame(self)
        frame_input.setFrameStyle(QtWidgets.QFrame.Shape.Panel | QtWidgets.QFrame.Shadow.Sunken)
        frame_input.setLayout(hor_input)
        return frame_input

    def set_softw_gates(self, softw_gates):
        self.old_softw_gates = deepcopy(softw_gates)
        self.softw_gates = softw_gates
        self.widgets = []
        for i, t in enumerate(self.softw_gates):
            l_track = QtWidgets.QLabel('track{}'.format(i))
            self.grid_trs.addWidget(l_track, i + 1, 0)
            self.widgets.append([])
            for j, s in enumerate(t):
                if i == 0:
                    l_scaler = QtWidgets.QLabel('scaler{}'.format(j))
                    l_scaler.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    self.grid_trs.addWidget(l_scaler, i, j + 1)
                self.grid_trs.addWidget(self._gen_input_frame(s[2], s[3]), i + 1, j + 1)
                # noinspection PyUnresolvedReferences
                self.widgets[-1][-1][0].editingFinished.connect(
                    lambda _i=i, _j=j: self.set_value(_i, _j, 2, self.widgets[_i][_j][0].value()))
                # noinspection PyUnresolvedReferences
                self.widgets[-1][-1][1].editingFinished.connect(
                    lambda _i=i, _j=j: self.set_value(_i, _j, 3, self.widgets[_i][_j][1].value()))
        for row in range(self.grid_trs.rowCount()):
            self.grid_trs.setRowStretch(row, 1)
        self.grid_trs.setRowStretch(0, 0)
        for col in range(self.grid_trs.columnCount()):
            self.grid_trs.setColumnStretch(col, 1)
        self.grid_trs.setColumnStretch(0, 0)

    def set_value(self, i, j, k, val):
        if self.softw_gates[i][j][k] == val:
            return
        self.softw_gates[i][j][k] = val
        self.gate_signal.emit()

    def revert_and_close(self):
        self.softw_gates = self.old_softw_gates
        self.gate_signal.emit()
        self.close()
