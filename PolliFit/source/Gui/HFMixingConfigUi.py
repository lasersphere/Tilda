"""
Created on 16.07.2022

@author: Patrick Mueller
"""

from copy import deepcopy
from PyQt5 import QtWidgets, QtCore
from Gui.Ui_HFMixingConfig import Ui_HFMixingConfig


def set_table(tab, t=None):
    for i in range(tab.rowCount()):
        for j in range(tab.columnCount()):
            if t is None:
                val = str(0.)
            else:
                try:
                    val = str(t[i][j])
                except IndexError:
                    val = str(0.)
            if tab.item(i, j) is None:
                w = QtWidgets.QTableWidgetItem(val)
                tab.setItem(i, j, w)
            else:
                tab.item(i, j).setText(val)


def get_table(tab):
    return [[0. if tab.item(i, j) is None else float(tab.item(i, j).text()) for j in range(tab.columnCount())]
            for i in range(tab.rowCount())]


class HFMixingConfigUi(QtWidgets.QWidget, Ui_HFMixingConfig):
    close_signal = QtCore.pyqtSignal()

    def __init__(self, iso, config=None):
        super(HFMixingConfigUi, self).__init__()
        self.setupUi(self)
        self.iso = iso

        if config is None:
            self.config = dict(enabled_l=False, enabled_u=False,
                               Jl=[self.iso.Jl, ], Ju=[self.iso.Ju, ], Tl=[[1.]], Tu=[[1.]])
        else:
            self.config = deepcopy(config)
        self.j_g = [self.iso.Jl]
        self.j_e = [self.iso.Ju]
        self.t_g = [[1.]]
        self.t_e = [[1.]]

        self.check_g.stateChanged.connect(self.toggle_g)
        self.line_j_g.editingFinished.connect(self.apply_line_j_g)
        self.b_diagonal_g.clicked.connect(self.set_diag_g)
        self.b_apply_g.clicked.connect(self.apply_g)
        self.b_reset_g.clicked.connect(self.load_config_g)

        self.check_e.stateChanged.connect(self.toggle_e)
        self.line_j_e.editingFinished.connect(self.apply_line_j_e)
        self.b_diagonal_e.clicked.connect(self.set_diag_e)
        self.b_apply_e.clicked.connect(self.apply_e)
        self.b_reset_e.clicked.connect(self.load_config_e)

        self.b_ok.clicked.connect(self.ok_and_close)
        self.b_apply.clicked.connect(self.apply)
        self.b_reset.clicked.connect(self.load_config)
        self.b_cancel.clicked.connect(self.reset_and_close)

        self.tab_g.cellChanged[int, int].connect(self.cast_cell_g)
        self.tab_e.cellChanged[int, int].connect(self.cast_cell_e)

        self.load_config()
    
    def cast_cell_g(self, i, j):
        self.tab_g.blockSignals(True)
        try:
            val = float(self.tab_g.item(i, j).text())
        except ValueError:
            val = self.t_g[i][j]
        self.tab_g.item(i, j).setText(str(val))
        self.t_g[i][j] = val
        if i != j:
            self.tab_g.item(j, i).setText(str(val))
            self.t_g[j][i] = val
        self.tab_g.blockSignals(False)
    
    def cast_cell_e(self, i, j):
        self.tab_e.blockSignals(True)
        try:
            val = float(self.tab_e.item(i, j).text())
        except ValueError:
            val = self.t_e[i][j]
        self.tab_e.item(i, j).setText(str(val))
        self.t_e[i][j] = val
        if i != j:
            self.tab_e.item(j, i).setText(str(val))
            self.t_e[j][i] = val
        self.tab_e.blockSignals(False)

    def toggle_g(self):
        self.hor_widget_g.setEnabled(self.check_g.isChecked())

    def toggle_e(self):
        self.hor_widget_e.setEnabled(self.check_e.isChecked())

    def load_config_g(self):
        self.tab_g.blockSignals(True)
        self.d_g.setValue(self.iso.Jl)
        self.j_g = [self.iso.Jl]
        self.line_j_g.setText(str(self.config['Jl']))
        self.apply_line_j_g()
        set_table(self.tab_g, self.config['Tl'])
        self.t_g = get_table(self.tab_g)
        if self.config['enabled_l']:
            self.check_e.setChecked(True)
        self.tab_g.blockSignals(False)

    def load_config_e(self):
        self.tab_e.blockSignals(True)
        self.d_e.setValue(self.iso.Ju)
        self.j_e = [self.iso.Ju]
        self.line_j_e.setText(str(self.config['Ju']))
        self.apply_line_j_e()
        set_table(self.tab_e, self.config['Tu'])
        self.t_e = get_table(self.tab_e)
        if self.config['enabled_u']:
            self.check_e.setChecked(True)
        self.tab_e.blockSignals(False)

    def load_config(self):
        self.load_config_g()
        self.load_config_e()

    def set_diag_g(self):
        self.tab_g.blockSignals(True)
        t = [[0. if i != j else float(self.tab_g.item(i, j).text())
              for j in range(self.tab_g.columnCount())] for i in range(self.tab_g.rowCount())]
        set_table(self.tab_g, t)
        self.tab_g.blockSignals(False)

    def set_diag_e(self):
        self.tab_e.blockSignals(True)
        t = [[0. if i != j else float(self.tab_e.item(i, j).text())
              for j in range(self.tab_e.columnCount())] for i in range(self.tab_e.rowCount())]
        set_table(self.tab_e, t)
        self.tab_e.blockSignals(False)

    def apply_line_j_g(self):
        self.tab_g.blockSignals(True)
        val = self.line_j_g.text()
        try:
            val = [float(j) for j in eval(val)]
            if self.d_g.value() not in val:
                raise ValueError()
        except (ValueError, TypeError, SyntaxError, NameError):
            val = self.j_g
        self.j_g = val
        self.line_j_g.setText(str(val))
        self.tab_g.setRowCount(len(val))
        self.tab_g.setColumnCount(len(val))
        t = get_table(self.tab_g)
        set_table(self.tab_g, t)
        self.t_g = get_table(self.tab_g)
        self.tab_g.blockSignals(False)

    def apply_line_j_e(self):
        self.tab_e.blockSignals(True)
        val = self.line_j_e.text()
        try:
            val = [float(j) for j in eval(val)]
            if self.d_e.value() not in val:
                raise ValueError()
        except (ValueError, TypeError, SyntaxError, NameError):
            val = self.j_e
        self.j_e = val
        self.line_j_e.setText(str(val))
        self.tab_e.setRowCount(len(val))
        self.tab_e.setColumnCount(len(val))
        t = get_table(self.tab_e)
        set_table(self.tab_e, t)
        self.t_e = get_table(self.tab_e)
        self.tab_e.blockSignals(False)
    
    def apply_g(self):
        self.config['enabled_l'] = self.check_g.isChecked()
        self.config['Jl'] = eval(self.line_j_g.text())
        self.config['Tl'] = get_table(self.tab_g)

    def apply_e(self):
        self.config['enabled_u'] = self.check_e.isChecked()
        self.config['Ju'] = eval(self.line_j_e.text())
        self.config['Tu'] = get_table(self.tab_e)

    def ok_and_close(self):
        self.apply()
        self.close_signal.emit()
        self.close()

    def apply(self):
        self.apply_g()
        self.apply_e()

    def reset_and_close(self):
        self.close_signal.emit()
        self.close()
