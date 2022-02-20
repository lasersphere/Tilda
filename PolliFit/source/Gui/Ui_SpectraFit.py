# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Users\Patrick M�ller\Documents\Python Projects\Tilda\PolliFit\source\Gui\Ui_SpectraFit.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SpectraFit(object):
    def setupUi(self, SpectraFit):
        SpectraFit.setObjectName("SpectraFit")
        SpectraFit.resize(900, 500)
        self.horizontalLayout = QtWidgets.QHBoxLayout(SpectraFit)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.vert_files = QtWidgets.QVBoxLayout()
        self.vert_files.setObjectName("vert_files")
        self.hor_run_select = QtWidgets.QHBoxLayout()
        self.hor_run_select.setObjectName("hor_run_select")
        self.c_run = QtWidgets.QComboBox(SpectraFit)
        self.c_run.setObjectName("c_run")
        self.hor_run_select.addWidget(self.c_run)
        self.c_iso = QtWidgets.QComboBox(SpectraFit)
        self.c_iso.setObjectName("c_iso")
        self.hor_run_select.addWidget(self.c_iso)
        self.c_line = QtWidgets.QComboBox(SpectraFit)
        self.c_line.setEnabled(False)
        self.c_line.setObjectName("c_line")
        self.hor_run_select.addWidget(self.c_line)
        self.vert_files.addLayout(self.hor_run_select)
        self.list_files = QtWidgets.QListWidget(SpectraFit)
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list_files.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.list_files.setObjectName("list_files")
        self.vert_files.addWidget(self.list_files)
        self.hor_file_options = QtWidgets.QHBoxLayout()
        self.hor_file_options.setObjectName("hor_file_options")
        self.b_select_all = QtWidgets.QPushButton(SpectraFit)
        self.b_select_all.setMaximumSize(QtCore.QSize(50, 16777215))
        self.b_select_all.setObjectName("b_select_all")
        self.hor_file_options.addWidget(self.b_select_all)
        self.b_select_col = QtWidgets.QPushButton(SpectraFit)
        self.b_select_col.setMaximumSize(QtCore.QSize(50, 16777215))
        self.b_select_col.setObjectName("b_select_col")
        self.hor_file_options.addWidget(self.b_select_col)
        self.b_select_acol = QtWidgets.QPushButton(SpectraFit)
        self.b_select_acol.setMaximumSize(QtCore.QSize(50, 16777215))
        self.b_select_acol.setObjectName("b_select_acol")
        self.hor_file_options.addWidget(self.b_select_acol)
        self.b_select_favorites = QtWidgets.QPushButton(SpectraFit)
        self.b_select_favorites.setEnabled(False)
        self.b_select_favorites.setMaximumSize(QtCore.QSize(50, 16777215))
        self.b_select_favorites.setObjectName("b_select_favorites")
        self.hor_file_options.addWidget(self.b_select_favorites)
        self.check_multi = QtWidgets.QCheckBox(SpectraFit)
        self.check_multi.setChecked(True)
        self.check_multi.setObjectName("check_multi")
        self.hor_file_options.addWidget(self.check_multi)
        self.vert_files.addLayout(self.hor_file_options)
        self.horizontalLayout.addLayout(self.vert_files)
        self.vert_parameters = QtWidgets.QVBoxLayout()
        self.vert_parameters.setObjectName("vert_parameters")
        self.tab_pars = QtWidgets.QTableWidget(SpectraFit)
        self.tab_pars.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.tab_pars.setObjectName("tab_pars")
        self.tab_pars.setColumnCount(4)
        self.tab_pars.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tab_pars.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tab_pars.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tab_pars.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tab_pars.setHorizontalHeaderItem(3, item)
        self.vert_parameters.addWidget(self.tab_pars)
        self.hor_parameters = QtWidgets.QHBoxLayout()
        self.hor_parameters.setObjectName("hor_parameters")
        self.b_load_pars = QtWidgets.QPushButton(SpectraFit)
        self.b_load_pars.setObjectName("b_load_pars")
        self.hor_parameters.addWidget(self.b_load_pars)
        self.b_reset_pars = QtWidgets.QPushButton(SpectraFit)
        self.b_reset_pars.setObjectName("b_reset_pars")
        self.hor_parameters.addWidget(self.b_reset_pars)
        self.b_save_pars = QtWidgets.QPushButton(SpectraFit)
        self.b_save_pars.setObjectName("b_save_pars")
        self.hor_parameters.addWidget(self.b_save_pars)
        self.vert_parameters.addLayout(self.hor_parameters)
        self.horizontalLayout.addLayout(self.vert_parameters)
        self.vert_control = QtWidgets.QVBoxLayout()
        self.vert_control.setObjectName("vert_control")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.vert_control.addItem(spacerItem)
        self.grid_options = QtWidgets.QGridLayout()
        self.grid_options.setObjectName("grid_options")
        self.check_hf_mixing = QtWidgets.QCheckBox(SpectraFit)
        self.check_hf_mixing.setEnabled(False)
        self.check_hf_mixing.setObjectName("check_hf_mixing")
        self.grid_options.addWidget(self.check_hf_mixing, 2, 1, 1, 1)
        self.c_routine = QtWidgets.QComboBox(SpectraFit)
        self.c_routine.setObjectName("c_routine")
        self.c_routine.addItem("")
        self.grid_options.addWidget(self.c_routine, 0, 1, 1, 1)
        self.check_save_ascii = QtWidgets.QCheckBox(SpectraFit)
        self.check_save_ascii.setObjectName("check_save_ascii")
        self.grid_options.addWidget(self.check_save_ascii, 3, 1, 1, 1)
        self.check_qi = QtWidgets.QCheckBox(SpectraFit)
        self.check_qi.setEnabled(False)
        self.check_qi.setObjectName("check_qi")
        self.grid_options.addWidget(self.check_qi, 2, 0, 1, 1)
        self.hor_chi2 = QtWidgets.QHBoxLayout()
        self.hor_chi2.setObjectName("hor_chi2")
        self.c_chi2 = QtWidgets.QCheckBox(SpectraFit)
        self.c_chi2.setText("")
        self.c_chi2.setChecked(True)
        self.c_chi2.setObjectName("c_chi2")
        self.hor_chi2.addWidget(self.c_chi2)
        self.l_chi2 = QtWidgets.QLabel(SpectraFit)
        self.l_chi2.setObjectName("l_chi2")
        self.hor_chi2.addWidget(self.l_chi2)
        self.hor_chi2.setStretch(1, 1)
        self.grid_options.addLayout(self.hor_chi2, 0, 0, 1, 1)
        self.check_save_to_db = QtWidgets.QCheckBox(SpectraFit)
        self.check_save_to_db.setObjectName("check_save_to_db")
        self.grid_options.addWidget(self.check_save_to_db, 3, 0, 1, 1)
        self.check_guess_offset = QtWidgets.QCheckBox(SpectraFit)
        self.check_guess_offset.setChecked(True)
        self.check_guess_offset.setObjectName("check_guess_offset")
        self.grid_options.addWidget(self.check_guess_offset, 1, 0, 1, 1)
        self.check_sum = QtWidgets.QCheckBox(SpectraFit)
        self.check_sum.setObjectName("check_sum")
        self.grid_options.addWidget(self.check_sum, 1, 1, 1, 1)
        self.vert_control.addLayout(self.grid_options)
        self.line_plot = QtWidgets.QFrame(SpectraFit)
        self.line_plot.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_plot.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_plot.setObjectName("line_plot")
        self.vert_control.addWidget(self.line_plot)
        self.grid_plot = QtWidgets.QGridLayout()
        self.grid_plot.setObjectName("grid_plot")
        self.b_trs = QtWidgets.QPushButton(SpectraFit)
        self.b_trs.setObjectName("b_trs")
        self.grid_plot.addWidget(self.b_trs, 1, 2, 1, 1)
        self.check_auto = QtWidgets.QCheckBox(SpectraFit)
        self.check_auto.setChecked(True)
        self.check_auto.setObjectName("check_auto")
        self.grid_plot.addWidget(self.check_auto, 1, 1, 1, 1)
        self.b_plot = QtWidgets.QPushButton(SpectraFit)
        self.b_plot.setObjectName("b_plot")
        self.grid_plot.addWidget(self.b_plot, 1, 0, 1, 1)
        self.check_x_as_freq = QtWidgets.QCheckBox(SpectraFit)
        self.check_x_as_freq.setObjectName("check_x_as_freq")
        self.grid_plot.addWidget(self.check_x_as_freq, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.s_fontsize = QtWidgets.QSpinBox(SpectraFit)
        self.s_fontsize.setMinimum(1)
        self.s_fontsize.setProperty("value", 10)
        self.s_fontsize.setObjectName("s_fontsize")
        self.horizontalLayout_2.addWidget(self.s_fontsize)
        self.label = QtWidgets.QLabel(SpectraFit)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.horizontalLayout_2.setStretch(1, 1)
        self.grid_plot.addLayout(self.horizontalLayout_2, 0, 2, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.edit_fmt = QtWidgets.QLineEdit(SpectraFit)
        self.edit_fmt.setMaximumSize(QtCore.QSize(25, 16777215))
        self.edit_fmt.setObjectName("edit_fmt")
        self.horizontalLayout_3.addWidget(self.edit_fmt)
        self.l_fmt = QtWidgets.QLabel(SpectraFit)
        self.l_fmt.setObjectName("l_fmt")
        self.horizontalLayout_3.addWidget(self.l_fmt)
        self.grid_plot.addLayout(self.horizontalLayout_3, 0, 1, 1, 1)
        self.vert_control.addLayout(self.grid_plot)
        self.line_control = QtWidgets.QFrame(SpectraFit)
        self.line_control.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_control.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_control.setObjectName("line_control")
        self.vert_control.addWidget(self.line_control)
        self.grid_actions = QtWidgets.QGridLayout()
        self.grid_actions.setObjectName("grid_actions")
        self.b_fit = QtWidgets.QPushButton(SpectraFit)
        self.b_fit.setObjectName("b_fit")
        self.grid_actions.addWidget(self.b_fit, 1, 0, 1, 1)
        self.progress_fit = QtWidgets.QProgressBar(SpectraFit)
        self.progress_fit.setProperty("value", 100)
        self.progress_fit.setObjectName("progress_fit")
        self.grid_actions.addWidget(self.progress_fit, 0, 0, 1, 1)
        self.vert_control.addLayout(self.grid_actions)
        self.vert_control.setStretch(0, 1)
        self.horizontalLayout.addLayout(self.vert_control)
        self.horizontalLayout.setStretch(1, 1)

        self.retranslateUi(SpectraFit)
        QtCore.QMetaObject.connectSlotsByName(SpectraFit)

    def retranslateUi(self, SpectraFit):
        _translate = QtCore.QCoreApplication.translate
        SpectraFit.setWindowTitle(_translate("SpectraFit", "Form"))
        self.b_select_all.setText(_translate("SpectraFit", "all"))
        self.b_select_col.setText(_translate("SpectraFit", "col"))
        self.b_select_acol.setText(_translate("SpectraFit", "acol"))
        self.b_select_favorites.setText(_translate("SpectraFit", "fav"))
        self.check_multi.setText(_translate("SpectraFit", "multi"))
        item = self.tab_pars.horizontalHeaderItem(0)
        item.setText(_translate("SpectraFit", "Parameter"))
        item = self.tab_pars.horizontalHeaderItem(1)
        item.setText(_translate("SpectraFit", "Value"))
        item = self.tab_pars.horizontalHeaderItem(2)
        item.setText(_translate("SpectraFit", "Fixed"))
        item = self.tab_pars.horizontalHeaderItem(3)
        item.setText(_translate("SpectraFit", "Linked"))
        self.b_load_pars.setText(_translate("SpectraFit", "load pars"))
        self.b_reset_pars.setText(_translate("SpectraFit", "reset pars"))
        self.b_save_pars.setText(_translate("SpectraFit", "save pars"))
        self.check_hf_mixing.setText(_translate("SpectraFit", "HF mixing"))
        self.c_routine.setItemText(0, _translate("SpectraFit", "curve_fit"))
        self.check_save_ascii.setText(_translate("SpectraFit", "save ASCII"))
        self.check_qi.setText(_translate("SpectraFit", "QI"))
        self.l_chi2.setText(_translate("SpectraFit", "<nobr>&Delta; for &chi; <sup>2</sup> = 1</nobr>"))
        self.check_save_to_db.setText(_translate("SpectraFit", "save to db"))
        self.check_guess_offset.setText(_translate("SpectraFit", "guess offset"))
        self.check_sum.setText(_translate("SpectraFit", "sum"))
        self.b_trs.setText(_translate("SpectraFit", "open trs plot"))
        self.check_auto.setText(_translate("SpectraFit", "auto"))
        self.b_plot.setText(_translate("SpectraFit", "plot"))
        self.check_x_as_freq.setText(_translate("SpectraFit", "x as freq."))
        self.label.setText(_translate("SpectraFit", "font size"))
        self.edit_fmt.setText(_translate("SpectraFit", "k."))
        self.l_fmt.setText(_translate("SpectraFit", "fmt"))
        self.b_fit.setText(_translate("SpectraFit", "fit"))
