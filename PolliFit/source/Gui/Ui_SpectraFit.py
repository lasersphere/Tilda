# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Users\Patrick\Documents\Python Projects\Tilda\PolliFit\source\Gui\Ui_SpectraFit.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SpectraFit(object):
    def setupUi(self, SpectraFit):
        SpectraFit.setObjectName("SpectraFit")
        SpectraFit.resize(900, 600)
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
        self.line_model = QtWidgets.QFrame(SpectraFit)
        self.line_model.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_model.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_model.setObjectName("line_model")
        self.vert_control.addWidget(self.line_model)
        self.hor_model = QtWidgets.QHBoxLayout()
        self.hor_model.setObjectName("hor_model")
        self.l_model_options = QtWidgets.QLabel(SpectraFit)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.l_model_options.setFont(font)
        self.l_model_options.setObjectName("l_model_options")
        self.hor_model.addWidget(self.l_model_options)
        self.l_model_file = QtWidgets.QLabel(SpectraFit)
        self.l_model_file.setStyleSheet("color: rgb(0, 0, 255);")
        self.l_model_file.setObjectName("l_model_file")
        self.hor_model.addWidget(self.l_model_file)
        self.hor_model.setStretch(0, 2)
        self.hor_model.setStretch(1, 3)
        self.vert_control.addLayout(self.hor_model)
        self.grid_model = QtWidgets.QGridLayout()
        self.grid_model.setObjectName("grid_model")
        self.c_lineshape = QtWidgets.QComboBox(SpectraFit)
        self.c_lineshape.setObjectName("c_lineshape")
        self.grid_model.addWidget(self.c_lineshape, 1, 1, 1, 1)
        self.hott_offset_order = QtWidgets.QHBoxLayout()
        self.hott_offset_order.setObjectName("hott_offset_order")
        self.edit_offset_order = QtWidgets.QLineEdit(SpectraFit)
        self.edit_offset_order.setMaximumSize(QtCore.QSize(100, 16777215))
        self.edit_offset_order.setObjectName("edit_offset_order")
        self.hott_offset_order.addWidget(self.edit_offset_order)
        self.l_offset_order = QtWidgets.QLabel(SpectraFit)
        self.l_offset_order.setObjectName("l_offset_order")
        self.hott_offset_order.addWidget(self.l_offset_order)
        self.grid_model.addLayout(self.hott_offset_order, 2, 1, 1, 1)
        self.check_hf_mixing = QtWidgets.QCheckBox(SpectraFit)
        self.check_hf_mixing.setEnabled(False)
        self.check_hf_mixing.setObjectName("check_hf_mixing")
        self.grid_model.addWidget(self.check_hf_mixing, 3, 1, 1, 1)
        self.hor_npeaks = QtWidgets.QHBoxLayout()
        self.hor_npeaks.setObjectName("hor_npeaks")
        self.s_npeaks = QtWidgets.QSpinBox(SpectraFit)
        self.s_npeaks.setMaximumSize(QtCore.QSize(40, 16777215))
        self.s_npeaks.setProperty("value", 1)
        self.s_npeaks.setObjectName("s_npeaks")
        self.hor_npeaks.addWidget(self.s_npeaks)
        self.l_npeaks = QtWidgets.QLabel(SpectraFit)
        self.l_npeaks.setObjectName("l_npeaks")
        self.hor_npeaks.addWidget(self.l_npeaks)
        self.grid_model.addLayout(self.hor_npeaks, 1, 0, 1, 1)
        self.check_offset_per_track = QtWidgets.QCheckBox(SpectraFit)
        self.check_offset_per_track.setChecked(True)
        self.check_offset_per_track.setObjectName("check_offset_per_track")
        self.grid_model.addWidget(self.check_offset_per_track, 2, 0, 1, 1)
        self.check_qi = QtWidgets.QCheckBox(SpectraFit)
        self.check_qi.setEnabled(False)
        self.check_qi.setObjectName("check_qi")
        self.grid_model.addWidget(self.check_qi, 3, 0, 1, 1)
        self.vert_control.addLayout(self.grid_model)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.vert_control.addItem(spacerItem1)
        self.line_fit = QtWidgets.QFrame(SpectraFit)
        self.line_fit.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_fit.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_fit.setObjectName("line_fit")
        self.vert_control.addWidget(self.line_fit)
        self.l_fit_options = QtWidgets.QLabel(SpectraFit)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.l_fit_options.setFont(font)
        self.l_fit_options.setObjectName("l_fit_options")
        self.vert_control.addWidget(self.l_fit_options)
        self.grid_fit = QtWidgets.QGridLayout()
        self.grid_fit.setObjectName("grid_fit")
        self.check_linked = QtWidgets.QCheckBox(SpectraFit)
        self.check_linked.setObjectName("check_linked")
        self.grid_fit.addWidget(self.check_linked, 4, 1, 1, 1)
        self.check_save_ascii = QtWidgets.QCheckBox(SpectraFit)
        self.check_save_ascii.setObjectName("check_save_ascii")
        self.grid_fit.addWidget(self.check_save_ascii, 5, 1, 1, 1)
        self.check_guess_offset = QtWidgets.QCheckBox(SpectraFit)
        self.check_guess_offset.setChecked(True)
        self.check_guess_offset.setObjectName("check_guess_offset")
        self.grid_fit.addWidget(self.check_guess_offset, 1, 0, 1, 1)
        self.edit_arithmetics = QtWidgets.QLineEdit(SpectraFit)
        self.edit_arithmetics.setObjectName("edit_arithmetics")
        self.grid_fit.addWidget(self.edit_arithmetics, 2, 0, 1, 1)
        self.c_routine = QtWidgets.QComboBox(SpectraFit)
        self.c_routine.setObjectName("c_routine")
        self.c_routine.addItem("")
        self.grid_fit.addWidget(self.c_routine, 0, 1, 1, 1)
        self.check_summed = QtWidgets.QCheckBox(SpectraFit)
        self.check_summed.setObjectName("check_summed")
        self.grid_fit.addWidget(self.check_summed, 4, 0, 1, 1)
        self.l_arithmetics = QtWidgets.QLabel(SpectraFit)
        self.l_arithmetics.setObjectName("l_arithmetics")
        self.grid_fit.addWidget(self.l_arithmetics, 2, 1, 1, 1)
        self.check_save_to_db = QtWidgets.QCheckBox(SpectraFit)
        self.check_save_to_db.setObjectName("check_save_to_db")
        self.grid_fit.addWidget(self.check_save_to_db, 5, 0, 1, 1)
        self.hor_chi2 = QtWidgets.QHBoxLayout()
        self.hor_chi2.setObjectName("hor_chi2")
        self.check_chi2 = QtWidgets.QCheckBox(SpectraFit)
        self.check_chi2.setText("")
        self.check_chi2.setChecked(True)
        self.check_chi2.setObjectName("check_chi2")
        self.hor_chi2.addWidget(self.check_chi2)
        self.l_chi2 = QtWidgets.QLabel(SpectraFit)
        self.l_chi2.setObjectName("l_chi2")
        self.hor_chi2.addWidget(self.l_chi2)
        self.hor_chi2.setStretch(1, 1)
        self.grid_fit.addLayout(self.hor_chi2, 1, 1, 1, 1)
        self.hor_trs = QtWidgets.QHBoxLayout()
        self.hor_trs.setObjectName("hor_trs")
        self.l_trs = QtWidgets.QLabel(SpectraFit)
        self.l_trs.setObjectName("l_trs")
        self.hor_trs.addWidget(self.l_trs)
        self.b_trs = QtWidgets.QPushButton(SpectraFit)
        self.b_trs.setObjectName("b_trs")
        self.hor_trs.addWidget(self.b_trs)
        self.grid_fit.addLayout(self.hor_trs, 3, 0, 1, 1)
        self.hor_trsplot = QtWidgets.QHBoxLayout()
        self.hor_trsplot.setObjectName("hor_trsplot")
        self.l_trsplot = QtWidgets.QLabel(SpectraFit)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.l_trsplot.setFont(font)
        self.l_trsplot.setObjectName("l_trsplot")
        self.hor_trsplot.addWidget(self.l_trsplot)
        self.b_trsplot = QtWidgets.QPushButton(SpectraFit)
        self.b_trsplot.setObjectName("b_trsplot")
        self.hor_trsplot.addWidget(self.b_trsplot)
        self.hor_trsplot.setStretch(1, 1)
        self.grid_fit.addLayout(self.hor_trsplot, 3, 1, 1, 1)
        self.grid_fit.setColumnStretch(0, 1)
        self.vert_control.addLayout(self.grid_fit)
        self.line_plot = QtWidgets.QFrame(SpectraFit)
        self.line_plot.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_plot.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_plot.setObjectName("line_plot")
        self.vert_control.addWidget(self.line_plot)
        self.l_plot_options = QtWidgets.QLabel(SpectraFit)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.l_plot_options.setFont(font)
        self.l_plot_options.setObjectName("l_plot_options")
        self.vert_control.addWidget(self.l_plot_options)
        self.grid_plot = QtWidgets.QGridLayout()
        self.grid_plot.setObjectName("grid_plot")
        self.check_x_as_freq = QtWidgets.QCheckBox(SpectraFit)
        self.check_x_as_freq.setChecked(True)
        self.check_x_as_freq.setObjectName("check_x_as_freq")
        self.grid_plot.addWidget(self.check_x_as_freq, 0, 0, 1, 1)
        self.check_auto = QtWidgets.QCheckBox(SpectraFit)
        self.check_auto.setChecked(True)
        self.check_auto.setObjectName("check_auto")
        self.grid_plot.addWidget(self.check_auto, 1, 1, 1, 1)
        self.b_plot = QtWidgets.QPushButton(SpectraFit)
        self.b_plot.setObjectName("b_plot")
        self.grid_plot.addWidget(self.b_plot, 1, 0, 1, 1)
        self.hor_fontsize = QtWidgets.QHBoxLayout()
        self.hor_fontsize.setObjectName("hor_fontsize")
        self.s_fontsize = QtWidgets.QSpinBox(SpectraFit)
        self.s_fontsize.setMinimum(1)
        self.s_fontsize.setProperty("value", 10)
        self.s_fontsize.setObjectName("s_fontsize")
        self.hor_fontsize.addWidget(self.s_fontsize)
        self.l_fontsize = QtWidgets.QLabel(SpectraFit)
        self.l_fontsize.setObjectName("l_fontsize")
        self.hor_fontsize.addWidget(self.l_fontsize)
        self.hor_fontsize.setStretch(1, 1)
        self.grid_plot.addLayout(self.hor_fontsize, 1, 2, 1, 1)
        self.hor_fmt = QtWidgets.QHBoxLayout()
        self.hor_fmt.setObjectName("hor_fmt")
        self.edit_fmt = QtWidgets.QLineEdit(SpectraFit)
        self.edit_fmt.setMaximumSize(QtCore.QSize(40, 16777215))
        self.edit_fmt.setObjectName("edit_fmt")
        self.hor_fmt.addWidget(self.edit_fmt)
        self.l_fmt = QtWidgets.QLabel(SpectraFit)
        self.l_fmt.setObjectName("l_fmt")
        self.hor_fmt.addWidget(self.l_fmt)
        self.grid_plot.addLayout(self.hor_fmt, 0, 2, 1, 1)
        self.vert_control.addLayout(self.grid_plot)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.vert_control.addItem(spacerItem2)
        self.line_action = QtWidgets.QFrame(SpectraFit)
        self.line_action.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_action.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_action.setObjectName("line_action")
        self.vert_control.addWidget(self.line_action)
        self.grid_action = QtWidgets.QGridLayout()
        self.grid_action.setObjectName("grid_action")
        self.b_fit = QtWidgets.QPushButton(SpectraFit)
        self.b_fit.setObjectName("b_fit")
        self.grid_action.addWidget(self.b_fit, 2, 0, 1, 1)
        self.progress_fit = QtWidgets.QProgressBar(SpectraFit)
        self.progress_fit.setProperty("value", 100)
        self.progress_fit.setObjectName("progress_fit")
        self.grid_action.addWidget(self.progress_fit, 1, 0, 1, 1)
        self.vert_control.addLayout(self.grid_action)
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
        self.b_load_pars.setText(_translate("SpectraFit", "load"))
        self.b_reset_pars.setText(_translate("SpectraFit", "reset"))
        self.b_save_pars.setText(_translate("SpectraFit", "save"))
        self.l_model_options.setText(_translate("SpectraFit", "Model options"))
        self.l_model_file.setText(_translate("SpectraFit", "<file>"))
        self.edit_offset_order.setText(_translate("SpectraFit", "[0]"))
        self.l_offset_order.setText(_translate("SpectraFit", "order"))
        self.check_hf_mixing.setText(_translate("SpectraFit", "HF mixing"))
        self.l_npeaks.setText(_translate("SpectraFit", "# peaks"))
        self.check_offset_per_track.setText(_translate("SpectraFit", "offset per track"))
        self.check_qi.setText(_translate("SpectraFit", "QI"))
        self.l_fit_options.setText(_translate("SpectraFit", "Fit options"))
        self.check_linked.setText(_translate("SpectraFit", "linked"))
        self.check_save_ascii.setText(_translate("SpectraFit", "save ASCII"))
        self.check_guess_offset.setText(_translate("SpectraFit", "guess offset"))
        self.c_routine.setItemText(0, _translate("SpectraFit", "curve_fit"))
        self.check_summed.setText(_translate("SpectraFit", "summed"))
        self.l_arithmetics.setText(_translate("SpectraFit", "arithmetics"))
        self.check_save_to_db.setText(_translate("SpectraFit", "save to db"))
        self.l_chi2.setText(_translate("SpectraFit", "<nobr>&Delta; for &chi; <sup>2</sup> = 1</nobr>"))
        self.l_trs.setText(_translate("SpectraFit", "TRS"))
        self.b_trs.setText(_translate("SpectraFit", "config ..."))
        self.l_trsplot.setText(_translate("SpectraFit", "<nobr>&larr;</nobr>"))
        self.b_trsplot.setText(_translate("SpectraFit", "open trs plot"))
        self.l_plot_options.setText(_translate("SpectraFit", "Plot options"))
        self.check_x_as_freq.setText(_translate("SpectraFit", "x as freq."))
        self.check_auto.setText(_translate("SpectraFit", "auto"))
        self.b_plot.setText(_translate("SpectraFit", "plot"))
        self.l_fontsize.setText(_translate("SpectraFit", "font size"))
        self.edit_fmt.setText(_translate("SpectraFit", ".k"))
        self.l_fmt.setText(_translate("SpectraFit", "fmt"))
        self.b_fit.setText(_translate("SpectraFit", "fit"))
