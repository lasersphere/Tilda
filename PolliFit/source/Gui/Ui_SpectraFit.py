# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Users\Patrick\Documents\Python Projects\Tilda\PolliFit\source\Gui\Ui_SpectraFit.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SpectraFit(object):
    def setupUi(self, SpectraFit):
        SpectraFit.setObjectName("SpectraFit")
        SpectraFit.resize(1031, 472)
        self.verticalLayout = QtWidgets.QVBoxLayout(SpectraFit)
        self.verticalLayout.setObjectName("verticalLayout")
        self.splitter = QtWidgets.QSplitter(SpectraFit)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setObjectName("splitter")
        self.vert_files = QtWidgets.QWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vert_files.sizePolicy().hasHeightForWidth())
        self.vert_files.setSizePolicy(sizePolicy)
        self.vert_files.setObjectName("vert_files")
        self.vert_files_layout = QtWidgets.QVBoxLayout(self.vert_files)
        self.vert_files_layout.setContentsMargins(0, 0, 0, 0)
        self.vert_files_layout.setObjectName("vert_files_layout")
        self.hor_run_select = QtWidgets.QHBoxLayout()
        self.hor_run_select.setObjectName("hor_run_select")
        self.c_run = QtWidgets.QComboBox(self.vert_files)
        self.c_run.setObjectName("c_run")
        self.hor_run_select.addWidget(self.c_run)
        self.c_iso = QtWidgets.QComboBox(self.vert_files)
        self.c_iso.setObjectName("c_iso")
        self.hor_run_select.addWidget(self.c_iso)
        self.hor_run_select.setStretch(0, 1)
        self.hor_run_select.setStretch(1, 1)
        self.vert_files_layout.addLayout(self.hor_run_select)
        self.list_files = QtWidgets.QListWidget(self.vert_files)
        self.list_files.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_files.setResizeMode(QtWidgets.QListView.Adjust)
        self.list_files.setObjectName("list_files")
        self.vert_files_layout.addWidget(self.list_files)
        self.scroll_files = QtWidgets.QScrollArea(self.vert_files)
        self.scroll_files.setMinimumSize(QtCore.QSize(0, 40))
        self.scroll_files.setMaximumSize(QtCore.QSize(16777215, 64))
        self.scroll_files.setFrameShadow(QtWidgets.QFrame.Plain)
        self.scroll_files.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll_files.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll_files.setWidgetResizable(True)
        self.scroll_files.setObjectName("scroll_files")
        self.hor_file_options = QtWidgets.QWidget()
        self.hor_file_options.setGeometry(QtCore.QRect(0, 0, 293, 45))
        self.hor_file_options.setObjectName("hor_file_options")
        self.hor_file_options_layout = QtWidgets.QHBoxLayout(self.hor_file_options)
        self.hor_file_options_layout.setObjectName("hor_file_options_layout")
        self.b_select_all = QtWidgets.QPushButton(self.hor_file_options)
        self.b_select_all.setMinimumSize(QtCore.QSize(0, 26))
        self.b_select_all.setMaximumSize(QtCore.QSize(50, 26))
        self.b_select_all.setObjectName("b_select_all")
        self.hor_file_options_layout.addWidget(self.b_select_all)
        self.b_select_col = QtWidgets.QPushButton(self.hor_file_options)
        self.b_select_col.setMinimumSize(QtCore.QSize(0, 26))
        self.b_select_col.setMaximumSize(QtCore.QSize(50, 26))
        self.b_select_col.setObjectName("b_select_col")
        self.hor_file_options_layout.addWidget(self.b_select_col)
        self.b_select_acol = QtWidgets.QPushButton(self.hor_file_options)
        self.b_select_acol.setMinimumSize(QtCore.QSize(0, 26))
        self.b_select_acol.setMaximumSize(QtCore.QSize(50, 26))
        self.b_select_acol.setObjectName("b_select_acol")
        self.hor_file_options_layout.addWidget(self.b_select_acol)
        self.b_select_favorites = QtWidgets.QPushButton(self.hor_file_options)
        self.b_select_favorites.setEnabled(False)
        self.b_select_favorites.setMinimumSize(QtCore.QSize(0, 26))
        self.b_select_favorites.setMaximumSize(QtCore.QSize(50, 26))
        self.b_select_favorites.setObjectName("b_select_favorites")
        self.hor_file_options_layout.addWidget(self.b_select_favorites)
        spacerItem = QtWidgets.QSpacerItem(2, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.hor_file_options_layout.addItem(spacerItem)
        self.check_multi = QtWidgets.QCheckBox(self.hor_file_options)
        self.check_multi.setChecked(True)
        self.check_multi.setObjectName("check_multi")
        self.hor_file_options_layout.addWidget(self.check_multi)
        self.scroll_files.setWidget(self.hor_file_options)
        self.vert_files_layout.addWidget(self.scroll_files)
        self.vert_files_layout.setStretch(1, 1)
        self.vert_parameters = QtWidgets.QWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vert_parameters.sizePolicy().hasHeightForWidth())
        self.vert_parameters.setSizePolicy(sizePolicy)
        self.vert_parameters.setObjectName("vert_parameters")
        self.vert_parameters_layout = QtWidgets.QVBoxLayout(self.vert_parameters)
        self.vert_parameters_layout.setContentsMargins(0, 0, 0, 0)
        self.vert_parameters_layout.setObjectName("vert_parameters_layout")
        self.tab_pars = QtWidgets.QTableWidget(self.vert_parameters)
        self.tab_pars.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tab_pars.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
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
        self.vert_parameters_layout.addWidget(self.tab_pars)
        self.scroll_parameters = QtWidgets.QScrollArea(self.vert_parameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scroll_parameters.sizePolicy().hasHeightForWidth())
        self.scroll_parameters.setSizePolicy(sizePolicy)
        self.scroll_parameters.setMinimumSize(QtCore.QSize(0, 40))
        self.scroll_parameters.setMaximumSize(QtCore.QSize(16777215, 64))
        self.scroll_parameters.setFrameShadow(QtWidgets.QFrame.Plain)
        self.scroll_parameters.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll_parameters.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll_parameters.setWidgetResizable(True)
        self.scroll_parameters.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.scroll_parameters.setObjectName("scroll_parameters")
        self.hor_parameters = QtWidgets.QWidget()
        self.hor_parameters.setGeometry(QtCore.QRect(0, 0, 409, 45))
        self.hor_parameters.setObjectName("hor_parameters")
        self.hor_parameters_layout = QtWidgets.QHBoxLayout(self.hor_parameters)
        self.hor_parameters_layout.setContentsMargins(6, 6, 6, 6)
        self.hor_parameters_layout.setObjectName("hor_parameters_layout")
        self.b_load_pars = QtWidgets.QPushButton(self.hor_parameters)
        self.b_load_pars.setMinimumSize(QtCore.QSize(0, 26))
        self.b_load_pars.setMaximumSize(QtCore.QSize(60, 26))
        self.b_load_pars.setObjectName("b_load_pars")
        self.hor_parameters_layout.addWidget(self.b_load_pars)
        self.b_save_pars = QtWidgets.QPushButton(self.hor_parameters)
        self.b_save_pars.setMinimumSize(QtCore.QSize(0, 26))
        self.b_save_pars.setMaximumSize(QtCore.QSize(60, 26))
        self.b_save_pars.setObjectName("b_save_pars")
        self.hor_parameters_layout.addWidget(self.b_save_pars)
        spacerItem1 = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.hor_parameters_layout.addItem(spacerItem1)
        self.b_up = QtWidgets.QPushButton(self.hor_parameters)
        self.b_up.setMinimumSize(QtCore.QSize(0, 26))
        self.b_up.setMaximumSize(QtCore.QSize(40, 26))
        self.b_up.setObjectName("b_up")
        self.hor_parameters_layout.addWidget(self.b_up)
        self.b_down = QtWidgets.QPushButton(self.hor_parameters)
        self.b_down.setMinimumSize(QtCore.QSize(0, 26))
        self.b_down.setMaximumSize(QtCore.QSize(40, 26))
        self.b_down.setObjectName("b_down")
        self.hor_parameters_layout.addWidget(self.b_down)
        spacerItem2 = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.hor_parameters_layout.addItem(spacerItem2)
        self.b_copy = QtWidgets.QPushButton(self.hor_parameters)
        self.b_copy.setMinimumSize(QtCore.QSize(0, 26))
        self.b_copy.setMaximumSize(QtCore.QSize(60, 26))
        self.b_copy.setObjectName("b_copy")
        self.hor_parameters_layout.addWidget(self.b_copy)
        self.b_reset_pars = QtWidgets.QPushButton(self.hor_parameters)
        self.b_reset_pars.setMinimumSize(QtCore.QSize(0, 26))
        self.b_reset_pars.setMaximumSize(QtCore.QSize(60, 26))
        self.b_reset_pars.setObjectName("b_reset_pars")
        self.hor_parameters_layout.addWidget(self.b_reset_pars)
        self.hor_parameters_layout.setStretch(2, 1)
        self.hor_parameters_layout.setStretch(5, 1)
        self.scroll_parameters.setWidget(self.hor_parameters)
        self.vert_parameters_layout.addWidget(self.scroll_parameters)
        self.vert_parameters_layout.setStretch(0, 1)
        self.vert_control = QtWidgets.QWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vert_control.sizePolicy().hasHeightForWidth())
        self.vert_control.setSizePolicy(sizePolicy)
        self.vert_control.setMinimumSize(QtCore.QSize(300, 0))
        self.vert_control.setMaximumSize(QtCore.QSize(300, 16777215))
        self.vert_control.setObjectName("vert_control")
        self.vert_control_layout = QtWidgets.QVBoxLayout(self.vert_control)
        self.vert_control_layout.setContentsMargins(0, 0, 0, 0)
        self.vert_control_layout.setObjectName("vert_control_layout")
        self.scroll_xontrol = QtWidgets.QScrollArea(self.vert_control)
        self.scroll_xontrol.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.scroll_xontrol.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.scroll_xontrol.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll_xontrol.setWidgetResizable(True)
        self.scroll_xontrol.setAlignment(QtCore.Qt.AlignCenter)
        self.scroll_xontrol.setObjectName("scroll_xontrol")
        self.vert_options = QtWidgets.QWidget()
        self.vert_options.setGeometry(QtCore.QRect(0, 0, 281, 518))
        self.vert_options.setObjectName("vert_options")
        self.vert_options_layout = QtWidgets.QVBoxLayout(self.vert_options)
        self.vert_options_layout.setContentsMargins(6, 6, 6, 6)
        self.vert_options_layout.setObjectName("vert_options_layout")
        spacerItem3 = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.vert_options_layout.addItem(spacerItem3)
        self.line_model = QtWidgets.QFrame(self.vert_options)
        self.line_model.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_model.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_model.setObjectName("line_model")
        self.vert_options_layout.addWidget(self.line_model)
        self.hor_model = QtWidgets.QHBoxLayout()
        self.hor_model.setObjectName("hor_model")
        self.l_model_options = QtWidgets.QLabel(self.vert_options)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.l_model_options.setFont(font)
        self.l_model_options.setObjectName("l_model_options")
        self.hor_model.addWidget(self.l_model_options)
        self.l_model_file = QtWidgets.QLabel(self.vert_options)
        self.l_model_file.setStyleSheet("color: rgb(0, 0, 255);")
        self.l_model_file.setObjectName("l_model_file")
        self.hor_model.addWidget(self.l_model_file)
        self.hor_model.setStretch(0, 2)
        self.hor_model.setStretch(1, 3)
        self.vert_options_layout.addLayout(self.hor_model)
        self.grid_model = QtWidgets.QWidget(self.vert_options)
        self.grid_model.setObjectName("grid_model")
        self.grid_model_layout = QtWidgets.QGridLayout(self.grid_model)
        self.grid_model_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_model_layout.setObjectName("grid_model_layout")
        self.check_offset_per_track = QtWidgets.QCheckBox(self.grid_model)
        self.check_offset_per_track.setObjectName("check_offset_per_track")
        self.grid_model_layout.addWidget(self.check_offset_per_track, 3, 0, 1, 1)
        self.c_lineshape = QtWidgets.QComboBox(self.grid_model)
        self.c_lineshape.setObjectName("c_lineshape")
        self.grid_model_layout.addWidget(self.c_lineshape, 1, 1, 1, 1)
        self.check_hf_mixing = QtWidgets.QCheckBox(self.grid_model)
        self.check_hf_mixing.setEnabled(False)
        self.check_hf_mixing.setObjectName("check_hf_mixing")
        self.grid_model_layout.addWidget(self.check_hf_mixing, 4, 1, 1, 1)
        self.hott_offset_order = QtWidgets.QHBoxLayout()
        self.hott_offset_order.setObjectName("hott_offset_order")
        self.edit_offset_order = QtWidgets.QLineEdit(self.grid_model)
        self.edit_offset_order.setMaximumSize(QtCore.QSize(100, 16777215))
        self.edit_offset_order.setObjectName("edit_offset_order")
        self.hott_offset_order.addWidget(self.edit_offset_order)
        self.l_offset_order = QtWidgets.QLabel(self.grid_model)
        self.l_offset_order.setObjectName("l_offset_order")
        self.hott_offset_order.addWidget(self.l_offset_order)
        self.grid_model_layout.addLayout(self.hott_offset_order, 3, 1, 1, 1)
        self.b_racah = QtWidgets.QPushButton(self.grid_model)
        self.b_racah.setObjectName("b_racah")
        self.grid_model_layout.addWidget(self.b_racah, 5, 1, 1, 1)
        self.check_qi = QtWidgets.QCheckBox(self.grid_model)
        self.check_qi.setEnabled(False)
        self.check_qi.setObjectName("check_qi")
        self.grid_model_layout.addWidget(self.check_qi, 4, 0, 1, 1)
        self.hor_npeaks = QtWidgets.QHBoxLayout()
        self.hor_npeaks.setObjectName("hor_npeaks")
        self.s_npeaks = QtWidgets.QSpinBox(self.grid_model)
        self.s_npeaks.setMaximumSize(QtCore.QSize(40, 16777215))
        self.s_npeaks.setMinimum(1)
        self.s_npeaks.setProperty("value", 1)
        self.s_npeaks.setObjectName("s_npeaks")
        self.hor_npeaks.addWidget(self.s_npeaks)
        self.l_npeaks = QtWidgets.QLabel(self.grid_model)
        self.l_npeaks.setObjectName("l_npeaks")
        self.hor_npeaks.addWidget(self.l_npeaks)
        self.grid_model_layout.addLayout(self.hor_npeaks, 1, 0, 1, 1)
        self.c_convolve = QtWidgets.QComboBox(self.grid_model)
        self.c_convolve.setObjectName("c_convolve")
        self.grid_model_layout.addWidget(self.c_convolve, 2, 1, 1, 1)
        self.l_convolve = QtWidgets.QLabel(self.grid_model)
        self.l_convolve.setObjectName("l_convolve")
        self.grid_model_layout.addWidget(self.l_convolve, 2, 0, 1, 1)
        self.grid_model_layout.setColumnStretch(0, 1)
        self.grid_model_layout.setColumnStretch(1, 1)
        self.vert_options_layout.addWidget(self.grid_model)
        spacerItem4 = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.vert_options_layout.addItem(spacerItem4)
        self.line_fit = QtWidgets.QFrame(self.vert_options)
        self.line_fit.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_fit.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_fit.setObjectName("line_fit")
        self.vert_options_layout.addWidget(self.line_fit)
        self.l_fit_options = QtWidgets.QLabel(self.vert_options)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.l_fit_options.setFont(font)
        self.l_fit_options.setObjectName("l_fit_options")
        self.vert_options_layout.addWidget(self.l_fit_options)
        self.grid_fit = QtWidgets.QWidget(self.vert_options)
        self.grid_fit.setObjectName("grid_fit")
        self.grid_fit_layout = QtWidgets.QGridLayout(self.grid_fit)
        self.grid_fit_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_fit_layout.setObjectName("grid_fit_layout")
        self.hor_chi2 = QtWidgets.QHBoxLayout()
        self.hor_chi2.setObjectName("hor_chi2")
        self.check_chi2 = QtWidgets.QCheckBox(self.grid_fit)
        self.check_chi2.setText("")
        self.check_chi2.setChecked(True)
        self.check_chi2.setObjectName("check_chi2")
        self.hor_chi2.addWidget(self.check_chi2)
        self.l_chi2 = QtWidgets.QLabel(self.grid_fit)
        self.l_chi2.setObjectName("l_chi2")
        self.hor_chi2.addWidget(self.l_chi2)
        self.hor_chi2.setStretch(1, 1)
        self.grid_fit_layout.addLayout(self.hor_chi2, 1, 1, 1, 1)
        self.hor_samples = QtWidgets.QHBoxLayout()
        self.hor_samples.setObjectName("hor_samples")
        self.s_samples_mc = QtWidgets.QSpinBox(self.grid_fit)
        self.s_samples_mc.setEnabled(False)
        self.s_samples_mc.setMinimum(100)
        self.s_samples_mc.setMaximum(1000000)
        self.s_samples_mc.setSingleStep(100)
        self.s_samples_mc.setObjectName("s_samples_mc")
        self.hor_samples.addWidget(self.s_samples_mc)
        self.l_samples_mc = QtWidgets.QLabel(self.grid_fit)
        self.l_samples_mc.setEnabled(False)
        self.l_samples_mc.setObjectName("l_samples_mc")
        self.hor_samples.addWidget(self.l_samples_mc)
        self.hor_samples.setStretch(0, 1)
        self.grid_fit_layout.addLayout(self.hor_samples, 2, 1, 1, 1)
        self.hor_trsplot = QtWidgets.QHBoxLayout()
        self.hor_trsplot.setObjectName("hor_trsplot")
        self.l_trsplot = QtWidgets.QLabel(self.grid_fit)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.l_trsplot.setFont(font)
        self.l_trsplot.setObjectName("l_trsplot")
        self.hor_trsplot.addWidget(self.l_trsplot)
        self.b_trsplot = QtWidgets.QPushButton(self.grid_fit)
        self.b_trsplot.setObjectName("b_trsplot")
        self.hor_trsplot.addWidget(self.b_trsplot)
        self.hor_trsplot.setStretch(1, 1)
        self.grid_fit_layout.addLayout(self.hor_trsplot, 4, 1, 1, 1)
        self.c_routine = QtWidgets.QComboBox(self.grid_fit)
        self.c_routine.setObjectName("c_routine")
        self.c_routine.addItem("")
        self.grid_fit_layout.addWidget(self.c_routine, 0, 1, 1, 1)
        self.check_save_to_db = QtWidgets.QCheckBox(self.grid_fit)
        self.check_save_to_db.setObjectName("check_save_to_db")
        self.grid_fit_layout.addWidget(self.check_save_to_db, 6, 0, 1, 1)
        self.check_summed = QtWidgets.QCheckBox(self.grid_fit)
        self.check_summed.setEnabled(False)
        self.check_summed.setObjectName("check_summed")
        self.grid_fit_layout.addWidget(self.check_summed, 5, 0, 1, 1)
        self.check_save_figure = QtWidgets.QCheckBox(self.grid_fit)
        self.check_save_figure.setObjectName("check_save_figure")
        self.grid_fit_layout.addWidget(self.check_save_figure, 6, 1, 1, 1)
        self.edit_arithmetics = QtWidgets.QLineEdit(self.grid_fit)
        self.edit_arithmetics.setObjectName("edit_arithmetics")
        self.grid_fit_layout.addWidget(self.edit_arithmetics, 3, 0, 1, 1)
        self.check_guess_offset = QtWidgets.QCheckBox(self.grid_fit)
        self.check_guess_offset.setChecked(True)
        self.check_guess_offset.setObjectName("check_guess_offset")
        self.grid_fit_layout.addWidget(self.check_guess_offset, 1, 0, 1, 1)
        self.check_linked = QtWidgets.QCheckBox(self.grid_fit)
        self.check_linked.setObjectName("check_linked")
        self.grid_fit_layout.addWidget(self.check_linked, 5, 1, 1, 1)
        self.check_cov_mc = QtWidgets.QCheckBox(self.grid_fit)
        self.check_cov_mc.setObjectName("check_cov_mc")
        self.grid_fit_layout.addWidget(self.check_cov_mc, 2, 0, 1, 1)
        self.hor_trs = QtWidgets.QHBoxLayout()
        self.hor_trs.setObjectName("hor_trs")
        self.l_trs = QtWidgets.QLabel(self.grid_fit)
        self.l_trs.setObjectName("l_trs")
        self.hor_trs.addWidget(self.l_trs)
        self.b_trs = QtWidgets.QPushButton(self.grid_fit)
        self.b_trs.setObjectName("b_trs")
        self.hor_trs.addWidget(self.b_trs)
        self.grid_fit_layout.addLayout(self.hor_trs, 4, 0, 1, 1)
        self.hor_arithmetics = QtWidgets.QHBoxLayout()
        self.hor_arithmetics.setObjectName("hor_arithmetics")
        self.l_arithmetics = QtWidgets.QLabel(self.grid_fit)
        self.l_arithmetics.setObjectName("l_arithmetics")
        self.hor_arithmetics.addWidget(self.l_arithmetics)
        self.check_arithmetics = QtWidgets.QCheckBox(self.grid_fit)
        self.check_arithmetics.setChecked(True)
        self.check_arithmetics.setObjectName("check_arithmetics")
        self.hor_arithmetics.addWidget(self.check_arithmetics)
        self.grid_fit_layout.addLayout(self.hor_arithmetics, 3, 1, 1, 1)
        self.hor_xaxis = QtWidgets.QHBoxLayout()
        self.hor_xaxis.setObjectName("hor_xaxis")
        self.l_xaxis = QtWidgets.QLabel(self.grid_fit)
        self.l_xaxis.setObjectName("l_xaxis")
        self.hor_xaxis.addWidget(self.l_xaxis)
        self.c_xaxis = QtWidgets.QComboBox(self.grid_fit)
        self.c_xaxis.setObjectName("c_xaxis")
        self.c_xaxis.addItem("")
        self.c_xaxis.addItem("")
        self.c_xaxis.addItem("")
        self.hor_xaxis.addWidget(self.c_xaxis)
        self.hor_xaxis.setStretch(1, 1)
        self.grid_fit_layout.addLayout(self.hor_xaxis, 0, 0, 1, 1)
        self.grid_fit_layout.setColumnStretch(0, 1)
        self.vert_options_layout.addWidget(self.grid_fit)
        spacerItem5 = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.vert_options_layout.addItem(spacerItem5)
        self.line_plot = QtWidgets.QFrame(self.vert_options)
        self.line_plot.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_plot.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_plot.setObjectName("line_plot")
        self.vert_options_layout.addWidget(self.line_plot)
        self.l_plot_options = QtWidgets.QLabel(self.vert_options)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.l_plot_options.setFont(font)
        self.l_plot_options.setObjectName("l_plot_options")
        self.vert_options_layout.addWidget(self.l_plot_options)
        self.grid_plot = QtWidgets.QWidget(self.vert_options)
        self.grid_plot.setObjectName("grid_plot")
        self.grid_plot_layout = QtWidgets.QGridLayout(self.grid_plot)
        self.grid_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_plot_layout.setObjectName("grid_plot_layout")
        self.hor_fmt = QtWidgets.QHBoxLayout()
        self.hor_fmt.setObjectName("hor_fmt")
        self.edit_fmt = QtWidgets.QLineEdit(self.grid_plot)
        self.edit_fmt.setMaximumSize(QtCore.QSize(40, 16777215))
        self.edit_fmt.setObjectName("edit_fmt")
        self.hor_fmt.addWidget(self.edit_fmt)
        self.l_fmt = QtWidgets.QLabel(self.grid_plot)
        self.l_fmt.setObjectName("l_fmt")
        self.hor_fmt.addWidget(self.l_fmt)
        self.grid_plot_layout.addLayout(self.hor_fmt, 0, 2, 1, 1)
        self.b_plot = QtWidgets.QPushButton(self.grid_plot)
        self.b_plot.setObjectName("b_plot")
        self.grid_plot_layout.addWidget(self.b_plot, 1, 0, 1, 1)
        self.check_x_as_freq = QtWidgets.QCheckBox(self.grid_plot)
        self.check_x_as_freq.setChecked(True)
        self.check_x_as_freq.setObjectName("check_x_as_freq")
        self.grid_plot_layout.addWidget(self.check_x_as_freq, 0, 0, 1, 1)
        self.check_auto = QtWidgets.QCheckBox(self.grid_plot)
        self.check_auto.setChecked(True)
        self.check_auto.setObjectName("check_auto")
        self.grid_plot_layout.addWidget(self.check_auto, 1, 1, 1, 1)
        self.hor_fontsize = QtWidgets.QHBoxLayout()
        self.hor_fontsize.setObjectName("hor_fontsize")
        self.s_fontsize = QtWidgets.QSpinBox(self.grid_plot)
        self.s_fontsize.setMinimum(1)
        self.s_fontsize.setProperty("value", 10)
        self.s_fontsize.setObjectName("s_fontsize")
        self.hor_fontsize.addWidget(self.s_fontsize)
        self.l_fontsize = QtWidgets.QLabel(self.grid_plot)
        self.l_fontsize.setObjectName("l_fontsize")
        self.hor_fontsize.addWidget(self.l_fontsize)
        self.grid_plot_layout.addLayout(self.hor_fontsize, 1, 2, 1, 1)
        self.b_save_ascii = QtWidgets.QPushButton(self.grid_plot)
        self.b_save_ascii.setObjectName("b_save_ascii")
        self.grid_plot_layout.addWidget(self.b_save_ascii, 2, 0, 1, 1)
        self.b_save_figure = QtWidgets.QPushButton(self.grid_plot)
        self.b_save_figure.setObjectName("b_save_figure")
        self.grid_plot_layout.addWidget(self.b_save_figure, 2, 1, 1, 1)
        self.hor_save_fig = QtWidgets.QHBoxLayout()
        self.hor_save_fig.setObjectName("hor_save_fig")
        self.l_as = QtWidgets.QLabel(self.grid_plot)
        self.l_as.setAlignment(QtCore.Qt.AlignCenter)
        self.l_as.setObjectName("l_as")
        self.hor_save_fig.addWidget(self.l_as)
        self.c_fig = QtWidgets.QComboBox(self.grid_plot)
        self.c_fig.setObjectName("c_fig")
        self.c_fig.addItem("")
        self.c_fig.addItem("")
        self.hor_save_fig.addWidget(self.c_fig)
        self.grid_plot_layout.addLayout(self.hor_save_fig, 2, 2, 1, 1)
        self.grid_plot_layout.setColumnStretch(0, 1)
        self.grid_plot_layout.setColumnStretch(1, 1)
        self.grid_plot_layout.setColumnStretch(2, 1)
        self.vert_options_layout.addWidget(self.grid_plot)
        spacerItem6 = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.vert_options_layout.addItem(spacerItem6)
        self.vert_options_layout.setStretch(0, 1)
        self.vert_options_layout.setStretch(4, 1)
        self.vert_options_layout.setStretch(8, 1)
        self.vert_options_layout.setStretch(12, 1)
        self.scroll_xontrol.setWidget(self.vert_options)
        self.vert_control_layout.addWidget(self.scroll_xontrol)
        self.vert_action = QtWidgets.QFrame(self.vert_control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.vert_action.sizePolicy().hasHeightForWidth())
        self.vert_action.setSizePolicy(sizePolicy)
        self.vert_action.setMinimumSize(QtCore.QSize(0, 64))
        self.vert_action.setMaximumSize(QtCore.QSize(16777215, 64))
        self.vert_action.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.vert_action.setObjectName("vert_action")
        self.vert_action_layout = QtWidgets.QVBoxLayout(self.vert_action)
        self.vert_action_layout.setContentsMargins(6, 6, 6, 6)
        self.vert_action_layout.setObjectName("vert_action_layout")
        self.progress_fit = QtWidgets.QProgressBar(self.vert_action)
        self.progress_fit.setEnabled(False)
        self.progress_fit.setMaximumSize(QtCore.QSize(16777215, 16))
        self.progress_fit.setProperty("value", 100)
        self.progress_fit.setObjectName("progress_fit")
        self.vert_action_layout.addWidget(self.progress_fit)
        self.hor_fit = QtWidgets.QWidget(self.vert_action)
        self.hor_fit.setMinimumSize(QtCore.QSize(0, 26))
        self.hor_fit.setObjectName("hor_fit")
        self.hor_fit_layout = QtWidgets.QHBoxLayout(self.hor_fit)
        self.hor_fit_layout.setContentsMargins(0, 0, 0, 0)
        self.hor_fit_layout.setObjectName("hor_fit_layout")
        self.b_fit = QtWidgets.QPushButton(self.hor_fit)
        self.b_fit.setMinimumSize(QtCore.QSize(0, 26))
        self.b_fit.setMaximumSize(QtCore.QSize(16777215, 26))
        self.b_fit.setObjectName("b_fit")
        self.hor_fit_layout.addWidget(self.b_fit)
        self.b_abort = QtWidgets.QPushButton(self.hor_fit)
        self.b_abort.setEnabled(False)
        self.b_abort.setMinimumSize(QtCore.QSize(0, 26))
        self.b_abort.setMaximumSize(QtCore.QSize(16777215, 26))
        self.b_abort.setObjectName("b_abort")
        self.hor_fit_layout.addWidget(self.b_abort)
        self.hor_fit_layout.setStretch(0, 1)
        self.hor_fit_layout.setStretch(1, 1)
        self.vert_action_layout.addWidget(self.hor_fit)
        self.vert_control_layout.addWidget(self.vert_action)
        self.vert_control_layout.setStretch(0, 1)
        self.verticalLayout.addWidget(self.splitter)

        self.retranslateUi(SpectraFit)
        self.c_xaxis.setCurrentIndex(0)
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
        self.b_save_pars.setText(_translate("SpectraFit", "save"))
        self.b_up.setText(_translate("SpectraFit", "<"))
        self.b_down.setText(_translate("SpectraFit", ">"))
        self.b_copy.setText(_translate("SpectraFit", "copy"))
        self.b_reset_pars.setText(_translate("SpectraFit", "reset"))
        self.l_model_options.setText(_translate("SpectraFit", "Model options"))
        self.l_model_file.setText(_translate("SpectraFit", "<file>"))
        self.check_offset_per_track.setText(_translate("SpectraFit", "offset per track"))
        self.check_hf_mixing.setText(_translate("SpectraFit", "HF mixing"))
        self.edit_offset_order.setText(_translate("SpectraFit", "[0]"))
        self.l_offset_order.setText(_translate("SpectraFit", "order"))
        self.b_racah.setText(_translate("SpectraFit", "Racah intensities"))
        self.check_qi.setText(_translate("SpectraFit", "QI"))
        self.l_npeaks.setText(_translate("SpectraFit", "# peaks"))
        self.l_convolve.setText(_translate("SpectraFit", "Convolve with"))
        self.l_fit_options.setText(_translate("SpectraFit", "Fit options"))
        self.l_chi2.setText(_translate("SpectraFit", "<nobr>&Delta; for &chi; <sup>2</sup> = 1</nobr>"))
        self.l_samples_mc.setText(_translate("SpectraFit", "# samples"))
        self.l_trsplot.setText(_translate("SpectraFit", "<nobr>&larr;</nobr>"))
        self.b_trsplot.setText(_translate("SpectraFit", "open trs plot"))
        self.c_routine.setItemText(0, _translate("SpectraFit", "curve_fit"))
        self.check_save_to_db.setText(_translate("SpectraFit", "save to db"))
        self.check_summed.setText(_translate("SpectraFit", "summed"))
        self.check_save_figure.setText(_translate("SpectraFit", "save figure"))
        self.check_guess_offset.setText(_translate("SpectraFit", "guess offset"))
        self.check_linked.setText(_translate("SpectraFit", "linked"))
        self.check_cov_mc.setText(_translate("SpectraFit", "cov. from Monte-Carlo"))
        self.l_trs.setText(_translate("SpectraFit", "TRS"))
        self.b_trs.setText(_translate("SpectraFit", "config ..."))
        self.l_arithmetics.setText(_translate("SpectraFit", "arithmetics"))
        self.check_arithmetics.setText(_translate("SpectraFit", "db"))
        self.l_xaxis.setText(_translate("SpectraFit", "x-axis"))
        self.c_xaxis.setItemText(0, _translate("SpectraFit", "ion frequencies"))
        self.c_xaxis.setItemText(1, _translate("SpectraFit", "lab frequencies"))
        self.c_xaxis.setItemText(2, _translate("SpectraFit", "DAC volt (TODO)"))
        self.l_plot_options.setText(_translate("SpectraFit", "Plot options"))
        self.edit_fmt.setText(_translate("SpectraFit", ".k"))
        self.l_fmt.setText(_translate("SpectraFit", "fmt"))
        self.b_plot.setText(_translate("SpectraFit", "plot"))
        self.check_x_as_freq.setText(_translate("SpectraFit", "x as freq."))
        self.check_auto.setText(_translate("SpectraFit", "auto"))
        self.l_fontsize.setText(_translate("SpectraFit", "font size"))
        self.b_save_ascii.setText(_translate("SpectraFit", "save ASCII"))
        self.b_save_figure.setText(_translate("SpectraFit", "save figure"))
        self.l_as.setText(_translate("SpectraFit", "as"))
        self.c_fig.setItemText(0, _translate("SpectraFit", ".png"))
        self.c_fig.setItemText(1, _translate("SpectraFit", ".pdf"))
        self.b_fit.setText(_translate("SpectraFit", "fit"))
        self.b_abort.setText(_translate("SpectraFit", "abort"))
