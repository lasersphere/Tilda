# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\Users\Patrick\Documents\Python Projects\Tilda\Tilda\Interface\LiveDataPlottingUi\Ui_LiveDataPlotting.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow_LiveDataPlotting(object):
    def setupUi(self, MainWindow_LiveDataPlotting):
        MainWindow_LiveDataPlotting.setObjectName("MainWindow_LiveDataPlotting")
        MainWindow_LiveDataPlotting.resize(1023, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow_LiveDataPlotting)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_sum = QtWidgets.QWidget()
        self.tab_sum.setObjectName("tab_sum")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_sum)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_outer_sum_plot = QtWidgets.QWidget(self.tab_sum)
        self.widget_outer_sum_plot.setObjectName("widget_outer_sum_plot")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_outer_sum_plot)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget_inner_sum_plot = QtWidgets.QWidget(self.widget_outer_sum_plot)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_inner_sum_plot.sizePolicy().hasHeightForWidth())
        self.widget_inner_sum_plot.setSizePolicy(sizePolicy)
        self.widget_inner_sum_plot.setObjectName("widget_inner_sum_plot")
        self.horizontalLayout.addWidget(self.widget_inner_sum_plot)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_3.setContentsMargins(-1, -1, 50, -1)
        self.verticalLayout_3.setSpacing(7)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(self.widget_outer_sum_plot)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.comboBox_select_sum_for_pmts = QtWidgets.QComboBox(self.widget_outer_sum_plot)
        self.comboBox_select_sum_for_pmts.setMaximumSize(QtCore.QSize(150, 16777215))
        self.comboBox_select_sum_for_pmts.setObjectName("comboBox_select_sum_for_pmts")
        self.verticalLayout_3.addWidget(self.comboBox_select_sum_for_pmts)
        self.lineEdit_arith_scaler_input = QtWidgets.QLineEdit(self.widget_outer_sum_plot)
        self.lineEdit_arith_scaler_input.setMaximumSize(QtCore.QSize(150, 16777215))
        self.lineEdit_arith_scaler_input.setMouseTracking(False)
        self.lineEdit_arith_scaler_input.setObjectName("lineEdit_arith_scaler_input")
        self.verticalLayout_3.addWidget(self.lineEdit_arith_scaler_input)
        self.label_arith_scaler_set = QtWidgets.QLabel(self.widget_outer_sum_plot)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_arith_scaler_set.sizePolicy().hasHeightForWidth())
        self.label_arith_scaler_set.setSizePolicy(sizePolicy)
        self.label_arith_scaler_set.setMaximumSize(QtCore.QSize(150, 16777215))
        self.label_arith_scaler_set.setObjectName("label_arith_scaler_set")
        self.verticalLayout_3.addWidget(self.label_arith_scaler_set)
        self.comboBox_sum_tr = QtWidgets.QComboBox(self.widget_outer_sum_plot)
        self.comboBox_sum_tr.setObjectName("comboBox_sum_tr")
        self.verticalLayout_3.addWidget(self.comboBox_sum_tr)
        self.browser_fitresults = QtWidgets.QTextBrowser(self.widget_outer_sum_plot)
        self.browser_fitresults.setObjectName("browser_fitresults")
        self.verticalLayout_3.addWidget(self.browser_fitresults)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.horizontalLayout.setStretch(0, 1)
        self.verticalLayout_2.addWidget(self.widget_outer_sum_plot)
        self.tabWidget.addTab(self.tab_sum, "")
        self.tab_timeres = QtWidgets.QWidget()
        self.tab_timeres.setObjectName("tab_timeres")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_timeres)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.spinBox = QtWidgets.QSpinBox(self.tab_timeres)
        self.spinBox.setKeyboardTracking(False)
        self.spinBox.setMinimum(10)
        self.spinBox.setMaximum(10000)
        self.spinBox.setSingleStep(10)
        self.spinBox.setProperty("value", 10)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 0, 2, 1, 1)
        self.pushButton_save_after_scan = QtWidgets.QPushButton(self.tab_timeres)
        self.pushButton_save_after_scan.setObjectName("pushButton_save_after_scan")
        self.gridLayout.addWidget(self.pushButton_save_after_scan, 0, 4, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.tab_timeres)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)
        self.checkBox = QtWidgets.QCheckBox(self.tab_timeres)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout.addWidget(self.checkBox, 0, 3, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout)
        self.splitter = QtWidgets.QSplitter(self.tab_timeres)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setOpaqueResize(True)
        self.splitter.setObjectName("splitter")
        self.widget_tres_plot = QtWidgets.QWidget(self.splitter)
        self.widget_tres_plot.setObjectName("widget_tres_plot")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_tres_plot)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.splitter_4 = QtWidgets.QSplitter(self.widget_tres_plot)
        self.splitter_4.setOrientation(QtCore.Qt.Vertical)
        self.splitter_4.setObjectName("splitter_4")
        self.splitter_2 = QtWidgets.QSplitter(self.splitter_4)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.widget_tres = QtWidgets.QWidget(self.splitter_2)
        self.widget_tres.setObjectName("widget_tres")
        self.widget_proj_t = QtWidgets.QWidget(self.splitter_2)
        self.widget_proj_t.setObjectName("widget_proj_t")
        self.splitter_3 = QtWidgets.QSplitter(self.splitter_4)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.widget_proj_v = QtWidgets.QWidget(self.splitter_3)
        self.widget_proj_v.setObjectName("widget_proj_v")
        self.widget_right_lower_corner = QtWidgets.QWidget(self.splitter_3)
        self.widget_right_lower_corner.setObjectName("widget_right_lower_corner")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget_right_lower_corner)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_5 = QtWidgets.QLabel(self.widget_right_lower_corner)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget_right_lower_corner)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_x_coord = QtWidgets.QLabel(self.widget_right_lower_corner)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_x_coord.sizePolicy().hasHeightForWidth())
        self.label_x_coord.setSizePolicy(sizePolicy)
        self.label_x_coord.setObjectName("label_x_coord")
        self.gridLayout_2.addWidget(self.label_x_coord, 0, 1, 1, 1)
        self.label_y_coord = QtWidgets.QLabel(self.widget_right_lower_corner)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_y_coord.sizePolicy().hasHeightForWidth())
        self.label_y_coord.setSizePolicy(sizePolicy)
        self.label_y_coord.setObjectName("label_y_coord")
        self.gridLayout_2.addWidget(self.label_y_coord, 0, 3, 1, 1)
        self.verticalLayout_5.addWidget(self.splitter_4)
        self.tableWidget_gates = QtWidgets.QTableWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget_gates.sizePolicy().hasHeightForWidth())
        self.tableWidget_gates.setSizePolicy(sizePolicy)
        self.tableWidget_gates.setMinimumSize(QtCore.QSize(0, 0))
        self.tableWidget_gates.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidget_gates.setObjectName("tableWidget_gates")
        self.tableWidget_gates.setColumnCount(7)
        self.tableWidget_gates.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_gates.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_gates.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_gates.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_gates.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_gates.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_gates.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_gates.setHorizontalHeaderItem(6, item)
        self.tableWidget_gates.horizontalHeader().setDefaultSectionSize(80)
        self.verticalLayout_4.addWidget(self.splitter)
        self.tabWidget.addTab(self.tab_timeres, "")
        self.tab_all_pmts = QtWidgets.QWidget()
        self.tab_all_pmts.setObjectName("tab_all_pmts")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.tab_all_pmts)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.widget_all_pmts = QtWidgets.QWidget(self.tab_all_pmts)
        self.widget_all_pmts.setObjectName("widget_all_pmts")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.widget_all_pmts)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.splitter_allpmts = QtWidgets.QSplitter(self.widget_all_pmts)
        self.splitter_allpmts.setOrientation(QtCore.Qt.Vertical)
        self.splitter_allpmts.setObjectName("splitter_allpmts")
        self.widget_all_pmts_plot = QtWidgets.QWidget(self.splitter_allpmts)
        self.widget_all_pmts_plot.setObjectName("widget_all_pmts_plot")
        self.widget_all_pmts_x_y_coords = QtWidgets.QWidget(self.splitter_allpmts)
        self.widget_all_pmts_x_y_coords.setObjectName("widget_all_pmts_x_y_coords")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget_all_pmts_x_y_coords)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_3 = QtWidgets.QLabel(self.widget_all_pmts_x_y_coords)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 6, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.widget_all_pmts_x_y_coords)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 0, 3, 1, 1)
        self.label_y_coor_all_pmts = QtWidgets.QLabel(self.widget_all_pmts_x_y_coords)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_y_coor_all_pmts.sizePolicy().hasHeightForWidth())
        self.label_y_coor_all_pmts.setSizePolicy(sizePolicy)
        self.label_y_coor_all_pmts.setMinimumSize(QtCore.QSize(120, 0))
        self.label_y_coor_all_pmts.setObjectName("label_y_coor_all_pmts")
        self.gridLayout_3.addWidget(self.label_y_coor_all_pmts, 0, 4, 1, 1)
        self.comboBox_all_pmts_sel_tr = QtWidgets.QComboBox(self.widget_all_pmts_x_y_coords)
        self.comboBox_all_pmts_sel_tr.setObjectName("comboBox_all_pmts_sel_tr")
        self.gridLayout_3.addWidget(self.comboBox_all_pmts_sel_tr, 0, 7, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.widget_all_pmts_x_y_coords)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 0, 0, 1, 2)
        self.label_x_coord_all_pmts = QtWidgets.QLabel(self.widget_all_pmts_x_y_coords)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_x_coord_all_pmts.sizePolicy().hasHeightForWidth())
        self.label_x_coord_all_pmts.setSizePolicy(sizePolicy)
        self.label_x_coord_all_pmts.setMinimumSize(QtCore.QSize(120, 0))
        self.label_x_coord_all_pmts.setObjectName("label_x_coord_all_pmts")
        self.gridLayout_3.addWidget(self.label_x_coord_all_pmts, 0, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.widget_all_pmts_x_y_coords)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 0, 8, 1, 1)
        self.lineEdit_sum_all_pmts = QtWidgets.QLineEdit(self.widget_all_pmts_x_y_coords)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_sum_all_pmts.sizePolicy().hasHeightForWidth())
        self.lineEdit_sum_all_pmts.setSizePolicy(sizePolicy)
        self.lineEdit_sum_all_pmts.setMouseTracking(False)
        self.lineEdit_sum_all_pmts.setObjectName("lineEdit_sum_all_pmts")
        self.gridLayout_3.addWidget(self.lineEdit_sum_all_pmts, 0, 10, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(200, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem1, 0, 5, 1, 1)
        self.comboBox_sum_all_pmts = QtWidgets.QComboBox(self.widget_all_pmts_x_y_coords)
        self.comboBox_sum_all_pmts.setObjectName("comboBox_sum_all_pmts")
        self.gridLayout_3.addWidget(self.comboBox_sum_all_pmts, 0, 9, 1, 1)
        self.verticalLayout_7.addWidget(self.splitter_allpmts)
        self.verticalLayout_6.addWidget(self.widget_all_pmts)
        self.tabWidget.addTab(self.tab_all_pmts, "")
        self.tab_pre_post_meas = QtWidgets.QWidget()
        self.tab_pre_post_meas.setObjectName("tab_pre_post_meas")
        self.tabWidget.addTab(self.tab_pre_post_meas, "")
        self.horizontalLayout_2.addWidget(self.tabWidget)
        MainWindow_LiveDataPlotting.setCentralWidget(self.centralwidget)
        self.dockWidget = QtWidgets.QDockWidget(MainWindow_LiveDataPlotting)
        self.dockWidget.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.dockWidget.setObjectName("dockWidget")
        self.widget_progress = QtWidgets.QWidget()
        self.widget_progress.setObjectName("widget_progress")
        self.dockWidget.setWidget(self.widget_progress)
        MainWindow_LiveDataPlotting.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dockWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow_LiveDataPlotting)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1023, 21))
        self.menuBar.setObjectName("menuBar")
        self.menuView = QtWidgets.QMenu(self.menuBar)
        self.menuView.setObjectName("menuView")
        self.menunorm = QtWidgets.QMenu(self.menuView)
        self.menunorm.setObjectName("menunorm")
        self.menufit = QtWidgets.QMenu(self.menuBar)
        self.menufit.setObjectName("menufit")
        self.menu_lineshape = QtWidgets.QMenu(self.menufit)
        self.menu_lineshape.setObjectName("menu_lineshape")
        self.menu_track = QtWidgets.QMenu(self.menufit)
        self.menu_track.setObjectName("menu_track")
        MainWindow_LiveDataPlotting.setMenuBar(self.menuBar)
        self.actionProgress = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.actionProgress.setObjectName("actionProgress")
        self.actionGraph_font_size = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.actionGraph_font_size.setObjectName("actionGraph_font_size")
        self.actionsum = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.actionsum.setCheckable(True)
        self.actionsum.setChecked(True)
        self.actionsum.setObjectName("actionsum")
        self.actionasymmetry = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.actionasymmetry.setCheckable(True)
        self.actionasymmetry.setObjectName("actionasymmetry")
        self.actionscans = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.actionscans.setCheckable(True)
        self.actionscans.setObjectName("actionscans")
        self.actionidentity = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.actionidentity.setCheckable(True)
        self.actionidentity.setChecked(True)
        self.actionidentity.setObjectName("actionidentity")
        self.actionshow_bins = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.actionshow_bins.setCheckable(True)
        self.actionshow_bins.setChecked(True)
        self.actionshow_bins.setObjectName("actionshow_bins")
        self.action_screenshot_to_clipboard = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_screenshot_to_clipboard.setObjectName("action_screenshot_to_clipboard")
        self.action_screenshot_all_to_clipboard = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_screenshot_all_to_clipboard.setObjectName("action_screenshot_all_to_clipboard")
        self.action_update = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_update.setObjectName("action_update")
        self.actionasd = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.actionasd.setObjectName("actionasd")
        self.action_screenshot_to_file = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_screenshot_to_file.setObjectName("action_screenshot_to_file")
        self.action_screenshot_all_to_file = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_screenshot_all_to_file.setObjectName("action_screenshot_all_to_file")
        self.action_fit = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_fit.setObjectName("action_fit")
        self.action_fit_auto = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_fit_auto.setCheckable(True)
        self.action_fit_auto.setEnabled(False)
        self.action_fit_auto.setObjectName("action_fit_auto")
        self.action_voigt = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_voigt.setCheckable(True)
        self.action_voigt.setChecked(True)
        self.action_voigt.setObjectName("action_voigt")
        self.action_lorentz = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_lorentz.setCheckable(True)
        self.action_lorentz.setObjectName("action_lorentz")
        self.action_gauss = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_gauss.setCheckable(True)
        self.action_gauss.setObjectName("action_gauss")
        self.action_fit_cursor = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_fit_cursor.setCheckable(True)
        self.action_fit_cursor.setObjectName("action_fit_cursor")
        self.action_fit_config = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_fit_config.setEnabled(False)
        self.action_fit_config.setObjectName("action_fit_config")
        self.action_track0 = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_track0.setObjectName("action_track0")
        self.action_clear = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_clear.setObjectName("action_clear")
        self.action_set_limits = QtWidgets.QAction(MainWindow_LiveDataPlotting)
        self.action_set_limits.setCheckable(True)
        self.action_set_limits.setObjectName("action_set_limits")
        self.menunorm.addAction(self.actionidentity)
        self.menunorm.addAction(self.actionscans)
        self.menuView.addAction(self.action_update)
        self.menuView.addSeparator()
        self.menuView.addAction(self.action_screenshot_to_clipboard)
        self.menuView.addAction(self.action_screenshot_all_to_clipboard)
        self.menuView.addAction(self.action_screenshot_to_file)
        self.menuView.addAction(self.action_screenshot_all_to_file)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionProgress)
        self.menuView.addAction(self.actionGraph_font_size)
        self.menuView.addSeparator()
        self.menuView.addAction(self.menunorm.menuAction())
        self.menuView.addAction(self.actionshow_bins)
        self.menu_lineshape.addAction(self.action_lorentz)
        self.menu_lineshape.addAction(self.action_gauss)
        self.menu_lineshape.addAction(self.action_voigt)
        self.menu_track.addAction(self.action_track0)
        self.menufit.addAction(self.action_fit)
        self.menufit.addAction(self.action_clear)
        self.menufit.addAction(self.action_fit_auto)
        self.menufit.addAction(self.action_fit_cursor)
        self.menufit.addAction(self.action_set_limits)
        self.menufit.addSeparator()
        self.menufit.addAction(self.menu_lineshape.menuAction())
        self.menufit.addAction(self.menu_track.menuAction())
        self.menufit.addSeparator()
        self.menufit.addAction(self.action_fit_config)
        self.menuBar.addAction(self.menuView.menuAction())
        self.menuBar.addAction(self.menufit.menuAction())

        self.retranslateUi(MainWindow_LiveDataPlotting)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow_LiveDataPlotting)

    def retranslateUi(self, MainWindow_LiveDataPlotting):
        _translate = QtCore.QCoreApplication.translate
        MainWindow_LiveDataPlotting.setWindowTitle(_translate("MainWindow_LiveDataPlotting", "MainWindow"))
        self.label.setText(_translate("MainWindow_LiveDataPlotting", "sum over:"))
        self.label_arith_scaler_set.setText(_translate("MainWindow_LiveDataPlotting", "TextLabel"))
        self.browser_fitresults.setHtml(_translate("MainWindow_LiveDataPlotting", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">Fit results:</span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_sum), _translate("MainWindow_LiveDataPlotting", "sum"))
        self.pushButton_save_after_scan.setText(_translate("MainWindow_LiveDataPlotting", "Save again after scan"))
        self.label_4.setText(_translate("MainWindow_LiveDataPlotting", "rebinning [ns]"))
        self.checkBox.setText(_translate("MainWindow_LiveDataPlotting", "apply for all tracks"))
        self.label_5.setText(_translate("MainWindow_LiveDataPlotting", "y:"))
        self.label_2.setText(_translate("MainWindow_LiveDataPlotting", "x:"))
        self.label_x_coord.setText(_translate("MainWindow_LiveDataPlotting", "-0.000"))
        self.label_y_coord.setText(_translate("MainWindow_LiveDataPlotting", "-0.000"))
        self.tableWidget_gates.setSortingEnabled(False)
        item = self.tableWidget_gates.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow_LiveDataPlotting", "track"))
        item = self.tableWidget_gates.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow_LiveDataPlotting", "scaler"))
        item = self.tableWidget_gates.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow_LiveDataPlotting", "v_min [V]"))
        item = self.tableWidget_gates.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow_LiveDataPlotting", "v_max [V]"))
        item = self.tableWidget_gates.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow_LiveDataPlotting", "t_min [us]"))
        item = self.tableWidget_gates.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow_LiveDataPlotting", "t_max [us]"))
        item = self.tableWidget_gates.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow_LiveDataPlotting", "show"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_timeres), _translate("MainWindow_LiveDataPlotting", "time resolved"))
        self.label_3.setText(_translate("MainWindow_LiveDataPlotting", "track:"))
        self.label_7.setText(_translate("MainWindow_LiveDataPlotting", "y:"))
        self.label_y_coor_all_pmts.setText(_translate("MainWindow_LiveDataPlotting", "y_value"))
        self.label_8.setText(_translate("MainWindow_LiveDataPlotting", "x:"))
        self.label_x_coord_all_pmts.setText(_translate("MainWindow_LiveDataPlotting", "x_value"))
        self.label_6.setText(_translate("MainWindow_LiveDataPlotting", "sum over:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_all_pmts), _translate("MainWindow_LiveDataPlotting", "all pmts"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_pre_post_meas), _translate("MainWindow_LiveDataPlotting", "pre/during/post scan measurements"))
        self.dockWidget.setWindowTitle(_translate("MainWindow_LiveDataPlotting", "progress"))
        self.menuView.setTitle(_translate("MainWindow_LiveDataPlotting", "view"))
        self.menunorm.setTitle(_translate("MainWindow_LiveDataPlotting", "normalize"))
        self.menufit.setTitle(_translate("MainWindow_LiveDataPlotting", "fit"))
        self.menu_lineshape.setTitle(_translate("MainWindow_LiveDataPlotting", "Voigt"))
        self.menu_track.setTitle(_translate("MainWindow_LiveDataPlotting", "track0"))
        self.actionProgress.setText(_translate("MainWindow_LiveDataPlotting", "progress"))
        self.actionProgress.setShortcut(_translate("MainWindow_LiveDataPlotting", "Ctrl+P"))
        self.actionGraph_font_size.setText(_translate("MainWindow_LiveDataPlotting", "graph font size"))
        self.actionsum.setText(_translate("MainWindow_LiveDataPlotting", "sum"))
        self.actionasymmetry.setText(_translate("MainWindow_LiveDataPlotting", "asymmetry"))
        self.actionscans.setText(_translate("MainWindow_LiveDataPlotting", "# scans"))
        self.actionidentity.setText(_translate("MainWindow_LiveDataPlotting", "identity"))
        self.actionshow_bins.setText(_translate("MainWindow_LiveDataPlotting", "show bins"))
        self.actionshow_bins.setShortcut(_translate("MainWindow_LiveDataPlotting", "Ctrl+B"))
        self.action_screenshot_to_clipboard.setText(_translate("MainWindow_LiveDataPlotting", "screenshot"))
        self.action_screenshot_to_clipboard.setShortcut(_translate("MainWindow_LiveDataPlotting", "Ctrl+C"))
        self.action_screenshot_all_to_clipboard.setText(_translate("MainWindow_LiveDataPlotting", "screenshot all"))
        self.action_screenshot_all_to_clipboard.setShortcut(_translate("MainWindow_LiveDataPlotting", "Ctrl+Shift+C"))
        self.action_update.setText(_translate("MainWindow_LiveDataPlotting", "update"))
        self.action_update.setShortcut(_translate("MainWindow_LiveDataPlotting", "F5"))
        self.actionasd.setText(_translate("MainWindow_LiveDataPlotting", "asd"))
        self.action_screenshot_to_file.setText(_translate("MainWindow_LiveDataPlotting", "save screenshot"))
        self.action_screenshot_to_file.setShortcut(_translate("MainWindow_LiveDataPlotting", "Ctrl+S"))
        self.action_screenshot_all_to_file.setText(_translate("MainWindow_LiveDataPlotting", "save screenshot all"))
        self.action_screenshot_all_to_file.setShortcut(_translate("MainWindow_LiveDataPlotting", "Ctrl+Shift+S"))
        self.action_fit.setText(_translate("MainWindow_LiveDataPlotting", "fit"))
        self.action_fit.setShortcut(_translate("MainWindow_LiveDataPlotting", "Ctrl+F"))
        self.action_fit_auto.setText(_translate("MainWindow_LiveDataPlotting", "auto"))
        self.action_voigt.setText(_translate("MainWindow_LiveDataPlotting", "Voigt"))
        self.action_lorentz.setText(_translate("MainWindow_LiveDataPlotting", "Lorentz"))
        self.action_gauss.setText(_translate("MainWindow_LiveDataPlotting", "Gauss"))
        self.action_fit_cursor.setText(_translate("MainWindow_LiveDataPlotting", "set center"))
        self.action_fit_cursor.setShortcut(_translate("MainWindow_LiveDataPlotting", "Ctrl+G"))
        self.action_fit_config.setText(_translate("MainWindow_LiveDataPlotting", "config ..."))
        self.action_track0.setText(_translate("MainWindow_LiveDataPlotting", "track0"))
        self.action_clear.setText(_translate("MainWindow_LiveDataPlotting", "clear"))
        self.action_clear.setShortcut(_translate("MainWindow_LiveDataPlotting", "Ctrl+Shift+F"))
        self.action_set_limits.setText(_translate("MainWindow_LiveDataPlotting", "set x-limits"))
        self.action_set_limits.setShortcut(_translate("MainWindow_LiveDataPlotting", "Ctrl+H"))
