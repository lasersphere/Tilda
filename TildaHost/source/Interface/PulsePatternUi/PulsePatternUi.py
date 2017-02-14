"""

Created on '16.01.2017'

@author:'simkaufm'

"""

import ast
import functools
import os
import sys
from copy import deepcopy

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

import Application.Config as CfgMain
import PyQtGraphPlotter as Pgplot
import Service.FileOperations.FolderAndFileHandling as FileHandl
from Interface.PulsePatternUi.PpgPeriodicWidgUi import PpgPeriodicWidgUi
from Interface.PulsePatternUi.PpgSimpleWidgUi import PpgSimpleWidgUi
from Interface.PulsePatternUi.Ui_PulsePattern import Ui_PulsePatternWin


class PulsePatternUi(QtWidgets.QMainWindow, Ui_PulsePatternWin):
    pulse_pattern_status = QtCore.pyqtSignal(str)
    cmd_list_signal = QtCore.pyqtSignal(list, str)

    def __init__(self, active_iso, track_name, main_gui, track_gui=None):
        super(PulsePatternUi, self).__init__()
        self.gui_cmd_list = []  # list of commands in gui, always updated when self.cmd_list_from_gui() is called.
        self.ticks_per_us = 100  # standard with 100 MHz, change if changed on hardware.
        self.setupUi(self)
        self.setWindowTitle('pulse pattern of %s %s' % (active_iso, track_name))
        self.active_iso = active_iso
        self.track_name = track_name
        self.main_gui = main_gui
        self.track_gui = track_gui
        self.periodic_widg = None
        self.simple_widg = None
        if main_gui is not None:
            # close other open ppg windows here
            if self.main_gui.pulse_pattern_win is not None:
                self.main_gui.pulse_pattern_win.close()

        ''' state related'''
        self.ppg_state = None
        self.rcvd_state('not initialised')
        if CfgMain._main_instance is not None:
            CfgMain._main_instance.ppg_state_callback(self.pulse_pattern_status)
            self.pulse_pattern_status.connect(self.rcvd_state)

        self.listWidget_cmd_list.setDragDropMode(self.listWidget_cmd_list.InternalMove)
        self.listWidget_cmd_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.pushButton_remove_selected.clicked.connect(self.remove_selected)
        self.pushButton_add_cmd.clicked.connect(self.add_before)
        self.pushButton_load_txt.clicked.connect(self.load_from_text)
        self.pushButton_save_txt.clicked.connect(self.save_to_text_file)

        self.pushButton_stop.clicked.connect(self.stop_pulse_pattern)
        self.pushButton_run_pattern.clicked.connect(self.run)
        self.pushButton_close.clicked.connect(self.close_and_confirm)

        ''' keyboard shortcuts '''
        QtWidgets.QShortcut(QtGui.QKeySequence("DEL"), self, self.remove_selected)
        QtWidgets.QShortcut(QtGui.QKeySequence("-"), self, self.remove_selected)
        QtWidgets.QShortcut(QtGui.QKeySequence("A"), self, self.add_before)
        QtWidgets.QShortcut(QtGui.QKeySequence("+"), self, self.add_before)
        QtWidgets.QShortcut(QtGui.QKeySequence("F5"), self, self.update_gr_v)
        QtWidgets.QShortcut(QtGui.QKeySequence("CTRL+R"), self, self.run)
        QtWidgets.QShortcut(QtGui.QKeySequence("CTRL+S"), self, self.stop_pulse_pattern)
        QtWidgets.QShortcut(QtGui.QKeySequence("CTRL+Q"), self, self.close_and_confirm)

        ''' help '''
        self.actionHelp.triggered.connect(self.open_help)

        ''' graphical view related '''
        self.listWidget_cmd_list.currentTextChanged.connect(self.update_gr_v)
        self.listWidget_cmd_list.currentRowChanged.connect(self.update_gr_v)
        self.listWidget_cmd_list.itemSelectionChanged.connect(self.update_gr_v)

        # self.listWidget_cmd_list.itemChanged.connect(self.update_gr_v)

        self.ch_pos_dict = {}
        self.gr_v_cursor_one = None
        self.gr_v_cursor_two = None
        self.gr_v_clicks = 0
        self.add_graph_view()

        ''' rcv cmd list from one of those: '''
        self.cmd_list_signal.connect(self.cmd_list_to_gui)

        ''' peridoic widget related '''
        self.periodic_widg = PpgPeriodicWidgUi(self, self.cmd_list_signal)
        self.tab_periodic_pattern.layout().addWidget(self.periodic_widg)

        ''' simple widget related '''
        self.simple_widg = PpgSimpleWidgUi(self, self.cmd_list_signal)
        self.tab_simple.layout().addWidget(self.simple_widg)

        self.show()
        self.tabWidget_periodic_pattern.setCurrentIndex(0)

    ''' cmd list related: '''
    def cmd_list_to_gui(self, cmd_list, caller_str=None, update_gr_v=True):
        """ write a list of str cmd to the gui """
        # remove all items tht were already in the list.
        self.listWidget_cmd_list.clear()
        self.listWidget_cmd_list.addItems(cmd_list)
        for i in range(self.listWidget_cmd_list.count()):
            self.listWidget_cmd_list.item(i).setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsDragEnabled)
        if update_gr_v:
            self.update_gr_v(caller_str)

    def cmd_list_from_gui(self):
        """ return a list of all cmds in the gui """
        items = []
        stop_index = -1
        num_of_items = self.listWidget_cmd_list.count()
        for i in range(num_of_items):
            cmd_str = self.listWidget_cmd_list.item(i).text()
            items.append(cmd_str)
            stop_index = i if 'stop' in cmd_str else stop_index
        if stop_index != num_of_items - 1 and stop_index >= 0:  # stop not at end of list
            items.insert(num_of_items - 1, items.pop(stop_index))
            self.cmd_list_to_gui(items, update_gr_v=False)
        self.gui_cmd_list = items
        return items

    def remove_selected(self):
        """ if an item is selected, remove this from the list. """
        items = self.listWidget_cmd_list.selectedItems()
        [self.listWidget_cmd_list.takeItem(self.listWidget_cmd_list.row(each)) for each in items]

    def add_before(self):
        """ add copy of selected command before the selected one. If none selected, place at end. """
        # cur_row = self.listWidget_cmd_list.currentRow()
        items = self.listWidget_cmd_list.selectedItems()
        if items:
            items_text = [each.text() for each in items]
            rows = [self.listWidget_cmd_list.row(each) for each in items]
            sorted_selection = list(sorted(zip(rows, items, items_text)))
            rows, items, items_text = zip(*sorted_selection)
        else:
            items_text = ['$cmd::1.0::0::0']
            rows = [0]
        self.listWidget_cmd_list.insertItems(rows[0], items_text)
        new_rows = range(rows[0], rows[0] + len(rows))
        new_items = [self.listWidget_cmd_list.item(row) for row in new_rows]
        for each_item in new_items:
            each_item.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable |
                QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsDragEnabled)

    def rcvd_state(self, state_str):
        """ when state of ppg changes this signal is emitted """
        self.ppg_state = state_str
        self.label_ppg_state.setText(state_str)

    """ cmd conversions (copied from ppg) """

    def convert_single_comand(self, cmd_str, ticks_per_us=None):
        """
        converts a single command to a tuple of length 4
        :param cmd_str: str, "$cmd::time_us::DIO0-39::DIO40-79"
            -> cmd: stop(0), jump(1), wait(2), time(3)
            time_us: float, time in us

        :param ticks_per_us: int, ticks per us, usually 100 (=100MHz), None for readout from fpga
        :return: numpy array, [int_cmd_num, int_time_in_ticks_or_other, int_DIO0-39, int_DIO40-79]
        """
        if ticks_per_us is None:
            ticks_per_us = self.ticks_per_us
        cmd_dict = {'$stop': 0, '$jump': 1, '$wait': 2, '$time': 3}
        cmd_list = cmd_str.split('::')
        if len(cmd_list) == 4:
            try:
                cmd_list[0] = cmd_dict.get(cmd_list[0], -1)
                for i in range(1, 4):
                    cmd_list[i] = ast.literal_eval(cmd_list[i])
                if cmd_list[0] == cmd_dict['$time']:  # $time
                    cmd_list[1] = cmd_list[1] * ticks_per_us
                cmd_list = np.asarray(cmd_list, dtype=np.int32)
                return cmd_list
            except Exception as e:
                print('error: could not convert the command: %s, error is: %s' % (cmd_str, e))
        else:
            return [-1] * 4

    def convert_list_of_cmds(self, cmd_list, ticks_per_us=None):
        """
        will convert a list of commands to a numpy array which can be fed to the fpga
        :param cmd_list: list of str, each cmd str looks like:
        cmd_str: str, "$cmd::time_us::DIO0-39::DIO40-79"
            -> cmd: stop(0), jump(1), wait(2), time(3)
            time_us: float, time in us
        :param ticks_per_us: int, ticks per us, usually 100 (=100MHz), None for readout from fpga
        :return: numpy array
        """
        ret_arr = np.zeros(0, dtype=np.int32)
        if ticks_per_us is None:
            ticks_per_us = self.ticks_per_us
        for each_cmd in cmd_list:
            ret_arr = np.append(ret_arr, self.convert_single_comand(each_cmd, ticks_per_us))
        return ret_arr

    def convert_np_arr_of_cmd_to_list_of_cmds(self, np_arr_cmds, ticks_per_us=None):
        ret = []
        if ticks_per_us is None:
            ticks_per_us = self.ticks_per_us
        for i in range(0, len(np_arr_cmds), 4):
            ret.append(self.convert_int_arr_to_singl_cmd(np_arr_cmds[i:i + 4], ticks_per_us))
        return ret

    def convert_int_arr_to_singl_cmd(self, int_arr, ticks_per_us=None):
        """
        will convert an array/tuple of integers with the 4 needed elements
         for one command to a string as like:
        [3, 100, 1, 0] -> "$cmd::time_us::DIO0-39::DIO40-79"
        :param int_arr: array of length 4 conatining ints
        :param ticks_per_us:
        :return:
        """
        if ticks_per_us is None:
            ticks_per_us = self.ticks_per_us
        cmd_dict = {0: '$stop', 1: '$jump', 2: '$wait', 3: '$time'}
        ret_cmd_str = '%s::%.2f::%s::%s' % (
            cmd_dict.get(int_arr[0], 'error'), int_arr[1] / ticks_per_us, int_arr[2], int_arr[3]
        )
        return ret_cmd_str

    def check_cmd_list_credibility(self, cmd_str_list):
        """ check if all patterns are long enough """
        fpga_cmd_transfer_time_ticks = 12  # ticks the fpga need to transfer the data mem->fifo->set_out
        wait_cmd_int = 2  # $wait is int 2
        time_cmd_int = 3  # $time is int 3
        cmd_np_list = self.convert_list_of_cmds(cmd_str_list)
        cmd_np_list = np.reshape(cmd_np_list, (len(cmd_str_list), 4))
        if wait_cmd_int in cmd_np_list[:, 0]:
            return True
        buffer_time_list = [
            each[1] - fpga_cmd_transfer_time_ticks for each in cmd_np_list if each[0] == time_cmd_int]
        buffer_time = sum(buffer_time_list)
        if buffer_time < 0:
            dial = QtWidgets.QDialog(self)
            QtWidgets.QMessageBox.warning(
                dial, 'Problem with pattern!',
                'Your pattern is not executable, because the single commands are'
                ' shorter then the command transfer time (120us).\n'
                'You can do one of the following:\n\n'
                'o increase the pulse duration of each pulse\n'
                'o introduce a longer pulse (the exec. time of this pulse is used to transfer the data)\n'
                'o an external trigger (->$wait) will (probably) give'
                ' the fpga enough time to transfer the pattern\n'
                '\n\nThe pattern needs %.2f us more.' % (abs(buffer_time) / 100))
            return False
        else:
            return True


    ''' saving and loading '''

    def load_from_text(self, button=None, txt_path=None):
        """ load the list of cmds from an existing text file """
        parent = QtWidgets.QFileDialog(self)
        # print('loading')
        if txt_path is None:
            if CfgMain._main_instance is None:
                start_path = os.path.basename(__file__)
            else:
                start_path = CfgMain._main_instance.working_directory
            txt_path, ok = QtWidgets.QFileDialog.getOpenFileName(
                parent, "select ppg .txt file", start_path, '*.txt')
        if txt_path:
            list_of_cmds = FileHandl.load_from_text_file(txt_path)
            self.cmd_list_to_gui(list_of_cmds)
            return txt_path

    def save_to_text_file(self):
        """ save the current settings to a text file """
        parent = QtWidgets.QFileDialog(self)
        if CfgMain._main_instance is None:
            start_path = os.path.basename(__file__)
        else:
            start_path = CfgMain._main_instance.working_directory
        path, ok = QtWidgets.QFileDialog.getSaveFileName(
            parent, "select ppg .txt file", start_path, '*.txt')
        if path:
            FileHandl.save_txt_file_line_by_line(path, self.cmd_list_from_gui())
            return path

    ''' periodic related '''

    def load_periodic(self, per_list):
        self.tabWidget_periodic_pattern.setCurrentIndex(1)
        print('loading periodic list: %s' % per_list)
        self.periodic_widg.setup_from_list(per_list)

    ''' simple related '''

    def load_simple_dict(self, simple_dict):
        """ load a simple dict to the simple tab """
        self.tabWidget_periodic_pattern.setCurrentIndex(2)
        print('loading simple dict: %s ' % simple_dict)
        self.simple_widg.load_from_simple_dict(simple_dict)

    ''' help '''
    def open_help(self):
        mb = QtWidgets.QMessageBox(self)
        QtWidgets.QMessageBox.information(
            mb, 'ppg help',
            'from: \n F. Ziegler et al., A new Pulse-Pattern Generator based on Lab'
            'VIEW-FPGA, Nucl. Instr. Meth. A 679 (2012) 1-6:\n\n'
            'Command\t Example\t Description\n'
            '------------------------------------------------------------\n'
            '$time\t $time::1000::123::123\t command::time[ms] ::DO0-31::DO32-63\n'
            '$wait\t $wait::1::123::123\t command::DI0-7::DO0-31::DO32-63\n'
            '($jump\t $jump::0::500::500\t command::address::unused::iterationnumber)*\n'
            '$stop\t $stop::0::123::123\t command::unused::DO0-31::DO32-63\n\n'
            '* jump commands currently not supported.'
        )

    """ graphical displaying """

    def add_graph_view(self):
        try:
            layout = QtWidgets.QVBoxLayout()
            self.gr_v_widg, self.gr_v_plt_itm = Pgplot.create_x_y_widget(y_label='channel', x_label='time [us]')
            self.gr_v_x_ax = self.gr_v_plt_itm.getAxis('left')

            layout.addWidget(self.gr_v_widg)
            x_pos_l = QtWidgets.QLabel('time [us]: ')
            y_pos_l = QtWidgets.QLabel('CH: ')
            self.x_pos = QtWidgets.QLabel()
            self.act_ch_label = QtWidgets.QLabel()
            self.delta_t_label = QtWidgets.QLabel()

            self.y_pos = QtWidgets.QLabel()
            layout2 = QtWidgets.QHBoxLayout()
            layout2.addWidget(x_pos_l)
            layout2.addWidget(self.x_pos)
            layout2.addWidget(QtWidgets.QLabel('active Channels (t):'))
            layout2.addWidget(self.act_ch_label)
            layout2.addWidget(QtWidgets.QLabel('delta t [us]:'))
            layout2.addWidget(self.delta_t_label)
            layout2.addWidget(y_pos_l)
            layout2.addWidget(self.y_pos)
            layout.addItem(layout2)
            self.plt_proxy = Pgplot.create_proxy(signal=self.gr_v_plt_itm.scene().sigMouseMoved,
                                     slot=functools.partial(self.mouse_moved, self.gr_v_plt_itm.vb),
                                     rate_limit=60)
            self.gr_v_mouse_click_proxy = Pgplot.create_proxy(signal=self.gr_v_plt_itm.scene().sigMouseClicked,
                                                 slot=functools.partial(self.mouse_clicked, self.gr_v_plt_itm.vb),
                                                 rate_limit=60)
            self.gr_v_cursor = Pgplot.create_infinite_line(0, pen=Pgplot.create_pen(125, 125, 125, width=0.5))
            self.gr_v_plt_itm.addItem(self.gr_v_cursor)
            self.widget_graph_view.setLayout(layout)
        except Exception as e:
            print('error while adding graphical view: %s' % e)

    def mouse_moved(self, viewbox, evt):
        """ called when mouse is moved within the gr_v_plt_item """
        point = viewbox.mapSceneToView(evt[0])
        self.print_point(point)

    def print_point(self, point):
        """
        display the point to the GUI
        """
        x_str = '%.2f' % point.x()
        self.x_pos.setText(x_str)
        self.gr_v_cursor.setPos(point.x())
        self.y_pos.setText(self.get_ch_from_y_coord(point.y()))
        hi_ch_lis, hi_ch_str = self.get_high_channels_at_time(point.x())
        self.act_ch_label.setText(hi_ch_str)

    def get_high_channels_at_time(self, time):
        """
        go through the pos lists in self.ch_pos_dict for each channel and
        check if this channel is active at this time if so, append it to the return list.
        :param time: float, time in us
        :return: tpl, (list of strings with high ch_names, str with ch names seperated by | )
        """
        high_channels = []
        hi_ch_str = ''
        for ch_name, ch_dict in sorted(self.ch_pos_dict.items()):
            pos = ch_dict.get('pos', [])
            ch_num = int(ch_name[2:])
            ch_num = -(ch_num + 1) if 'DI' in ch_name else ch_num
            if pos:
                for i, hi_lo_tpl in enumerate(pos):
                    if i < (len(pos) - 1):
                        if pos[i][0] < time < pos[i + 1][0] and hi_lo_tpl[1] == ch_num + 0.5:
                            high_channels.append(ch_name)
                            hi_ch_str += '| %s ' % ch_name
        return high_channels, hi_ch_str[1:]

    def mouse_clicked(self, viewbox, evt):
        """ called on mouseclick within the graph_view_box """
        point = viewbox.mapSceneToView(evt[0].scenePos())
        self.drop_cursor(point)

    def drop_cursor(self, point):
        """  for each click add a """
        self.gr_v_clicks += 1
        new_pos = point.x()
        if self.gr_v_clicks % 2:  # odd number of clicks -> work on cursor one
            if self.gr_v_cursor_one is None:
                self.gr_v_cursor_one = Pgplot.create_infinite_line(new_pos, pen=Pgplot.create_pen(255, 128, 0, width=0.5),
                                                                   movable=True)
                self.gr_v_cursor_one.sigPositionChangeFinished.connect(self.cursor_moved)
                self.gr_v_plt_itm.addItem(self.gr_v_cursor_one)
            else:
                self.gr_v_cursor_one.setPos(new_pos)
        else:
            if self.gr_v_cursor_two is None:
                self.gr_v_cursor_two = Pgplot.create_infinite_line(new_pos, pen=Pgplot.create_pen(255, 128, 0, width=0.5),
                                                                   movable=True)
                self.gr_v_cursor_two.sigPositionChangeFinished.connect(self.cursor_moved)
                self.gr_v_plt_itm.addItem(self.gr_v_cursor_two)
            else:
                self.gr_v_cursor_two.setPos(new_pos)
        self.cursor_moved()

    def cursor_moved(self):
        """ checks cursor position and gives the time delta to the gui """
        cursor_one_pos = None
        cursor_two_pos = None
        t_dif = 0
        if self.gr_v_cursor_one:
            cursor_one_pos = self.gr_v_cursor_one.value()
        if self.gr_v_cursor_two:
            cursor_two_pos = self.gr_v_cursor_two.value()
        if cursor_one_pos is not None and cursor_two_pos is not None:
            t_dif = abs(cursor_one_pos - cursor_two_pos)
        elif cursor_one_pos is not None:
            t_dif = abs(cursor_one_pos - self.gr_v_cursor.value())
        elif cursor_two_pos is not None:
            t_dif = abs(cursor_two_pos - self.gr_v_cursor.value())
        self.delta_t_label.setText('%.2f' % t_dif)

    def get_ch_from_y_coord(self, y_coord):
        """ this will print the CH at y_coord to the gui, if there is a CH at this y coord """
        if y_coord < 0:  # trigger line
            y_coord += 1
            ret = 'DI'
        else:
            ret = 'DO'
        abs_y = abs(y_coord)
        if 0 < abs_y % 1 < 0.5:
            ret += '%d' % int(abs_y)
        else:
            ret = ''
        return ret

    def update_gr_v(self, caller_str=None):
        """ updates the graphic view and adds a line for each item """
        try:
            old_list = deepcopy(self.gui_cmd_list)
            new_list = self.cmd_list_from_gui()
            stop_in_cmds_list = [(i, each) for i, each in enumerate(new_list) if 'stop' in each]
            stop_in_cmds = len(stop_in_cmds_list) != 0
            if old_list != new_list:
                # print('updating praphical view, caller: %s of type: %s' % (caller_str, type(caller_str)))
                # manual change on the cmd list -> therefore clear all other
                # widgets because their data is probably now corrupted
                # if caller str is specified, only clear all other widgets.
                if isinstance(caller_str, str):
                    if caller_str in ['simple', 'periodic']:
                        caller_str = [caller_str]
                    else:
                        caller_str = ['simple', 'periodic']
                else:
                    # print('caller is not a string but of type: %s' % type(caller_str))
                    caller_str = ['simple', 'periodic']
                if self.periodic_widg is not None and 'simple' in caller_str:
                    # print('clearing periodic table, due to changes in list view')
                    self.periodic_widg.list_view_was_changed()
                if self.simple_widg is not None and 'periodic' in caller_str:
                    # print('clearing simple tab, due to changes in list view')
                    self.simple_widg.list_view_was_changed()
                # print('updating graphics view')
                self.ch_pos_dict, valid_lines = self.get_gr_v_pos_from_list_of_cmds(
                    self.cmd_list_from_gui(), ret_dict=self.ch_pos_dict)
                lines_to_remove = [each for each in self.ch_pos_dict.keys() if each not in valid_lines]
                self.remove_lines(self.gr_v_plt_itm, lines_to_remove)
                self.add_lines(self.gr_v_plt_itm)
            else:
                pass
                # print('nope not a new list, not redrawing here')
            selected_rows = [self.listWidget_cmd_list.row(each) for each in self.listWidget_cmd_list.selectedItems()]
            self.highlight_selected_list_view_item(selected_rows, stop_in_cmds)
        except Exception as e:
            print('error while updating graphical view: %s' % e)

    def remove_lines(self, plt_item, remove_list):
        ch_pos_dict = self.ch_pos_dict
        for each in remove_list:
            # print('removing: %s' % each)
            if ch_pos_dict.get(each, {}).get('line', None):
                plt_item.removeItem(ch_pos_dict[each]['line'])
                plt_item.removeItem(ch_pos_dict[each]['low_line'])
                plt_item.removeItem(ch_pos_dict[each]['high_line'])
                self.ch_pos_dict.pop(each)

    def add_lines(self, plt_item):
        """ add a roi polyline line for every channel in ch_pos_dict """
        ch_pos_dict = self.ch_pos_dict
        major_ticks = []
        for ch, ch_dict in ch_pos_dict.items():
            ch_low_val = int(ch_dict['pos'][0][1])  # 0.5 -> 0 etc.
            ch_high_val = ch_low_val + 0.5
            major_ticks.append((ch_low_val, ch + '_low'))
            major_ticks.append((ch_high_val, ch + '_high'))
            if ch_dict.get('line', None) is not None:
                ch_dict['line'].blockSignals(True)
                ch_dict['line'].setPoints(ch_dict['pos'])
                ch_dict['line'].blockSignals(False)
            else:
                ch_dict['line'] = Pgplot.create_roi_polyline(ch_dict['pos'], movable=False)
                if 'DO' in ch:  # outputs blue
                    pen = Pgplot.pg.mkPen('b', width=3)
                else:  # triggers red
                    pen = Pgplot.pg.mkPen('r', width=3)
                ch_dict['line'].setPen(pen)
                # create custom grid in order not to draw minor ticks
                ch_dict['low_line'] = Pgplot.create_infinite_line(
                    ch_low_val, angle=0, pen=Pgplot.create_pen(125, 125, 125, style=QtCore.Qt.DashLine))
                ch_dict['high_line'] = Pgplot.create_infinite_line(
                    ch_high_val, angle=0, pen=Pgplot.create_pen(125, 125, 125, style=QtCore.Qt.DashLine))
                plt_item.addItem(ch_dict['line'])
                plt_item.addItem(ch_dict['low_line'])
                plt_item.addItem(ch_dict['high_line'])
        self.gr_v_x_ax.setTicks([major_ticks, []])

    def get_gr_v_pos_from_list_of_cmds(self, list_of_cmds=None, ret_dict={}):
        """ get a dictionary for all active channels containing a list of positions when high or low. """
        if list_of_cmds is None:
            list_of_cmds = self.cmd_list_from_gui()
        valid_lines = []  # only add edited ones, other must be deleted
        ticks_per_us = 100
        int_cmd_list = self.convert_list_of_cmds(list_of_cmds, ticks_per_us)
        int_cmd_list = int_cmd_list.reshape((int_cmd_list.size / 4, 4))
        # get highest ch number
        ch_int = np.max(int_cmd_list[:, 2])
        max_dio_0to31 = np.int(np.log2(ch_int)) if ch_int > 1 else 1
        # max_dio_32to63 = int(np.log2(np.max(int_cmd_list[:, 3])))
        # reset trigger positions:
        for key, val in ret_dict.items():
            if 'DI' in key:
                if val.get('pos', False):
                    val['pos'] = {}

        for ch in range(0, max_dio_0to31 + 1):
            ch_bit = 2 ** ch
            low = ch
            high = ch + 0.5
            ch_pos_list = []
            time = 0
            for each_cmd in int_cmd_list:
                # for each command two points are added.
                if each_cmd[0] == 0:  # $stop
                    y_pos = high if ch_bit & each_cmd[2] != 0 else low
                    ch_pos_list.append([time, y_pos])
                    time += 0.5  # add another us at the stop
                    ch_pos_list.append([time, y_pos])
                    ch_pos_list.append([time, y_pos])
                    time += 0.5  # add another us at the stop
                    ch_pos_list.append([time, y_pos])
                elif each_cmd[0] == 1:  # $jump
                    pass

                elif each_cmd[0] == 2:  # $wait
                    y_pos = high if ch_bit & each_cmd[2] != 0 else low
                    ch_pos_list.append([time, y_pos])
                    time += 1  # draw 1 us before ris edge of trigger
                    ch_pos_list.append([time, y_pos])
                    # on each wait cmd check which channels are active
                    if ch == 0:  # only do this for first call
                        tr_ch_max = np.int(np.log2(each_cmd[1])) if each_cmd[1] > 1 else 1
                        for tr_ch in range(0, tr_ch_max + 1):
                            tr_ch_bit = 2 ** tr_ch
                            tr_low = -1 - tr_ch
                            tr_high = -0.5 - tr_ch
                            tr_active = tr_ch_bit & each_cmd[1] != 0
                            if tr_active:
                                tr_pos = [
                                    [time - 1, tr_low], [time, tr_low],
                                    [time, tr_high], [time + 0.5, tr_high],
                                    [time + 0.5, tr_low], [time + 2, tr_low],
                                ]
                                if ret_dict.get('DI%s' % tr_ch, None) is None:
                                    ret_dict['DI%s' % tr_ch] = {}
                                if ret_dict['DI%s' % tr_ch].get('pos', []):
                                    ret_dict['DI%s' % tr_ch]['pos'] += tr_pos
                                else:
                                    ret_dict['DI%s' % tr_ch]['pos'] = tr_pos
                                valid_lines.append('DI%s' % tr_ch)

                elif each_cmd[0] == 3:  # $time
                    y_pos = high if ch_bit & each_cmd[2] != 0 else low
                    ch_pos_list.append([time, y_pos])
                    time += (each_cmd[1] / ticks_per_us)
                    ch_pos_list.append([time, y_pos])
            if ch_pos_list:
                if ret_dict.get('DO%s' % ch, None) is None:
                    ret_dict['DO%s' % ch] = {}
                ret_dict['DO%s' % ch]['pos'] = ch_pos_list
                valid_lines.append('DO%s' % ch)
        return ret_dict, valid_lines

    def highlight_selected_list_view_item(self, indice_list, stop_in_list=True):
        """ highlight the currently selected command at index in the graphical view. """
        cmd_np_list = self.convert_list_of_cmds(self.gui_cmd_list)
        cmd_np_list = np.reshape(cmd_np_list, (len(self.gui_cmd_list), 4))
        trigger_indices = [
            (i, cmd[1]) for i, cmd in enumerate(cmd_np_list) if cmd[0] == 2]
        for ch_name, ch_dict in self.ch_pos_dict.items():
            ch_line = ch_dict.get('line', False)
            ch_int = 2 ** int(ch_name[2:])
            normal_ch = True
            if ch_line:   # there is an existing line for the channel
                segments = ch_line.segments
                if 'DO' in ch_name:  # it is an output channel
                    standard_pen = Pgplot.pg.mkPen('b', width=3)
                else:  # it is a trigger channel
                    normal_ch = False
                    standard_pen = Pgplot.pg.mkPen('r', width=3)
                highlight_pen = Pgplot.create_pen('g', width=3)  # highlighted green
                white_pen = Pgplot.create_pen('w', width=3)  # for separating the stop pulse
                for i, each in enumerate(segments):
                    if each.currentPen != standard_pen:
                        # overwrite all segments wiht the standard pen for this type
                        each.setPen(standard_pen)
                    for chosen_index in indice_list:
                        # highlight the chosen segments which are mentioned in the indice list
                        if normal_ch:
                            if chosen_index == self.listWidget_cmd_list.count() - 1 and stop_in_list:
                                segments[-1].setPen(highlight_pen)
                            else:
                                segments[chosen_index * 2].setPen(highlight_pen)
                        else:  # each trigger always has 6 segments, only highlight first one.
                            trig_ind = [tr[0] for tr in trigger_indices]
                            # print(trig_ind)
                            if chosen_index in trig_ind:
                                trig_num = sum([1 for each in trigger_indices
                                                if each[0] <= chosen_index and each[1] & ch_int != 0])
                                if trig_num > 0:
                                    trig_ind = trig_num - 1
                                    segments[trig_ind * 6].setPen(highlight_pen)
                    if stop_in_list and i == len(segments) - 3 and normal_ch:
                        each.setPen(white_pen)

    """ talk to dev """

    def stop_pulse_pattern(self):
        """ will stop the pulse pattern. """
        if CfgMain._main_instance is not None:
            CfgMain._main_instance.ppg_stop()
        else:
            print('error, stopping pulse pattern not possible, because there is no main active')

    def run(self):
        """ run the pulse pattern with the pattern as in the gui. """
        cmd_list = self.cmd_list_from_gui()
        if self.check_cmd_list_credibility(cmd_list):
            if cmd_list:
                if CfgMain._main_instance:
                    CfgMain._main_instance.ppg_load_pattern(cmd_list)
                else:
                    print('error, running pulse pattern not possible, because there is no main active')

    ''' window related '''

    def close_and_confirm(self):
        """ close the window and store the pulse pattern in the scan pars of the active iso """
        items = self.cmd_list_from_gui()
        # print('items in gui: ', items)
        if self.track_gui is not None:
            self.track_gui.buffer_pars['pulsePattern'] = {}
            self.track_gui.buffer_pars['pulsePattern']['cmdList'] = items
            self.track_gui.buffer_pars['pulsePattern']['periodicList'] = deepcopy(
                self.periodic_widg.return_periodic_list())
            self.track_gui.buffer_pars['pulsePattern']['simpleDict'] = deepcopy(
                self.simple_widg.return_simple_dict()
            )
        self.close()

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        if self.main_gui is not None:
            # tell main window that this window is closed.
            self.main_gui.close_pulse_pattern_win()
        if self.track_gui is not None:
            self.track_gui.close_pulse_pattern_window()


if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = PulsePatternUi(None, '', None)
    # gui.load_from_text(txt_path='E:\\TildaDebugging\\Pulsepattern123Pattern.txt')
    # print(gui.get_gr_v_pos_from_list_of_cmds())
    app.exec_()
