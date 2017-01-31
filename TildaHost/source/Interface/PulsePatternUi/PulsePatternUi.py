"""

Created on '16.01.2017'

@author:'simkaufm'

"""

import ast
import sys
import os

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from functools import partial
from copy import deepcopy

import Application.Config as CfgMain
import Service.FileOperations.FolderAndFileHandling as FileHandl
from Interface.PulsePatternUi.Ui_PulsePattern import Ui_PulsePatternWin
from Interface.PulsePatternUi.PpgPeriodicWidgUi import PpgPeriodicWidgUi
from Interface.PulsePatternUi.PpgSimpleWidgUi import PpgSimpleWidgUi

import PyQtGraphPlotter as Pgplot


class PulsePatternUi(QtWidgets.QMainWindow, Ui_PulsePatternWin):
    pulse_pattern_status = QtCore.pyqtSignal(str)
    cmd_list_signal = QtCore.pyqtSignal(list)

    def __init__(self, active_iso, track_name, main_gui, track_gui=None):
        super(PulsePatternUi, self).__init__()
        self.gui_cmd_list = []  # list of commands in gui, always updated when self.cmd_list_from_gui() is called.
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

        ''' help '''
        self.actionHelp.triggered.connect(self.open_help)

        ''' graphical view related '''
        self.listWidget_cmd_list.currentTextChanged.connect(self.update_gr_v)
        self.listWidget_cmd_list.currentRowChanged.connect(self.update_gr_v)
        # self.listWidget_cmd_list.itemChanged.connect(self.update_gr_v)

        self.ch_pos_dict = {}
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

    def cmd_list_to_gui(self, cmd_list):
        """ write a list of str cmd to the gui """
        # remove all items tht were already in the list.
        self.listWidget_cmd_list.clear()
        self.listWidget_cmd_list.addItems(cmd_list)
        for i in range(self.listWidget_cmd_list.count()):
            self.listWidget_cmd_list.item(i).setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsDragEnabled)
        self.update_gr_v(cmd_list)

    def cmd_list_from_gui(self):
        """ return a list of all cmds in the gui """
        items = []
        for i in range(self.listWidget_cmd_list.count()):
            items.append(self.listWidget_cmd_list.item(i).text())
        self.gui_cmd_list = items
        return items

    def remove_selected(self):
        """ if an item is selected, remove this from the list. """
        cur_row = self.listWidget_cmd_list.currentRow()
        if cur_row != -1:
            self.listWidget_cmd_list.takeItem(cur_row)

    def close_and_confirm(self):
        """ close the window and store the pulse pattern in the scan pars of the active iso """
        items = self.cmd_list_from_gui()
        # print('items in gui: ', items)
        if self.track_gui is not None:
            self.track_gui.buffer_pars['pulsePattern'] = {}
            self.track_gui.buffer_pars['pulsePattern']['cmdList'] = items
            self.track_gui.buffer_pars['pulsePattern']['periodicList'] = deepcopy(
                self.periodic_widg.return_periodic_list())
        self.close()

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        if self.main_gui is not None:
            # tell main window that this window is closed.
            self.main_gui.close_pulse_pattern_win()
        if self.track_gui is not None:
            self.track_gui.close_pulse_pattern_window()

    def add_before(self):
        """ add copy of selected command before the selected one. If none selected, place at end. """
        cur_row = self.listWidget_cmd_list.currentRow()
        if cur_row == -1:
            if self.listWidget_cmd_list.count():
                cur_row = self.listWidget_cmd_list.count() - 1
            else:
                cur_row = 0
        old_item = self.listWidget_cmd_list.item(cur_row)
        if old_item is None:
            self.listWidget_cmd_list.insertItem(cur_row, '$cmd::1.0::0::0')
        else:
            self.listWidget_cmd_list.insertItem(cur_row, old_item.text())
        self.listWidget_cmd_list.item(cur_row).setFlags(
            QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsDragEnabled)

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

    def rcvd_state(self, state_str):
        """ when state of ppg changes this signal is emitted """
        self.ppg_state = state_str
        self.label_ppg_state.setText(state_str)

    ''' periodic related '''
    def load_periodic(self, per_list):
        print('loading periodic list: %s' % per_list)
        self.periodic_widg.setup_from_list(per_list)

    ''' help '''
    def open_help(self):
        mb = QtWidgets.QMessageBox(self)
        QtWidgets.QMessageBox.information(
            mb, 'ppg help',
            'from: \n F. Ziegler et al., A newPulse-PatternGeneratorbasedonLab'
            'VIEWFPGA, Nucl. Instr. Meth. A 679 (2012) 1-6:\n\n'
            'Command\t Example\t Description\n'
            '------------------------------------------------------------\n'
            '$time\t $time::1000::123::123\t command::time[ms] ::DO0-31::DO32-63\n'
            '$wait\t $wait::1::123::123\t command::DI0-7::DO0-31::DO32-63\n'
            '$jump\t $jump::0::500::500\t command::address::unused::iterationnumber\n'
            '$stop\t $stop::0::123::123\t command::unused::DO0-31::DO32-63'
        )

    """ graphical displaying """
    def add_graph_view(self):
        try:
            layout = QtWidgets.QVBoxLayout()
            self.gr_v_widg, self.gr_v_plt_itm = Pgplot.create_x_y_widget(y_label='channel', x_label='time [us]')
            layout.addWidget(self.gr_v_widg)
            self.widget_graph_view.setLayout(layout)
        except Exception as e:
            print('error while adding graphical view: %s' % e)

    def update_gr_v(self, external_list=None):
        """ updates the graphic view and adds a line for each item """
        try:
            print('updating praphical view, ext. list: %s' % external_list)
            old_list = deepcopy(self.gui_cmd_list)
            new_list = self.cmd_list_from_gui()
            if old_list != new_list:
                if not isinstance(external_list, list):
                    # manual change on the cmd list -> therefore clear all other
                    # widgets because their data is probably now corrupted
                    if self.periodic_widg is not None:
                        print('clearing periodic table')
                        self.periodic_widg.list_view_was_changed()
                # print('updating graphics view')
                self.ch_pos_dict, valid_lines = self.get_gr_v_pos_from_list_of_cmds(
                    self.cmd_list_from_gui(), ret_dict=self.ch_pos_dict)
                lines_to_remove = [each for each in self.ch_pos_dict.keys() if each not in valid_lines]
                self.remove_lines(self.gr_v_plt_itm, lines_to_remove)
                self.add_lines(self.gr_v_plt_itm)
            else:
                pass
                # print('nope not a new list, not redrawing here')
        except Exception as e:
            print('error while updating graphical view: %s' % e)

    def remove_lines(self, plt_item, remove_list):
        ch_pos_dict = self.ch_pos_dict
        for each in remove_list:
            # print('removing: %s' % each)
            if ch_pos_dict.get(each, {}).get('line', None):
                plt_item.removeItem(ch_pos_dict[each]['line'])
                self.ch_pos_dict.pop(each)

    def add_lines(self, plt_item):
        """ add a roi polyline line for every channel in ch_pos_dict """
        ch_pos_dict = self.ch_pos_dict
        for ch, ch_dict in ch_pos_dict.items():
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
                plt_item.addItem(ch_dict['line'])

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
                    time += 1  # add another us at the stop
                    ch_pos_list.append([time, y_pos])
                elif each_cmd[0] == 1:  # $jump
                    pass

                elif each_cmd[0] == 2:  # $wait
                    y_pos = high if ch_bit & each_cmd[2] != 0 else low
                    ch_pos_list.append([time, y_pos])
                    time += 1  # draw 1 us before ris edge of trigger
                    ch_pos_list.append([time, y_pos])
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
                                [time + 0.5, tr_low], [time + 1.5, tr_low],
                            ]
                            if ret_dict.get('DI%s' % tr_ch, None) is None:
                                ret_dict['DI%s' % tr_ch] = {}
                            ret_dict['DI%s' % tr_ch]['pos'] = tr_pos
                            valid_lines.append('DI%s' % tr_ch)
                        else:
                            if ret_dict.get('DI%s' % tr_ch, None) is None:
                                ret_dict['DI%s' % tr_ch] = {}
                            ret_dict['DI%s' % tr_ch]['pos'] = []
                            valid_lines.append('DI%s' % tr_ch)
                        # else:  # might work
                        #     ret_dict['DI%s' % tr_ch]['pos'] += tr_pos

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
            ticks_per_us = self.read_ticks_per_us()
        cmd_dict = {'$stop': 0, '$jump': 1, '$wait': 2, '$time': 3}
        cmd_list = cmd_str.split('::')
        if len(cmd_list) == 4:
            try:
                cmd_list[0] = cmd_dict.get(cmd_list[0], -1)
                for i in range(1, 4):
                    cmd_list[i] = ast.literal_eval(cmd_list[i])
                if cmd_list[0] == 3:  # $time
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
            ticks_per_us = 100  # TODO think about if this is ok to be static
        for each_cmd in cmd_list:
            ret_arr = np.append(ret_arr, self.convert_single_comand(each_cmd, ticks_per_us))
        return ret_arr

    def convert_np_arr_of_cmd_to_list_of_cmds(self, np_arr_cmds, ticks_per_us=None):
        ret = []
        if ticks_per_us is None:
            ticks_per_us = self.read_ticks_per_us()
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
            ticks_per_us = self.read_ticks_per_us()
        cmd_dict = {0: '$stop', 1: '$jump', 2: '$wait', 3: '$time'}
        ret_cmd_str = '%s::%.2f::%s::%s' % (
            cmd_dict.get(int_arr[0], 'error'), int_arr[1] / ticks_per_us, int_arr[2], int_arr[3]
        )
        return ret_cmd_str

    """ talk to dev """

    def stop_pulse_pattern(self):
        """ will stop the pulse pattern. """
        CfgMain._main_instance.ppg_stop()

    def run(self):
        """ run the pulse pattern with the pattern as in the gui. """
        cmd_list = self.cmd_list_from_gui()
        if cmd_list:
            CfgMain._main_instance.ppg_load_pattern(cmd_list)

            # if __name__ == '__main__':
            #     from Driver.COntrolFpga.PulsePatternGenerator import PulsePatternGenerator as PPG
            #     ppg = PPG()
            #     ppg_test_path = 'D:\\Debugging\\trs_debug\\Pulsepattern132Pattern.txt'
            #     app = QtWidgets.QApplication(sys.argv)
            #     gui = PulsePatternUi(None, '')
            #     # gui.cmd_list_to_gui(cmd_str)
            #     app.exec_()


if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = PulsePatternUi(None, '', None)
    # gui.load_from_text(txt_path='E:\\TildaDebugging\\Pulsepattern123Pattern.txt')
    # print(gui.get_gr_v_pos_from_list_of_cmds())
    app.exec_()
