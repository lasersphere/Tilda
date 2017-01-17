"""

Created on '16.01.2017'

@author:'simkaufm'

"""

import ast

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

import Application.Config as CfgMain
import Service.FileOperations.FolderAndFileHandling as FileHandl
from Interface.PulsePatternUi.Ui_PulsePattern import Ui_PulsePatternWin


class PulsePatternUi(QtWidgets.QMainWindow, Ui_PulsePatternWin):
    pulse_pattern_status = QtCore.pyqtSignal(str)

    def __init__(self, active_iso, track_name, main_gui, track_gui=None):
        super(PulsePatternUi, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('pulse pattern of %s %s' % (active_iso, track_name))
        self.active_iso = active_iso
        self.track_name = track_name
        self.main_gui = main_gui
        self.track_gui = track_gui
        if self.main_gui.pulse_pattern_win is not None:
            self.main_gui.pulse_pattern_win.close()

        ''' state related'''
        self.ppg_state = None
        self.rcvd_state('not initialised')
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


        self.show()

    def cmd_list_to_gui(self, cmd_list):
        """ write a list of str cmd to the gui """
        # remove all items tht were already in the list.
        self.listWidget_cmd_list.clear()
        self.listWidget_cmd_list.addItems(cmd_list)
        for i in range(self.listWidget_cmd_list.count()):
            self.listWidget_cmd_list.item(i).setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsDragEnabled)

    def cmd_list_from_gui(self):
        """ return a list of all cmds in the gui """
        items = []
        for i in range(self.listWidget_cmd_list.count()):
            items.append(self.listWidget_cmd_list.item(i).text())
        return items

    def remove_selected(self):
        """ if an item is selected, remove this from the list. """
        cur_row = self.listWidget_cmd_list.currentRow()
        if cur_row != -1:
            self.listWidget_cmd_list.takeItem(cur_row)

    def close_and_confirm(self):
        """ close the window and store the pulse pattern in the scan pars of the active iso """
        items = self.cmd_list_from_gui()
        print('items in gui: ', items)
        if self.track_gui is not None:
            self.track_gui.buffer_pars['pulsePattern'] = {}
            self.track_gui.buffer_pars['pulsePattern']['cmdList'] = items
            print('wrote to buffer pars: ', self.track_gui.buffer_pars['pulsePattern']['cmdList'])
        self.close()

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        if self.main_gui is not None:
            # tell main window that this window is closed.
            self.main_gui.close_pulse_pattern_win()

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

    def load_from_text(self):
        """ load the list of cmds from an existing text file """
        parent = QtWidgets.QFileDialog(self)
        path, ok = QtWidgets.QFileDialog.getOpenFileName(
            parent, "select ppg .txt file", CfgMain._main_instance.working_directory, '*.txt')
        if path:
            list_of_cmds = FileHandl.load_from_text_file(path)
            self.cmd_list_to_gui(list_of_cmds)
            return path

    def save_to_text_file(self):
        """ save the current settings to a text file """
        parent = QtWidgets.QFileDialog(self)
        path, ok = QtWidgets.QFileDialog.getSaveFileName(
            parent, "select ppg .txt file", CfgMain._main_instance.working_directory, '*.txt')
        if path:
            FileHandl.save_txt_file_line_by_line(path, self.cmd_list_from_gui())
            return path

    def rcvd_state(self, state_str):
        """ when state of ppg changes this signal is emitted """
        self.ppg_state = state_str
        self.label_ppg_state.setText(state_str)

    """ graphical displaying """
    # TODO

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
            ticks_per_us = self.read_ticks_per_us()
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
