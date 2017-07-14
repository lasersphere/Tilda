"""
Created on 

@author: simkaufm

Module Description:  Gui for configuring the outbits for each track
"""

import logging
import numpy as np
from PyQt5 import QtWidgets, QtGui
from copy import deepcopy


from Interface.OutBitsUi.Ui_OutBits import Ui_outbits
from Interface.OutBitsUi.UiSingleOutBit import UiSingleOutBit


class OutBitsUi(QtWidgets.QMainWindow, Ui_outbits):

    def __init__(self, parent, outbit_dict={}, confirm_settings_signal=None):
        super(OutBitsUi, self).__init__(parent=parent)
        self.setupUi(self)

        self.temp_outbit_dict = outbit_dict
        self.confirm_settings_signal = confirm_settings_signal

        self.load_dict_to_list_widg()

        """ connect buttons """
        self.pushButton_cancel.clicked.connect(self.close)
        self.pushButton_ok.clicked.connect(self.confirm)

        self.pushButton_edit_selected.clicked.connect(self.edit_selected)
        self.pushButton_add_outbit.clicked.connect(self.add_item)
        self.pushButton_remove_selected.clicked.connect(self.rem_selected)

        """ keyboard shortcuts """
        QtWidgets.QShortcut(QtGui.QKeySequence("A"), self, self.add_item)
        QtWidgets.QShortcut(QtGui.QKeySequence("E"), self, self.edit_selected)
        QtWidgets.QShortcut(QtGui.QKeySequence("DEL"), self, self.rem_selected)
        QtWidgets.QShortcut(QtGui.QKeySequence("O"), self, self.confirm)
        QtWidgets.QShortcut(QtGui.QKeySequence("ESC"), self, self.close)

        """ mouse shortcuts """
        self.listWidget_outbits.itemDoubleClicked.connect(self.edit_selected)

        self.show()
        logging.info('opened outbits config gui')
        logging.debug('outbit gui has temp outbit dict: %s' % self.temp_outbit_dict)

    def load_dict_to_list_widg(self):
        """ fill the listwidget with items as stated in self.temp_outbit_dict """
        self.listWidget_outbits.clear()
        self.check_steps_unique()
        for outb_name, outb_cmd_list in sorted(self.temp_outbit_dict.items()):
            for cmd_tpl in outb_cmd_list:
                self.listWidget_outbits.insertItem(0, self.name_cmd(outb_name, cmd_tpl))

    def name_cmd(self, bit_name, cmd_tuple):
        """ create a string from bitname, and the command tuple for the listwidget """
        ret_str = ''
        ret_str += '%s |' % bit_name
        ret_str += ' %s | %s | scan/step num.: %s' % cmd_tuple
        return ret_str

    def edit_selected(self, item=None, just_added=False):
        """ edit the currently selected item """
        cur_ind = self.listWidget_outbits.currentRow()
        if cur_ind != -1:
            outb_name, lis_index, cmd = self.find_list_item_in_dict(cur_ind)
            dial = UiSingleOutBit(self, deepcopy(self.temp_outbit_dict), outb_name, lis_index, cmd)
            if dial.exec():  # will return 1 for ok
                if outb_name:  # this means this is an existing command
                    # first delete the old cmd
                    del self.temp_outbit_dict[outb_name][lis_index]
                new_outb_name, new_cmd = dial.get_cmd()
                # then append the cmd at the end of the new outbit list
                toggle_in_new_outbit_cmds = any(
                    [each[0] == 'toggle' for each in self.temp_outbit_dict.get(new_outb_name, [])]
                ) or new_cmd[0] == 'toggle'
                if toggle_in_new_outbit_cmds:  # if its toggle command delete all other commands
                    logging.warning('for toggle mode only one command is allowed! deleting all others now!')
                    self.temp_outbit_dict[new_outb_name] = [new_cmd]
                else:  # append command if its not toggle
                    if self.temp_outbit_dict.get(new_outb_name, None) is None:
                        self.temp_outbit_dict[new_outb_name] = []
                    self.temp_outbit_dict[new_outb_name] += [new_cmd]
                for i, each in enumerate(self.temp_outbit_dict[new_outb_name]):
                    # set all commands for this step to either scan or step mode depending on the newly selected one.
                    self.temp_outbit_dict[new_outb_name][i] = (
                        self.temp_outbit_dict[new_outb_name][i][0],
                        new_cmd[1],
                        self.temp_outbit_dict[new_outb_name][i][2])
                if len(self.temp_outbit_dict[new_outb_name]) > 40:
                    del self.temp_outbit_dict[new_outb_name][0]
                    logging.warning('to many commands for %s deleting first one! Only 40 cmds allowed!' % new_outb_name)
                self.load_dict_to_list_widg()
                return True
        else:
            return False

    def check_steps_unique(self):
        """ go through self.temp_outbit_dict and check that the stepnumber for each outbit is unique """
        for outb_name, outb_cmd_l in self.temp_outbit_dict.items():
            outb_flat_steps = np.array([cmd[2] for cmd in outb_cmd_l])
            u, indices = np.unique(outb_flat_steps, return_index=True)
            # only keep commands with unique steps
            for i, each in enumerate(outb_cmd_l):
                if i not in indices:
                    logging.warning(
                        'deleting cmd %s in %s because stepnumber is not unique!'
                        % (str(self.temp_outbit_dict[outb_name][i]), outb_name))
                    del self.temp_outbit_dict[outb_name][i]

    def add_item(self):
        """ add an item and open the dialog for editing """
        self.listWidget_outbits.insertItem(0, self.name_cmd('', ('', '', -1)))
        self.listWidget_outbits.setCurrentRow(0)
        if not self.edit_selected():
            # user clicked on cancel
            self.listWidget_outbits.takeItem(0)

    def rem_selected(self):
        """ remove the selected item """
        cur_ind = self.listWidget_outbits.currentRow()
        if cur_ind != -1:
            outb_name, index, cmd = self.find_list_item_in_dict(cur_ind)
            if outb_name:
                self.listWidget_outbits.takeItem(cur_ind)
                del self.temp_outbit_dict[outb_name][index]
                logging.debug('removed %s %s at index %d' % (outb_name, str(cmd), index))

    def find_list_item_in_dict(self, index):
        """ items in list can appear in random order, therefor find the position in the temp_outbit_dict """
        cur_item = self.listWidget_outbits.item(index)
        cur_text = cur_item.text()
        outb_name, on_off_togle, step_scan, scan_step_num = cur_text.replace(
            ' ', '').replace('scan/stepnum.:', '').split('|')
        cmd = (on_off_togle, step_scan, int(scan_step_num))
        if cmd in self.temp_outbit_dict.get(outb_name, []):
            index = self.temp_outbit_dict.get(outb_name, []).index(cmd)
            return outb_name, index, cmd
        else:
            return '', -1, ('', '', -1)

    ''' close and confirm '''

    def confirm(self):
        """ confirm the selected settings and tell parent """
        if self.confirm_settings_signal is not None:
            self.confirm_settings_signal.emit(self.temp_outbit_dict)
        self.close()

if __name__ == '__main__':
    import sys

    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s %(funcName)s(%(lineno)d) %(message)s')

    app_log = logging.getLogger()
    app_log.setLevel(logging.DEBUG)
    app_log.info('Log level set to ' + 'DEBUG')

    draft_outbits = {
        'outbit0': [('toggle', 'scan', 0)],
        'outbit1': [('on', 'step', 1), ('off', 'step', 1)],
        'outbit2': [('on', 'step', 1), ('off', 'step', 5)]
    }

    app = QtWidgets.QApplication(sys.argv)
    gui = OutBitsUi(None, draft_outbits)
    # gui.load_from_text(txt_path='E:\\TildaDebugging\\Pulsepattern123Pattern.txt')
    # print(gui.get_gr_v_pos_from_list_of_cmds())
    app.exec_()
