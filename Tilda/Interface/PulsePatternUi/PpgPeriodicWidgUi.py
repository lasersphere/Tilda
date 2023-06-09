"""

Created on '18.01.2017'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets, QtGui, Qt
from functools import partial
from copy import deepcopy
import logging

from Tilda.Interface.PulsePatternUi.Ui_PpgPeriodicWidg import Ui_PpgPeriodicWidg
from Tilda.Interface.PulsePatternUi.ChannelUi import ChannelUi
from Tilda.Interface.PulsePatternUi.TriggerUi import TriggerUi
from Tilda.Interface.PulsePatternUi.StopUi import StopUi


class PpgPeriodicWidgUi(QtWidgets.QWidget, Ui_PpgPeriodicWidg):

    def __init__(self, parent, cmd_list_callback_signal):
        super(PpgPeriodicWidgUi, self).__init__(parent)
        self.main_ppg_win = parent
        self.cmd_list_callback_signal = cmd_list_callback_signal
        self.setupUi(self)
        self.show()

        self.list_of_item_dicts = []  # this will hold all periodic infos. its a list not a dict due to ordering.
        self._backup = []  # will be used when something was cleared or so and user wants to reload

        self.output_channels = ['DO%s' % i for i in range(0, 32)]
        # self.sys_rep_rate_changed(100)

        # since no signal for moving is known, better use the buttons
        # self.listWidget_periodic_pattern.setDragDropMode(self.listWidget_periodic_pattern.InternalMove)

        ''' connect buttons '''
        self.pushButton_add_ch.clicked.connect(self.add_ch)
        self.pushButton_add_trig.clicked.connect(self.add_trig)
        self.pushButton_remove_selected_periodic.clicked.connect(self.rem_selected)
        self.pushButton_edit_selected.clicked.connect(self.dbl_clicked)
        self.pushButton_move_up.clicked.connect(partial(self.move_item, -1))
        self.pushButton_move_down.clicked.connect(partial(self.move_item, +1))
        self.pushButton_reload_settings.clicked.connect(self.reload_last_settings)

        ''' keyboard shortcuts '''
        QtWidgets.QShortcut(QtGui.QKeySequence("UP"), self, partial(self.move_item, -1))
        QtWidgets.QShortcut(QtGui.QKeySequence("DOWN"), self, partial(self.move_item, +1))

        ''' signals '''
        self.listWidget_periodic_pattern.itemDoubleClicked.connect(self.dbl_clicked)
        self.doubleSpinBox_sys_rep_rate.valueChanged.connect(self.sys_rep_rate_changed)

        self.doubleSpinBox_sys_rep_rate.setKeyboardTracking(False)

        ''' example '''
        example_data = [
            {'type': 'sysRep', 'repRateUs': 10},
            {'trigName': 'NoName', 'trigChannels': [0], 'actCh': [1], 'type': 'trig'},
            {'chName': 'NoName', 'inverted': True, 'outCh': 'DO0',
             'delayUs': 15.0, 'widthUs': 1.0, 'type': 'ch', 'numOfPulses': 3},
            {'chName': 'NoName', 'inverted': False, 'outCh': 'DO1', 'delayUs': 1.2,
             'widthUs': 0.8, 'type': 'ch', 'numOfPulses': 3},
            {'chName': 'NoName', 'inverted': False, 'outCh': 'DO2',
             'delayUs': 5.0, 'widthUs': 0.5, 'type': 'ch', 'numOfPulses': 2},
            {'actCh': [0, 1, 2], 'type': 'stop'}
        ]

        # self.setup_from_list(example_data)
        # self.get_cmd_list()
        # self.get_ch_high_low_list(self.list_of_item_dicts[2], 0, 10)

    def add_ch(self, ch_dict=None):
        """ add a channel """
        cur_ind = self.listWidget_periodic_pattern.currentRow() + 1
        used_outputs = [ch['outCh'] for ch in self.list_of_item_dicts if ch['type'] == 'ch']
        if cur_ind == -1:
            cur_ind = 0
        if not isinstance(ch_dict, dict):
            out_ch = [ch for ch in self.output_channels if ch not in used_outputs][0]
            ch_dict = {
                'type': 'ch',
                'chName': 'NoName',
                'outCh': out_ch,
                'numOfPulses': 1,
                'widthUs': 1.0,
                'delayUs': 1.0,
                'inverted': False
            }
        self._backup = deepcopy(self.list_of_item_dicts)
        self.list_of_item_dicts.insert(cur_ind, ch_dict)
        self.listWidget_periodic_pattern.insertItem(cur_ind, 'ch')
        self.name_item(cur_ind)
        self.add_stop()

    def name_item(self, cur_ind):
        """ name the item in the list, by its parameters """
        item_dict = self.list_of_item_dicts[cur_ind]
        item_type = item_dict['type']
        name = 'item'
        if item_type == 'trig':
            name = 'trigger: %s | input Channels: %s | active ch. while wait: %s' % (
                item_dict['trigName'], item_dict['trigChannels'], item_dict['actCh']
            )
        elif item_type == 'ch':
            name = 'ch: %s | ouput: %s | #pulses %s | width [us]: %s | delay [us]: %s | inverted: %s' % (
                item_dict['chName'], item_dict['outCh'], item_dict['numOfPulses'],
                item_dict['widthUs'], item_dict['delayUs'], item_dict['inverted']
            )
        elif item_type == 'stop':
            name = 'stop | active ch when stopped: %s' % str(item_dict['actCh'])[1:-1]
        elif item_type == 'sysRep':
            name = 'system rep rate [us]: %s ' % item_dict['repRateUs']
        self.listWidget_periodic_pattern.item(cur_ind).setText(name)

    def add_trig(self, trig_dict=None):
        """ add a trigger element at beginning of list """
        trig_exists_already = True in [each['type'] == 'trig' for each in self.list_of_item_dicts]
        if not trig_exists_already:
            sys_rep = True in [each['type'] == 'sysRep' for each in self.list_of_item_dicts]
            cur_ind = 1 if sys_rep else 0
            if not isinstance(trig_dict, dict):
                trig_dict = {
                    'type': 'trig',
                    'trigName': 'NoName',
                    'trigChannels': [],
                    'actCh': []
                }
            self.listWidget_periodic_pattern.insertItem(cur_ind, 'trigger')
            self._backup = deepcopy(self.list_of_item_dicts)
            self.list_of_item_dicts.insert(cur_ind, trig_dict)
            self.name_item(cur_ind)
            self.add_stop()
        else:
            logging.warning('did not add another trigger, in periodic mode only one trigger is supported for now.')

    def add_stop(self, stop_dict=None):
        """ add a stop cmd at end of cmds if not yet there.
         Move the exisiting stop command to the end if not yet done so """
        stop_in_list = [each['type'] == 'stop' for each in self.list_of_item_dicts]
        if True in stop_in_list:  # there is already a stop in the list, move it to the end
            old_index = stop_in_list.index(True)
            if stop_dict is not None:  # if stop dict given by call, replace the exisitng one
                self.list_of_item_dicts[old_index] = stop_dict
            new_ind = len(self.list_of_item_dicts)
            if old_index != new_ind:  # needs to be moved to the end.
                self.list_of_item_dicts.insert(new_ind, self.list_of_item_dicts.pop(old_index))
                self.listWidget_periodic_pattern.insertItem(new_ind, self.listWidget_periodic_pattern.takeItem(old_index))
        else:  # add stop at the end
            if not isinstance(stop_dict, dict):
                stop_dict = {
                    'type': 'stop',
                    'actCh': []
                }
            self.listWidget_periodic_pattern.addItem('stop')
            self.list_of_item_dicts.append(stop_dict)
        self.name_item(len(self.list_of_item_dicts) - 1)
        self.get_cmd_list()

    def rem_selected(self):
        """ remove the selected item """
        cur_ind = self.listWidget_periodic_pattern.currentRow()
        self._backup = deepcopy(self.list_of_item_dicts)
        self.list_of_item_dicts.pop(cur_ind)
        if cur_ind != -1:
            self.listWidget_periodic_pattern.takeItem(cur_ind)
        self.add_stop()

    def dbl_clicked(self, item=None):
        """ called when item is double clicked/edit item is clicked -> open gui """
        cur_ind = self.listWidget_periodic_pattern.currentRow()
        if cur_ind != -1:
            try:
                self._backup = deepcopy(self.list_of_item_dicts)
                dial = None
                if self.list_of_item_dicts[cur_ind]['type'] == 'ch':
                    used_outputs = [ch['outCh'] for ch in self.list_of_item_dicts if ch['type'] == 'ch']
                    own_ch = self.list_of_item_dicts[cur_ind]['outCh']
                    used_outputs.remove(own_ch)
                    dial = ChannelUi(self, self.list_of_item_dicts[cur_ind], used_outputs)
                elif self.list_of_item_dicts[cur_ind]['type'] == 'trig':
                    dial = TriggerUi(self, self.list_of_item_dicts[cur_ind])
                elif self.list_of_item_dicts[cur_ind]['type'] == 'stop':
                    dial = StopUi(self, self.list_of_item_dicts[cur_ind])
                elif self.list_of_item_dicts[cur_ind]['type'] == 'sysRep':
                    self.doubleSpinBox_sys_rep_rate.setFocus(Qt.Qt.OtherFocusReason)
                if dial is not None:
                    if dial.exec():  # will return 1 for ok
                        self.list_of_item_dicts[cur_ind] = dial.get_dict_from_gui()
                        self.name_item(cur_ind)
                        self.get_cmd_list()
            except Exception as e:
                logging.error('error while double clicked: %s ' % e, exc_info=True)

    def sys_rep_rate_changed(self, val):
        # print('system rep rate changed to: %s' % val)
        rep_rate_dict = {
            'type': 'sysRep',
            'repRateUs': deepcopy(val)
        }
        sys_rep_in_list = [each['type'] == 'sysRep' for each in self.list_of_item_dicts]
        self._backup = deepcopy(self.list_of_item_dicts)
        if True in sys_rep_in_list:  # there is already a sysrep in the list, delete it
            old_index = sys_rep_in_list.index(True)
            self.listWidget_periodic_pattern.takeItem(old_index)
            self.list_of_item_dicts.pop(old_index)
        # insert sys period item at front of list.
        self.list_of_item_dicts.insert(0, rep_rate_dict)
        self.listWidget_periodic_pattern.insertItem(0, 'system repetition rate')
        self.doubleSpinBox_sys_rep_rate.blockSignals(True)
        self.doubleSpinBox_sys_rep_rate.setValue(val)
        self.doubleSpinBox_sys_rep_rate.blockSignals(False)
        self.name_item(0)
        self.add_stop()

    def move_item(self, up_down):
        """ move an item in list up (-1) or down (+1) """
        cur_ind = self.listWidget_periodic_pattern.currentRow()
        self._backup = deepcopy(self.list_of_item_dicts)
        if cur_ind != -1:
            move_list = ['ch']
            itm_type = self.list_of_item_dicts[cur_ind]['type']
            if itm_type in move_list:
                min_ind = 0
                trig_exists_already = True in [each['type'] == 'trig' for each in self.list_of_item_dicts]
                sys_rep_exists_alrdy = True in [each['type'] == 'sysRep' for each in self.list_of_item_dicts]
                min_ind += 1 if trig_exists_already else 0
                min_ind += 1 if sys_rep_exists_alrdy else 0
                new_ind = cur_ind + up_down
                if min_ind <= new_ind <= len(self.list_of_item_dicts):
                    # move is possible
                    self.list_of_item_dicts.insert(new_ind, self.list_of_item_dicts.pop(cur_ind))
                    self.listWidget_periodic_pattern.insertItem(new_ind, self.listWidget_periodic_pattern.takeItem(cur_ind))
                    self.listWidget_periodic_pattern.setCurrentRow(new_ind)
                    self.add_stop()

    def setup_from_list(self, list_of_dicts):
        """ setup from list of dicts """
        for ind, each_dict in enumerate(list_of_dicts):
            if each_dict['type'] == 'trig':
                self.add_trig(each_dict)
            elif each_dict['type'] == 'ch':
                self.add_ch(each_dict)
            elif each_dict['type'] == 'stop':
                self.add_stop(each_dict)
            elif each_dict['type'] == 'sysRep':
                self.sys_rep_rate_changed(each_dict['repRateUs'])
            self.listWidget_periodic_pattern.setCurrentRow(ind)

    def get_cmd_list(self):
        """ from the periodic setup create a cmd list as it is useable for the list view
         and send it to the list view via the self.cmd_list_callback_signal """
        sys_rep_in_list = [each['type'] == 'sysRep' for each in self.list_of_item_dicts]
        if True in sys_rep_in_list:
            old_index = sys_rep_in_list.index(True)
            sys_per = self.list_of_item_dicts[old_index]['repRateUs']
        else:
            sys_per = -1
            # print('error: no system period defined! will not create command list')
            return []
        trig_list = []
        ch_hi_lo_list = []
        t_0 = 0
        stop_cmd = ''
        for each in self.list_of_item_dicts:
            if each['type'] == 'trig':
                act_ch = sum([2 ** ch for ch in each['actCh']])
                trig_ch = sum([2 ** ch for ch in each['trigChannels']])
                # only dios 0-33 supported for now.
                trig_list.append('$wait::%s::%s::0' % (trig_ch, act_ch))
                t_0 += 1
            elif each['type'] == 'ch':
                ch_hi_lo_list.append(self.get_ch_high_low_list(each, sys_per, t_0=t_0))
            elif each['type'] == 'stop':
                act_ch = sum([2 ** ch for ch in each['actCh']])
                stop_cmd = '$stop::0::%s::0' % act_ch
        ch_hi_lo_list_combined = self.combine_ch_high_low_lists(ch_hi_lo_list)
        cmd_list = self.ch_hi_lo_list_to_cmd_list(ch_hi_lo_list_combined)
        if trig_list:
            cmd_list.insert(0, trig_list[0])
        if stop_cmd:
            cmd_list.append(stop_cmd)
        logging.debug('emitting %s, from %s, value is %s'
                      % ('cmd_list_callback_signal',
                         'Interface.PulsePatternUi.PpgPeriodicWidgUi.PpgPeriodicWidgUi#get_cmd_list',
                         str((cmd_list, 'periodic'))))
        self.cmd_list_callback_signal.emit(cmd_list, 'periodic')
        return cmd_list

    def get_ch_high_low_list(self, ch_dict, sys_per_us, t_0=0):
        """
        calc if the channel is high or low a t the querried time
        :param ch_dict:
        :param time: float, time which is questionable
        :return:
        """
        ch_int = 2 ** int(ch_dict['outCh'][2:])  # 'DO2' -> 2
        n_of_pulses = ch_dict['numOfPulses']
        width_us = ch_dict['widthUs']
        delay_us = ch_dict['delayUs']
        inverted = ch_dict['inverted']
        hi_lo_list = []
        hi_lo_list.append([ch_int if inverted else 0, t_0, ch_int])
        t_first_pulse = t_0 + delay_us
        hi_lo_list.append([ch_int if not inverted else 0, t_first_pulse, ch_int])  # first pulse
        t_pulse_end = t_first_pulse + width_us
        hi_lo_pulse_end = inverted
        hi_lo_list.append([ch_int if hi_lo_pulse_end else 0, t_pulse_end, ch_int])
        for i in range(1, n_of_pulses):  # one pulse already done -> start with 1 instead of 0
            hi_lo_list.append([ch_int if hi_lo_list[-2][0] else 0, hi_lo_list[-2][1] + sys_per_us, ch_int])
            hi_lo_list.append([ch_int if hi_lo_list[-2][0] else 0, hi_lo_list[-2][1] + sys_per_us, ch_int])
        # in the end keep this state for sys_per - delay - ch width
        hi_lo_list.append(
            [ch_int if hi_lo_list[-1][0] else 0, hi_lo_list[-1][1] + sys_per_us - delay_us - width_us, ch_int])
        return hi_lo_list

    def get_ch_high_low_at_time(self, time, ch_hi_lo_list):
        """ check if the ch_ch_hi_lo_list is high or low at the given time and
         return ([ch_int, time, ch_int], index of this element in ch_hi_lo_list) """
        time_hi_lo = [each for each in ch_hi_lo_list if each[1] <= time]  # each: [ch_int, time, True/False]
        index = len(time_hi_lo) - 1
        if index == -1:  # not in list ...
            time_hi_lo_ele = [0, 0, 0]
        else:
            time_hi_lo_ele = time_hi_lo[-1]
        time_hi_lo_ele[1] = time
        return time_hi_lo_ele, index

    def combine_ch_high_low_lists(self, all_ch_high_lo_lists):
        """ by iteratively calling this function, add up all all single channel high low list to a common one """
        ret_list = []
        timing = []
        for singl_ch_high_lo_lis in all_ch_high_lo_lists:
            timing += [each[1] for each in singl_ch_high_lo_lis]
        timing = sorted(set(timing))
        all_same_len_sorted = []
        for singl_ch_high_lo_lis in all_ch_high_lo_lists:
            for each_time in timing:
                elem, ind = self.get_ch_high_low_at_time(each_time, deepcopy(singl_ch_high_lo_lis))
                if each_time != singl_ch_high_lo_lis[ind][1]:
                    singl_ch_high_lo_lis.insert(ind, elem)
            new_sorted = list(sorted(singl_ch_high_lo_lis, key=lambda _each: _each[1]))
            all_same_len_sorted.append(new_sorted)
        for ind, _time in enumerate(timing):
            ret_list.append(
                [_time,
                 sum([all_same_len_sorted[ch_ind][ind][0] for ch_ind in range(0, len(all_same_len_sorted))])]
            )
        return ret_list

    def ch_hi_lo_list_to_cmd_list(self, ch_hi_lo_list):
        """ create a cmd list (['$cmd::0.5::1::0', ...]) from a channel high low list """
        cmd_list = []
        for i, each in enumerate(ch_hi_lo_list):
            if i < len(ch_hi_lo_list) - 1:
                cmd_len = ch_hi_lo_list[i + 1][0] - each[0]
                cmd = '$time::%.2f::%s::0' % (cmd_len, each[1])
                cmd_list.append(cmd)
        return cmd_list

    def list_view_was_changed(self):
        """ since it is for now not possible to create a peridic pattern
         from the list of commands this tab must be cleared """
        if self.list_of_item_dicts:
            self._backup = deepcopy(self.list_of_item_dicts)
        self.list_of_item_dicts = []
        self.listWidget_periodic_pattern.clear()

    def return_periodic_list(self):
        return self.list_of_item_dicts

    def reload_last_settings(self):
        """ load the backup data (only runtime) """
        if self._backup:
            logging.info('loading backup list of commands: %s ' % self._backup)
            self.list_of_item_dicts = []
            self.listWidget_periodic_pattern.clear()
            self.setup_from_list(self._backup)
            self.get_cmd_list()
        else:
            logging.debug('no backup data available')
