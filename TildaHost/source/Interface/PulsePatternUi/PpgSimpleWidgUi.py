"""

Created on '18.01.2017'

@author:'simkaufm'

Module description:
Widget for creating a command list for the pulse generator with the standard tasks
of opening and closing the beam gate and the rfq.

"""

from copy import deepcopy
from functools import partial

from PyQt5 import QtWidgets

from Interface.PulsePatternUi.Ui_PpgSimpleWidg import Ui_PpgSimpleWidg


class PpgSimpleWidgUi(QtWidgets.QWidget, Ui_PpgSimpleWidg):

    def __init__(self, parent, cmd_list_callback_signal):
        super(PpgSimpleWidgUi, self).__init__(parent)
        self.setupUi(self)
        self.show()
        self.sys_rep_rate_us = 0

        self.cmd_list_callback_signal = cmd_list_callback_signal

        self.simple_dict = {}

        self.output_channels = ['DO%s' % i for i in range(0, 33)]
        self.input_channels = ['DI%s' % i for i in range(0, 8)]

        ''' fill comboboxes with possible values '''
        self.comboBox_proton_trig_yes_no.addItems(['yes', 'no'])
        self.comboBox_proton_trig_input_ch.addItems(self.input_channels)

        self.comboBox_rfqcb_out_ch.addItems(self.output_channels)
        self.comboBox_rfqcb_out_ch.removeItem(1)  # remove DO1
        self.comboBox_rfq_state.addItems(['pulsed', 'always open', 'always closed'])

        self.comboBox_beam_gate_out_ch.addItems(self.output_channels[1:])
        self.comboBox_beam_gate_state.addItems(['pulsed', 'always open', 'always closed'])

        ''' connect comboboxes etc. '''
        self.comboBox_proton_trig_yes_no.currentIndexChanged.connect(self.gui_changed)
        self.comboBox_proton_trig_input_ch.currentIndexChanged.connect(self.gui_changed)

        self.comboBox_rfq_state.currentIndexChanged.connect(self.gui_changed)
        self.comboBox_rfqcb_out_ch.currentTextChanged.connect(partial(self.combobox_output_changed, 'rfq'))
        self.spinBox_rfqcb_num_of_bunches.valueChanged.connect(self.gui_changed)
        self.doubleSpinBox_rfqcb_delay_ms.valueChanged.connect(self.gui_changed)
        self.doubleSpinBox_rfqcb_acc_time_ms.valueChanged.connect(self.gui_changed)
        self.doubleSpinBox_rfqcb_release_time_ms.valueChanged.connect(self.gui_changed)

        self.comboBox_beam_gate_state.currentIndexChanged.connect(self.gui_changed)
        self.comboBox_beam_gate_out_ch.currentTextChanged.connect(partial(self.combobox_output_changed, 'beamgate'))
        self.doubleSpinBox_beam_gate_delay_ms.valueChanged.connect(self.gui_changed)
        self.doubleSpinBox_beam_gate_open_ms.valueChanged.connect(self.gui_changed)

        self.doubleSpinBox_system_period.valueChanged.connect(self.sys_rep_rate_changed)

        ''' example: '''
        simple_dict = {
            'protonTrig': 'yes',
            'protonTrigInput': 'DI0',
            'rfqState': 'pulsed',
            'rfqOutpCh': 'DO0',
            'rfqNumOfBunches': 3,
            'rfqDelayMs': 0.02,
            'rfqAccTimeMs': 0.01,
            'rfqRelTimeMs': 0.01,
            'beamGateState': 'always open',
            'beamGateOutpCh': 'DO1',
            'beamGateDelayMs': 0.0,
            'beamGateOpenTimeMs': 0.0
        }
        # self.load_from_simple_dict(simple_dict)

    def load_from_simple_dict(self, simple_dict):
        """ load the values in the simple dict to the gui """
        self.comboBox_proton_trig_yes_no.setCurrentText(simple_dict.get('protonTrig', 'yes'))
        self.comboBox_proton_trig_input_ch.setCurrentText(simple_dict.get('protonTrigInput', 'DI0'))

        self.comboBox_rfq_state.setCurrentText(simple_dict.get('rfqState', 'pulsed'))
        self.comboBox_rfqcb_out_ch.setCurrentText(simple_dict.get('rfqOutpCh', 'DO0'))
        self.spinBox_rfqcb_num_of_bunches.setValue(simple_dict.get('rfqNumOfBunches', 1))
        self.doubleSpinBox_rfqcb_delay_ms.setValue(simple_dict.get('rfqDelayMs', 0))
        self.doubleSpinBox_rfqcb_acc_time_ms.setValue(simple_dict.get('rfqAccTimeMs', 10))
        self.doubleSpinBox_rfqcb_release_time_ms.setValue(simple_dict.get('rfqRelTimeMs', 0.1))

        self.comboBox_beam_gate_state.setCurrentText(simple_dict.get('beamGateState', 'pulsed'))
        self.comboBox_beam_gate_out_ch.setCurrentText(simple_dict.get('beamGateOutpCh', 'DO1'))

        self.doubleSpinBox_beam_gate_delay_ms.setValue(simple_dict.get('beamGateDelayMs', 0))
        self.doubleSpinBox_beam_gate_open_ms.setValue(simple_dict.get('beamGateOpenTimeMs', 10))
        # self.simple_dict = simple_dict

    def combobox_output_changed(self, caller_id):
        """ do not allow the output to be visible in both comboboxes. """

        rfq_cur_text = deepcopy(self.comboBox_rfqcb_out_ch.currentText())
        bg8_cur_text = deepcopy(self.comboBox_beam_gate_out_ch.currentText())
        self.comboBox_beam_gate_out_ch.blockSignals(True)
        self.comboBox_rfqcb_out_ch.blockSignals(True)
        if caller_id == 'rfq':
            self.comboBox_beam_gate_out_ch.clear()
            outpts = [each for each in self.output_channels if rfq_cur_text != each]
            self.comboBox_beam_gate_out_ch.addItems(outpts)
            if bg8_cur_text != rfq_cur_text:
                self.comboBox_beam_gate_out_ch.setCurrentText(bg8_cur_text)
        elif caller_id == 'beamgate':
            self.comboBox_rfqcb_out_ch.clear()
            outpts = [each for each in self.output_channels if bg8_cur_text != each]
            self.comboBox_rfqcb_out_ch.addItems(outpts)
            if bg8_cur_text != rfq_cur_text:
                self.comboBox_rfqcb_out_ch.setCurrentText(rfq_cur_text)
        self.gui_changed()
        self.comboBox_beam_gate_out_ch.blockSignals(False)
        self.comboBox_rfqcb_out_ch.blockSignals(False)

    def gui_changed(self):
        """ this ios called whenever something in the gui changes """
        pr_yes_no = self.comboBox_proton_trig_yes_no.currentText()
        pr_inp = self.comboBox_proton_trig_input_ch.currentText()
        rfq_state = self.comboBox_rfq_state.currentText()
        rfq_out_ch = self.comboBox_rfqcb_out_ch.currentText()
        rfq_num_bunches = self.spinBox_rfqcb_num_of_bunches.value()
        rfq_delay_ms = self.doubleSpinBox_rfqcb_delay_ms.value()
        rfq_acc_time = self.doubleSpinBox_rfqcb_acc_time_ms.value()
        rfq_rel_time = self.doubleSpinBox_rfqcb_release_time_ms.value()
        beam_gate_state = self.comboBox_beam_gate_state.currentText()
        beam_gate_out_ch = self.comboBox_beam_gate_out_ch.currentText()
        beam_gate_delay_ms = self.doubleSpinBox_beam_gate_delay_ms.value()
        beam_gate_open_ms = self.doubleSpinBox_beam_gate_open_ms.value()
        self.simple_dict = {
            'protonTrig': pr_yes_no,
            'protonTrigInput': pr_inp,
            'rfqState': rfq_state,
            'rfqOutpCh': rfq_out_ch,
            'rfqNumOfBunches': rfq_num_bunches,
            'rfqDelayMs': rfq_delay_ms,
            'rfqAccTimeMs': rfq_acc_time,
            'rfqRelTimeMs': rfq_rel_time,
            'beamGateState': beam_gate_state,
            'beamGateOutpCh': beam_gate_out_ch,
            'beamGateDelayMs': beam_gate_delay_ms,
            'beamGateOpenTimeMs': beam_gate_open_ms
        }
        pr_yes_no_bool = pr_yes_no == 'yes'
        # if not proton triggered also repeat the delay
        new_sys_rep_rate_ms = (rfq_acc_time + rfq_rel_time) \
            if pr_yes_no_bool else (rfq_delay_ms + rfq_acc_time + rfq_rel_time)
        self.sys_rep_rate_changed(new_sys_rep_rate_ms)
        self.block_controls()
        cmd_list = self.get_cmd_list_from_simple_dict()
        if cmd_list:
            # print('emitting')
            self.cmd_list_callback_signal.emit(cmd_list, 'simple')

    def sys_rep_rate_changed(self, sys_rep_rate_ms):
        """ system rep rate was changed, adjust accumulation time if this does not match """
        self.doubleSpinBox_system_period.blockSignals(True)
        self.doubleSpinBox_system_period.setValue(sys_rep_rate_ms)
        self.doubleSpinBox_system_period.blockSignals(False)
        self.sys_rep_rate_us = sys_rep_rate_ms * 1000
        pr_yes_no_bool = self.comboBox_proton_trig_yes_no.currentText() == 'yes'
        rfq_acc_time_us = deepcopy(self.doubleSpinBox_rfqcb_acc_time_ms.value() * 1000)
        rfq_rel_time_us = self.doubleSpinBox_rfqcb_release_time_ms.value() * 1000
        rfq_delay_us = self.doubleSpinBox_rfqcb_delay_ms.value() * 1000
        dif = 0
        if pr_yes_no_bool:
            sys_rep_should_be = (rfq_acc_time_us + rfq_rel_time_us)
        else:
            sys_rep_should_be = (rfq_delay_us + rfq_acc_time_us + rfq_rel_time_us)
        dif = self.sys_rep_rate_us - sys_rep_should_be
        if dif:
            new_acc_time = (rfq_acc_time_us + dif) / 1000
            # print('new_acc_time', new_acc_time)
            self.doubleSpinBox_rfqcb_acc_time_ms.setValue(new_acc_time)

    def block_controls(self):
        """ this will activate / deactivate controls depending on if proton triggered or not """
        pr_tr_yes_no = False if self.simple_dict.get('protonTrig', 'no') == 'no' else True
        bg8_state = self.simple_dict.get('beamGateState', 'pulsed')
        bg8_always_open = True if bg8_state in ['always open', 'always closed'] else False
        rfq_state = self.simple_dict.get('rfqState', 'pulsed')
        rfq_always_open = True if rfq_state in ['always open', 'always closed'] else False

        # simplify if proton triggered or not
        self.comboBox_proton_trig_input_ch.setEnabled(pr_tr_yes_no)

        if not pr_tr_yes_no:  # not proton triggered
            self.spinBox_rfqcb_num_of_bunches.blockSignals(True)
            self.spinBox_rfqcb_num_of_bunches.setValue(1)
            self.simple_dict['rfqNumOfBunches'] = 1
            self.spinBox_rfqcb_num_of_bunches.blockSignals(False)

        # simplify if beam gate is always open:
        self.doubleSpinBox_beam_gate_delay_ms.setEnabled(not bg8_always_open)  # and pr_tr_yes_no)
        self.doubleSpinBox_beam_gate_open_ms.setDisabled(bg8_always_open)

        # simplify if rfq always open
        self.spinBox_rfqcb_num_of_bunches.setEnabled(pr_tr_yes_no)
        self.doubleSpinBox_rfqcb_delay_ms.setEnabled(not rfq_always_open)  # and pr_tr_yes_no)
        self.doubleSpinBox_rfqcb_acc_time_ms.setDisabled(rfq_always_open)
        self.doubleSpinBox_rfqcb_release_time_ms.setDisabled(rfq_always_open)

    def get_cmd_list_from_simple_dict(self):
        """ in the simple command dict everything from the gui is stored and
         this holds all relevant info for creating the pulse pattern.
          Therefore this will create the neede command list. """
        cmd_list = []
        pr_tr_yes_no = False if self.simple_dict.get('protonTrig', 'no') == 'no' else True
        pr_tr_inp_int = 2 ** int(self.simple_dict.get('protonTrigInput')[2:])

        rfq_always_open = True if self.simple_dict.get('rfqState', '') == 'always open' else False
        rfq_always_closed = True if self.simple_dict.get('rfqState', '') == 'always closed' else False
        rfq_high_int = 2 ** int(self.simple_dict.get('rfqOutpCh', 'DO0')[2:]) if not rfq_always_open else 0
        rfq_low_int = rfq_high_int if rfq_always_closed else 0

        bg8_always_open = True if self.simple_dict.get('beamGateState', 'pulsed') == 'always open' else False
        bg8_always_closed = True if self.simple_dict.get('beamGateState', 'pulsed') == 'always closed' else False
        bg8_high_int = 2 ** int(self.simple_dict.get('beamGateOutpCh', 'DO0')[2:]) if not bg8_always_closed else 0
        bg8_low_int = bg8_high_int if bg8_always_open else 0  # if bg8 always open set low level to high level
        if pr_tr_yes_no:  # its proton triggered
            # rfq and beamgate closed while waiting if not always open
            cmd_list.append('$wait::%s::%s::0' % (pr_tr_inp_int, bg8_low_int + rfq_low_int))
        cmd_list += self.create_proton_triggered_cmd_ch_list(pr_tr_yes_no)
        cmd_list.append('$stop::0::%s::0' % (bg8_low_int + rfq_low_int))
        # print(cmd_list)
        return cmd_list

    def append_time_cmd_to_list(self, cmd_list, time_us, ch):
        if time_us > 0:
            cmd_list.append('$time::%.2f::%s::0' % (time_us, ch))

    def create_proton_triggered_cmd_ch_list(self, pr_tr_yes_no):
        """ create a cmd list from the self.simple_dict """
        rfq_always_open = True if self.simple_dict.get('rfqState', '') == 'always open' else False
        rfq_always_closed = True if self.simple_dict.get('rfqState', '') == 'always closed' else False
        rfq_num_of_bunches = self.simple_dict.get('rfqNumOfBunches')
        rfq_delay_us = round(self.simple_dict.get('rfqDelayMs') * 1000, 2)  # round to 10ns resolution
        rfq_acc_time_us = round(self.simple_dict.get('rfqAccTimeMs', 0) * 1000, 2)

        bg8_always_open = True if self.simple_dict.get('beamGateState', 'pulsed') == 'always open' else False
        bg8_always_closed = True if self.simple_dict.get('beamGateState', 'pulsed') == 'always closed' else False
        bg8_delay_us = round(self.simple_dict.get('beamGateDelayMs', 0) * 1000, 2)
        bg8_open_time_us = round(self.simple_dict.get('beamGateOpenTimeMs', 0) * 1000, 2)

        ch_hi_lo_list = []
        t_0 = 0
        rfq_ch_str = self.simple_dict.get('rfqOutpCh', 'DO0')
        bg8_ch_str = self.simple_dict.get('beamGateOutpCh', 'DO1')
        widths = [rfq_acc_time_us, bg8_open_time_us]
        delays = [rfq_delay_us, bg8_delay_us]
        always_open = [rfq_always_closed, bg8_always_open]
        always_closed = [rfq_always_open, bg8_always_closed]
        for i, each in enumerate([rfq_ch_str, bg8_ch_str]):
            ch_hi_lo_list.append(
                self.get_ch_high_low_list(
                    each, rfq_num_of_bunches, widths[i],
                    delays[i], self.sys_rep_rate_us, t_0=t_0, pr_triggered=pr_tr_yes_no,
                    always_high=always_open[i], always_low=always_closed[i]
                ))
        ch_hi_lo_list_combined = self.combine_ch_high_low_lists(ch_hi_lo_list)
        cmd_list = self.ch_hi_lo_list_to_cmd_list(ch_hi_lo_list_combined)
        return cmd_list

    def get_ch_high_low_list(self, ch_str, n_of_pulses, width_us,
                             delay_us, sys_per_us, t_0=0, pr_triggered=True, always_high=False, always_low=False):
        """
        create a list of list with each containing: [[ch_int_high/ch_int_low, leftedge_time, ch_int], ... ]
        """
        ch_high_int = 2 ** int(ch_str[2:])  # 'DO2' -> 2 ** 2
        ch_low_int = 0
        if always_high:
            return [[ch_high_int, 0, ch_high_int], [ch_high_int, delay_us + sys_per_us * n_of_pulses, ch_high_int]]
        if width_us == 0 or always_low:
            ch_high_int = 0
            if always_low:
                return [[ch_high_int, 0, ch_high_int], [ch_high_int, delay_us + sys_per_us * n_of_pulses, ch_high_int]]
        inverted = False
        hi_lo_list = []
        t_first_pulse = t_0
        if 0 < delay_us:  # only apply delay when it is larger then 0
            if delay_us < sys_per_us or pr_triggered:
                # print('applying delay', delay_us, sys_per_us, pr_triggered)
                # only use delay if this is not longer than sys period or use when proton triggered
                hi_lo_list.append([ch_high_int if inverted else ch_low_int, t_0, ch_high_int])  # delay
                t_first_pulse = t_0 + delay_us

        if 0 < width_us < sys_per_us:
            hi_lo_list.append([ch_high_int if not inverted else ch_low_int, t_first_pulse, ch_high_int])  # first pulse begin
            t_pulse_end = t_first_pulse + width_us
            hi_lo_pulse_end = inverted
            hi_lo_list.append([ch_high_int if hi_lo_pulse_end else ch_low_int, t_pulse_end, ch_high_int])  # first pulse end
            for i in range(1, n_of_pulses):  # one pulse already done -> start with 1 instead of 0
                hi_lo_list.append([ch_high_int if hi_lo_list[-2][0] else ch_low_int, hi_lo_list[-2][1] + sys_per_us, ch_high_int])  # begin
                hi_lo_list.append([ch_high_int if hi_lo_list[-2][0] else ch_low_int, hi_lo_list[-2][1] + sys_per_us, ch_high_int])  # end
            # in the end keep this state for sys_per - delay - ch width
            if pr_triggered:
                hi_lo_list.append(
                    [ch_high_int if hi_lo_list[-1][0] else ch_low_int, hi_lo_list[-1][1] + sys_per_us - width_us, ch_high_int])
            else:  # for no proton triggered also account the delay to the system period
                remaining_time = sys_per_us - delay_us - width_us
                new_time = hi_lo_list[-1][1] + remaining_time
                if new_time > 0:
                    hi_lo_list.append(
                        [ch_high_int if hi_lo_list[-1][0] else ch_low_int,
                         hi_lo_list[-1][1] + remaining_time, ch_high_int])
                if remaining_time < 0:
                    hi_lo_list.pop(-2)  # pop the last value which was to long beforehand.
        else:
            # width exceeds or is equal to the system period or is zero
            # -> keep state as in first pulse for n_of_pulses * sys_rep_rate_us
            hi_lo_list.append([ch_high_int if not inverted else ch_low_int, t_first_pulse, ch_high_int])  # first pulse
            end_time = t_first_pulse + sys_per_us * n_of_pulses - delay_us if not pr_triggered \
                else t_first_pulse + sys_per_us * n_of_pulses
            hi_lo_list.append(
                [ch_high_int if inverted else ch_low_int, end_time, ch_high_int])  # end
        hi_lo_list = [[each[0], round(each[1], 2), each[2]] for each in hi_lo_list]
        # print(hi_lo_list)
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
        # print(timing)
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

    def return_simple_dict(self):
        """ just use this to return the simple dict with all info """
        return self.simple_dict

    def list_view_was_changed(self):
        """ the listview was chnaged therefore the simple dict need to cleared.
         Values can stay in the gui though. """
        self.simple_dict = {}
