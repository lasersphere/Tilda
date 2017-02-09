"""

Created on '12.01.2017'

@author:'simkaufm'

"""

import ast
import time
from copy import deepcopy

import numpy as np

from Driver.COntrolFpga import PulsePatternGeneratorConfig as PPGCfg
from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling


class PulsePatternGenerator(FPGAInterfaceHandling):
    def __init__(self):
        self.config = PPGCfg
        super(PulsePatternGenerator, self).__init__(
            self.config.bitfilePath, self.config.bitfileSignature, self.config.fpgaResource)
        self.state_changed_callback_signal = None

    ''' useful functions '''

    def load(self, data, mem_addr=0, start_after_load=True, reset_before_load=True):
        """
        will load the data to the ppg
        can only write 4000 data elements per call
        :param data: numpy array containing the data
        :param mem_addr: int, starting position to write to
            -> keep all data with lower index untouched and write new data to index and beyond.
        :return:
        """
        if reset_before_load:
            self.reset()
        to_much_data = None
        state_num, state_name = self.read_state()
        num_of_data = len(data)
        if state_name in ['idle', 'stopped']:
            if num_of_data > 4000:
                to_much_data = deepcopy(data[4000:])
                data = deepcopy(data[:4000])
            self.set_mem_addr(mem_addr)
            self.write_to_target(data)
            self.set_load(True)
            time.sleep(0.002)
            print('%s %s' % self.read_state())
            if to_much_data is not None:
                self.load(to_much_data, mem_addr + 4000)
            num_of_cmds = self.read_number_of_cmds()
            if num_of_cmds == num_of_data // 4:
                if to_much_data is None and start_after_load:
                    self.start()
                # time.sleep(1)
                # print('state is: %s %s' % self.read_state())
                # print('error code: %s %s' % self.read_error_code())
                # print('fpga status : %s ' % self.checkFpgaStatus())
                # print('fifo empty: %s ' % self.read_fifo_empty())
                # print('error code: %s %s' % self.read_error_code())
                # print('stop sctl: %s' % self.read_stop_sctl())
                # print('start sctl: %s' % self.read_start_sctl())
                timeout = 0
                while not self.read_start_sctl():  # only when start_sctl is True, the pattern is really executing!
                    if timeout > 5:
                        print('error: ppg pattern did not start within 5 seconds!')
                        return False
                    timeout += 0.001
                    time.sleep(0.001)
                print('ppg successfully started after about %.2f s' % timeout)
                return True
            else:
                print('error, number of commands (%s)'
                      ' does not match length of data // 4 (%s)' % (num_of_cmds, num_of_data))
                return False
        else:
            print('cannot load data to fpga,'
                  ' since it is not in idle state, state is: %s %s' % (state_num, state_name))

    def start(self, run_continous=True, start_addr=0):
        """
        run the loaded commands
        :param start_addr:
        :return:
        """
        self.set_continous(run_continous)
        state_num, state_name = self.read_state()
        if state_name in ['idle', 'stopped']:
            self.set_mem_addr(start_addr)
            self.set_run(True)
            time.sleep(0.001)
            self.read_state()
            return self.read_error_code()
        else:
            print('cannot start the ppg,'
                  ' since it is not in idle state, state is: %s %s' % (state_num, state_name))

    def stop(self):
        """
        stop ppg and go to idle state
        :return:
        """
        self.set_stop(True)
        time.sleep(0.001)
        self.read_state()

    def reset(self):
        """
        goes to reset state in which all memory entries are deleted.
        :return:
        """
        max_tries = 1000
        state_num, state_name = self.read_state()
        if state_name not in ['idle', 'stopped']:
            self.set_stop(True)
        self.set_reset(True)
        tries = 0
        state_num, state_name = self.read_state()
        while state_name != 'idle' and tries < max_tries:
            time.sleep(0.001)
            state_num, state_name = self.read_state()
            tries += 1

    def deinit_ppg(self):
        """
        stops the fpga
        :return:
        """
        self.DeInitFpga()

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

    def query_command(self, mem_address=-1):
        """
        query the command at the chosen memory address
        :param mem_address: int, memory address at which the command is. -1 for all
        :return: numpy array with the commands
        """
        ret = np.zeros(0, dtype=np.int32)
        state_num, state_name = self.read_state()
        if state_name in ['idle']:
            if mem_address == -1:
                addresses = list(range(self.read_number_of_cmds()))
            else:
                addresses = [mem_address]
            for mem_address in addresses:
                self.set_mem_addr(mem_address)
                self.set_query(True)
                time.sleep(0.001)
                readback = self.read_from_target(4)
                ret = np.append(ret, readback['newData'])
                self.set_query(False)
            return ret
        else:
            print('error not able to querry command from ppg when not in idle state')
            return ret

    def connect_to_state_changed_signal(self, callback_signal):
        """
        use this in order to connect a signal to the state changed function and
         emit the name of the satte each time this is changed.
        :param callback_signal: pyqtboundsignal(str)
        """
        self.state_changed_callback_signal = callback_signal
        self.read_state()

    def disconnect_to_state_changed_signal(self):
        """
        disconnect the signal
        """
        self.state_changed_callback_signal = None

    ''' read Indicators: '''

    def read_fifo_empty(self):
        return self.ReadWrite(self.config.fifo_empty).value

    def read_start_sctl(self):
        return self.ReadWrite(self.config.start_sctl).value

    def read_stop_sctl(self):
        return self.ReadWrite(self.config.stop_sctl).value

    def read_error_code(self):
        """
        will read the error code from the device.
        :return: tuple, (int=errornumber, str=error_message)
        """
        err = self.ReadWrite(self.config.error_code).value
        # err_dict gained from labview
        err_dict = {
            0: 'everything fine',
            1: 'not initialised',
            2: 'invalid start address',
            3: 'wrong time',
            4: 'more than one error'}
        err_message = err_dict.get(err, 'error unknown')
        if err != 0:
            print('error: ppg yields errorcode: %s <-> %s' % (err, err_message))
        return err, err_message

    def read_state(self):
        state_num = self.ReadWrite(self.config.state).value
        state_name = 'not a state'
        if state_num in self.config.ppg_state_dict.values():
            state_name = [state_na for state_na, state_nu in self.config.ppg_state_dict.items()
                          if state_nu == state_num][0]
        if self.state_changed_callback_signal is not None:
            self.state_changed_callback_signal.emit(state_name)
        return state_num, state_name

    def read_elements_loaded(self):
        return self.ReadWrite(self.config.elements_loaded).value

    def read_revision(self):
        return self.ReadWrite(self.config.revision).value

    def read_ticks_per_us(self):
        return self.ReadWrite(self.config.ticks_per_us).value

    def read_number_of_cmds(self):
        return self.ReadWrite(self.config.number_of_cmds).value

    def read_stop_addr(self):
        return self.ReadWrite(self.config.stop_addr).value

    ''' write controls '''

    def set_continous(self, cont_bool):
        self.ReadWrite(self.config.continuous, cont_bool)
        return self.checkFpgaStatus()

    def set_load(self, load_bool):
        self.ReadWrite(self.config.load, load_bool)
        return self.checkFpgaStatus()

    def set_query(self, query_bool):
        self.ReadWrite(self.config.query, query_bool)
        return self.checkFpgaStatus()

    def set_replace(self, replace_bool):
        self.ReadWrite(self.config.replace, replace_bool)
        return self.checkFpgaStatus()

    def set_reset(self, cont_bool):
        self.ReadWrite(self.config.reset, cont_bool)
        return self.checkFpgaStatus()

    def set_run(self, cont_bool):
        self.ReadWrite(self.config.run, cont_bool)
        return self.checkFpgaStatus()

    def set_stop(self, cont_bool):
        self.ReadWrite(self.config.stop, cont_bool)
        return self.checkFpgaStatus()

    def set_useJump(self, cont_bool):
        self.ReadWrite(self.config.useJump, cont_bool)
        return self.checkFpgaStatus()

    def set_mem_addr(self, mem_addr_int):
        self.ReadWrite(self.config.mem_addr, mem_addr_int)
        return self.checkFpgaStatus()

    ''' FiFo Access '''

    def read_from_target(self, num_of_ele=-1):
        """
        read Data from host sided Buffer called 'transferToHost' to an Array.
        Can later be fed into a pipeline system.
        :return: dictionary,
        nOfEle = int, number of Read Elements, newData = numpy Array containing all data that was read
               elemRemainInFifo = int, number of Elements still in FifoBuffer
        """
        result = self.ReadU32Fifo(self.config.DMA_down['ref'], num_of_ele)
        result.update(newData=result['newData'])
        return result

    def write_to_target(self, data):
        """
        function to pass a set of commands to the ppg
        :param data: numpy array containing the commands
        :return:
        """
        still_free = self.WriteU32Fifo(self.config.DMA_up['ref'], data)
        return still_free


if __name__ == '__main__':
    ppg_obj = PulsePatternGenerator()
    # print('ticks: %s ' % ppg_obj.read_ticks_per_us())
    # functions = inspect.getmembers(ppg_obj, predicate=inspect.ismethod)
    # # vererbte functionen haben glücklicherweise keinen unterstrich.
    # functions = [func for func in functions if func[0].find('_') != -1 and func[0] != '__init__']
    # print(functions)
    # for each in functions:
    #     if 'read' in each[0]:
    #         print(each[0], each[1]())
    # example_data_old = np.array(
    #     [
    #         3, 100, 0, 0,
    #         3, 100, 1, 0,
    #         3, 100, 3, 0,
    #         3, 100, 2, 0,
    #         3, 100, 0, 0,
    #         3, 100, 1, 0,
    #         3, 100, 3, 0,
    #         3, 100, 2, 0,
    #         3, 100, 0, 0,
    #         3, 100, 1, 0,
    #         3, 100, 3, 0,
    #         3, 100, 2, 0,
    #     ]
    # )
    # print(example_data_old)
    # ppg_obj.load(example_data_old)
    # ppg_obj.set_continous(True)
    # ppg_obj.start()
    # input('press anything to stop')
    #
    example_data = [
        "$time::1::0::0",
        "$time::1::1::0",
        "$time::1::3::0",
        "$time::1::2::0",
        "$time::1::0::0",
        "$time::1::1::0",
        "$time::1::3::0",
        "$time::1::2::0",
        "$time::1::0::0",
        "$time::1::1::0",
        "$time::1::3::0",
        "$time::1::2::0",
        "$stop::0::1::0",  # always use stop command in the end!
    ]
    example_data = ppg_obj.convert_list_of_cmds(example_data, 100)
    print(example_data)
    ppg_obj.reset()
    ppg_obj.load(example_data, start_after_load=False)
    # input('press anything to stop')
    cmds = ppg_obj.query_command(-1)
    print('cmds from target: %s ' % cmds)
    conv_cmds = ppg_obj.convert_np_arr_of_cmd_to_list_of_cmds(cmds, 100)
    print(conv_cmds)
    ppg_test_path = 'D:\\Debugging\\trs_debug\\Pulsepattern132Pattern.txt'
    # reconv_cmds = ppg_obj.convert_list_of_cmds(conv_cmds, 100)
    # print(reconv_cmds)
    ppg_obj.start()

    # ppg_test_path = 'D:\\Debugging\\trs_debug\\Pulsepattern132Pattern.txt'
    # print(ppg_obj.load_from_file(ppg_test_path, load_to_fpga=True, run_fpga_with_this=True))
    input('press anything to stop')
    print(ppg_obj.read_state())
    ppg_obj.stop()
    input('press anything to stop')
    print(ppg_obj.read_state())

    ppg_obj.deinit_ppg()
