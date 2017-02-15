"""

Created on '12.01.2017'

@author:'simkaufm'

"""

from Driver.COntrolFpga import PulsePatternGeneratorConfig as PPGCfg
from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling


class PulsePatternGeneratorDummy(FPGAInterfaceHandling):
    def __init__(self):
        self.config = PPGCfg
        super(PulsePatternGeneratorDummy, self).__init__(
            self.config.bitfilePath, self.config.bitfileSignature, self.config.fpgaResource, dummy=True)
        self.state_changed_callback_signal = None

    ''' useful functions '''

    def load(self, data, mem_addr=0, start_after_load=True, reset_before_load=True):
        pass

    def start(self, run_continous=True, start_addr=0):
        pass

    def stop(self):
        pass

    def reset(self):
        pass

    def deinit_ppg(self, finalize_com=False):
        pass

    def convert_single_comand(self, cmd_str, ticks_per_us=None):
        pass

    def convert_list_of_cmds(self, cmd_list, ticks_per_us=None):
        pass

    def convert_np_arr_of_cmd_to_list_of_cmds(self, np_arr_cmds, ticks_per_us=None):
        pass

    def convert_int_arr_to_singl_cmd(self, int_arr, ticks_per_us=None):
        pass

    def query_command(self, mem_address=-1):
        pass

    def connect_to_state_changed_signal(self, callback_signal):
        pass

    def disconnect_to_state_changed_signal(self):
        pass

    ''' read Indicators: '''

    def read_fifo_empty(self):
        return False

    def read_start_sctl(self):
        return False

    def read_stop_sctl(self):
        return False

    def read_error_code(self):
        return 0, 'everything fine'

    def read_state(self):
        return 0, 'idle'

    def read_elements_loaded(self):
        return -1

    def read_revision(self):
        return -1

    def read_ticks_per_us(self):
        return 100

    def read_number_of_cmds(self):
        return -1

    def read_stop_addr(self):
        return -1

    ''' write controls '''

    def set_continous(self, cont_bool):
        pass

    def set_load(self, load_bool):
        pass

    def set_query(self, query_bool):
        pass

    def set_replace(self, replace_bool):
        pass

    def set_reset(self, cont_bool):
        pass

    def set_run(self, cont_bool):
        pass

    def set_stop(self, cont_bool):
        pass

    def set_useJump(self, cont_bool):
        pass

    def set_mem_addr(self, mem_addr_int):
        pass

    ''' FiFo Access '''

    def read_from_target(self, num_of_ele=-1):
        pass

    def write_to_target(self, data):
        return 0


