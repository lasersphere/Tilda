"""

Created on '30.05.2016'

@author:'simkaufm'

Description:

Class to wrap all Multimeters
functions that should be accessible:
range
input resistance
precision
trigger source


"""

from Driver.DigitalMultiMeter.NI4071 import Ni4071
from Driver.DigitalMultiMeter.DMMdummy import DMMdummy


class DMMControl:
    def __init__(self):
        self.types = ['Ni4071', 'dummy']
        self.dmm = {}

    def find_dmm_by_type(self, type_str, address):
        """
        find the class for the given type of dmm and initiate it at the given address
        :param type_str: str, type of dmm
        :param address: str, adress of the device (Slot, IP, COM Port etc.)
        :return:str, name of the initiated dev
        """
        dev = None
        name = ''
        print('preparing ', type_str, address)
        if type_str == 'Ni4071':
            try:
                dev = Ni4071(address_str=address)
                name = dev.name
            except Exception as e:
                print('starting dmm did not work exception is:', e)
        elif type_str == 'dummy':
            try:
                dev = DMMdummy(address_str=address)
                name = dev.name
            except Exception as e:
                print('starting dmm did not work exception is:', e)
        if dev is not None:
            self.dmm[dev.name] = dev
            return name

    def config_dmm(self, dmm_name, config_dict, reset_dev):
        """
        configure the dmm
        :param dmm_name: str, name of the dev and key in self.dmm
        :param config_dict: dict, dictionary for the given dmm
        :param reset_dev: bool, True for resetting the device before configuration
        """
        self.dmm[dmm_name].load_from_config_dict(config_dict, reset_dev)

    def start_measurement(self, dmm_name):
        """
        call this to start a measurement on the given dmm
        must have been configured in advanced!
        :param dmm_name: str, name of dev
        """
        self.dmm[dmm_name].initiate_measurement()

    def get_raw_config_pars(self, dmm_name):
        """
        return all needed config parameters as a dict of tuples:
        key: (name_str, type, valid_vals)
        :param dmm_name: str, name of the dmm
        :return: dict, key: (name_str, type, valid_vals)
        """
        print('dmm to emit:', self.dmm)
        return self.dmm[dmm_name].emit_config_pars()
        # use dicts to specify for the individual dmm

    def read_from_multimeter(self, dmm_name):
        """
        function to read all available values from the multimeter
        :param dmm_name: str, name of the dmm, 'all' to read all active
        :return: dict, {key=dmm_name: np.array=read values}
        """
        ret = {dmm_name: self.dmm[dmm_name].fetch_multiple_meas(-1)}  # -1 to read all available values
        return ret

    def read_from_all_active_multimeters(self):
        """
        reads all available values from all active dmms
        :return: dict, key is name of dmm
        """
        ret_dict = {}
        act_dmms = list(self.dmm.keys())
        if len(act_dmms):
            for dmm_name in act_dmms:
                ret_dict[dmm_name] = self.read_from_multimeter(dmm_name)[dmm_name]
            return ret_dict
        else:
            return None
    # maybe feed this to pipeline directly later on.

    def de_init_dmm(self, dmm_name):
        """
        deinitialize the given multimeter and remove it from the self.dmm dictionary
        :param dmm_name: str, name of the given device.
        """
        if dmm_name == 'all':
            for dmm_name in list(self.dmm.keys()):
                self.dmm[dmm_name].de_init_dmm()
            self.dmm = {}
            return None
        self.dmm[dmm_name].de_init_dmm()
        self.dmm.pop(dmm_name)