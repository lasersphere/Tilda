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
        if type_str == 'Ni4071':
            dev = Ni4071(address_str=address)
            name = dev.name
        elif type_str == 'dummy':
            dev = DMMdummy(address_str=address)
            name = dev.name
        if dev is not None:
            self.dmm[dev.name] = dev  # will this fail?
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
        return self.dmm[dmm_name].emit_config_pars()
        # use dicts to specify for the individual dmm

    def read_from_multimeter(self, dmm_name):
        """
        function to read all available values from the multimeter
        :param dmm_name: str, name if the dev
        :return: np.array, containing all values
        """
        return self.dmm[dmm_name].fetch_multiple_meas(-1)

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
