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

import time

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
                name = dev.name  # 'type_addr'
            except Exception as e:
                print('starting dmm did not work exception is:', e)
        elif type_str == 'dummy':
            try:
                dev = DMMdummy(address_str=address)
                name = dev.name  # 'type_addr'
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
        print('active dmms:', self.dmm)
        print('loading from config: ', config_dict)
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
        or None for no reading
        """
        ret = self.dmm[dmm_name].fetch_multiple_meas(-1)  # -1 to read all available values
        if ret.any():  # ret must be numpy array. if it has no values return None
            ret_dict = {dmm_name: ret}
        else:
            ret_dict = None
        return ret_dict

    def read_from_all_active_multimeters(self):
        """
        reads all available values from all active dmms
        :return: dict, key is name of dmm
        or None, if no reading
        """
        ret_dict = {}
        act_dmms = list(self.dmm.keys())
        if len(act_dmms):
            for dmm_name in act_dmms:
                reading = self.read_from_multimeter(dmm_name)  # None for no reading
                if reading is not None:
                    ret_dict[dmm_name] = reading[dmm_name]
                else:
                    ret_dict[dmm_name] = None
            if ret_dict != {}:
                return ret_dict
            else:
                return None
        else:
            return None

    # maybe feed this to pipeline directly later on.

    def get_active_dmms(self):
        """
        function to return a dict of all active dmms
        :return: dict of tuples, {dmm_name: (type_str, address_str, configPars_dict)}
        """
        ret = {}
        for key, val in self.dmm.items():
            ret[key] = (self.dmm[key].type, self.dmm[key].address,
                        self.get_raw_config_pars(key))
        return ret

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


        # if __name__ == "__main__":
        #     inst = DMMControl()
        #     dmm_name = inst.find_dmm_by_type('Ni4071', 'PXI1Slot5')
        #     conf = inst.dmm[dmm_name].config_dict
        #     print(conf)
        #     conf['triggerSource'] = 'interval'
        #     print('raw:', inst.get_raw_config_pars(dmm_name))
        #     inst.config_dmm(dmm_name, conf, True)
        #     inst.start_measurement(dmm_name)
        #     while True:
        #         print(inst.read_from_all_active_multimeters())  # fix it!
        #         # print(inst.read_from_multimeter(dmm_name))
        #         time.sleep(0.2)
