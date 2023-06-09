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

import logging
from copy import deepcopy

from Tilda.Driver.DigitalMultiMeter.Agilent import Agilent
from Tilda.Driver.DigitalMultiMeter.DMMdummy import DMMdummy
from Tilda.Driver.DigitalMultiMeter.NI4071 import Ni4071
from Tilda.Driver.DigitalMultiMeter.AgilentM918x import AgilentM918x
from Tilda.Driver.DigitalMultiMeter.Agilent3458A import Agilent3458A


class DMMControl:
    def __init__(self):
        self.types = ['Ni4071', 'dummy', 'Agilent_34461A',
                      'Agilent_34401A', 'Agilent_M918x', 'Agilent_3458A']
        self.dmm = {}
        # dict for storing all active dmm objects.
        # key is the name of the device, which is the type_address
        # and should therefore be unique anyhow

    def find_dmm_by_type(self, type_str, address):
        """
        find the class for the given type of dmm and initiate it at the given address
        :param type_str: str, type of dmm
        :param address: str, adress of the device (Slot, IP, COM Port etc.)
        :return:str, name of the initiated dev
        """
        dev = None
        name = ''
        logging.info('preparing digital multimeter of type: %s at address: %s' % (type_str, address))
        if type_str == 'Ni4071':
            try:
                dev = Ni4071(address_str=address)
                name = dev.name  # 'type_addr'
            except Exception as e:
                logging.error('starting dmm (type: %s, addr.: %s) did not work exception is: %s'
                              % (type_str, address, e), exc_info=True)
        elif type_str == 'dummy':
            try:
                dev = DMMdummy(address_str=address)
                name = dev.name  # 'type_addr'
            except Exception as e:
                logging.error('starting dmm (type: %s, addr.: %s) did not work exception is: %s'
                              % (type_str, address, e), exc_info=True)
        elif type_str == 'Agilent_34461A':
            try:
                dev = Agilent(address_str=address, type_num='34461A')
                name = dev.name  # 'type_addr'
            except Exception as e:
                logging.error('starting dmm (type: %s, addr.: %s) did not work exception is: %s'
                              % (type_str, address, e), exc_info=True)
        elif type_str == 'Agilent_34401A':
            try:
                dev = Agilent(address_str=address, type_num='34401A')
                name = dev.name  # 'type_addr'
            except Exception as e:
                logging.error('starting dmm (type: %s, addr.: %s) did not work exception is: %s'
                              % (type_str, address, e), exc_info=True)
        elif type_str == 'Agilent_M918x':
            try:
                dev = AgilentM918x(address_str=address)
                name = dev.name  # 'type_addr'
            except Exception as e:
                logging.error('starting dmm (type: %s, addr.: %s) did not work exception is: %s'
                              % (type_str, address, e), exc_info=True)
        elif type_str == 'Agilent_3458A':
            try:
                dev = Agilent3458A(address_str=address)
                name = dev.name  # 'type_addr'
            except Exception as e:
                logging.error('starting dmm (type: %s, addr.: %s) did not work exception is: %s'
                              % (type_str, address, e), exc_info=True)
        if dev is not None:
            self.dmm[dev.name] = dev
            return name
        else:
            return None

    def config_dmm(self, dmm_name, config_dict, reset_dev):
        """
        configure the dmm
        :param dmm_name: str, name of the dev and key in self.dmm
        :param config_dict: dict, dictionary for the given dmm
        :param reset_dev: bool, True for resetting the device before configuration
        """
        logging.info('active dmms: ' + str(self.dmm))
        logging.debug('loading from config: ' + str(config_dict))
        self.dmm[dmm_name].load_from_config_dict(config_dict, reset_dev)

    def start_measurement(self, dmm_name):
        """
        call this to start a measurement on the given dmm
        must have been configured in advanced!
        :param dmm_name: str, name of dev
        """
        if dmm_name == 'all':
            for dmm_name in list(self.dmm.keys()):
                self.dmm[dmm_name].initiate_measurement()
        elif dmm_name in list(self.dmm.keys()):
            self.dmm[dmm_name].initiate_measurement()

    def start_pre_configured_meas(self, dmm_name, pre_config_name):
        """
        set the dmm to a predefined configuration in that it reads out a value every now and then.
        this will configure the dmm and afterwards initiate the measurement directly.
        """
        logging.info('starting %s meas on dmm: %s' % (pre_config_name, dmm_name))
        if dmm_name == 'all':
            for dmm_name in list(self.dmm.keys()):
                self.dmm[dmm_name].set_to_pre_conf_setting(pre_config_name)
        elif dmm_name in list(self.dmm.keys()):
            self.dmm[dmm_name].set_to_pre_conf_setting(pre_config_name)

    def stopp_measurement(self, dmm_name):
        """
        aborts the measurement on the dmm
        :param dmm_name:str, name of dev = type_addr
        :return:
        """
        if dmm_name == 'all':
            for dmm_name in list(self.dmm.keys()):
                self.dmm[dmm_name].abort_meas()
        else:
            if dmm_name in list(self.dmm.keys()):
                self.dmm[dmm_name].abort_meas()

    def get_raw_config_pars(self, dmm_name):
        """
        return all needed config parameters as a dict of tuples:
        key: (name_str, type, valid_vals)
        :param dmm_name: str, name of the dmm
        :return: dict, key: (name_str, type, valid_vals)
        """
        # print('dmm to emit:', dmm_name, self.dmm)
        return self.dmm[dmm_name].emit_config_pars()
        # use dicts to specify for the individual dmm

    def get_available_preconfigs(self, dmm_name):
        """
        get a list of all available config parameters
        :param dmm_name: str, name of dev
        :return:
        """
        configs = self.dmm[dmm_name].pre_configs
        for enum in configs:
            self.dmm[dmm_name].get_accuracy(enum.value)
        return configs

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

    def get_active_dmms(self):
        """
        function to return a dict of all active dmms
        :return: dict of tuples, {dmm_name: (type_str, address_str, state_str, last_readback, configPars_dict)}
        """
        ret = {}
        for key, val in self.dmm.items():
            ret[key] = (self.dmm[key].type, self.dmm[key].address, self.dmm[key].state,
                        self.dmm[key].last_readback, self.get_raw_config_pars(key))
        return ret

    def software_trigger_dmm(self, dmm_name):
        """
        fire a software trigger to the dmm in order that it will measure a value.
        :param dmm_name: str, name of dmm 'all' for all dmms
        """
        if dmm_name == 'all':
            for dmm_name in list(self.dmm.keys()):
                self.dmm[dmm_name].send_software_trigger()
        else:
            self.dmm[dmm_name].send_software_trigger()

    def get_accuracy(self, dmm_name, config):
        """ get the accuracy tuple from the dmm with the given config """
        return deepcopy(self.dmm[dmm_name].get_accuracy(config))

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
        if self.dmm.get(dmm_name, None) is not None:
            self.dmm[dmm_name].de_init_dmm()
            self.dmm.pop(dmm_name)

# if __name__ == "__main__":
#     inst = DMMControl()
#     dmm_name = inst.find_dmm_by_type('Ni4071', 'PXI1Slot5')
#     # conf = inst.dmm[dmm_name].config_dict
#     # print(conf)
#     # conf['triggerSource'] = 'interval'
#     # print('raw:', inst.get_raw_config_pars(dmm_name))
#     # inst.config_dmm(dmm_name, conf, True)
#     inst.start_periodic_measurement('all')
#     # inst.start_periodic_measurement(dmm_name)
#     while True:
#         print(inst.read_from_all_active_multimeters())  # fix it!
#         # print(inst.read_from_multimeter(dmm_name))
#         time.sleep(0.2)
