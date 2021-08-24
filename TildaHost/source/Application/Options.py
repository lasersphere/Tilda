"""

Created on '10.05.2021'

@author:'lrenth'

"""

import os
import ast
import configparser
import yaml
import logging
import Physics


class Options:
    """
    A class for storing local option settings
    """

    def __init__(self, file_path):
        #self.format = 'YAML'  # Options 'YAML', 'INI'
        # define path to local options.yaml
        self.options_file_path = file_path
        path, ext = file_path.split('.')
        self.default_file = path + '_default.' + ext

    def load_from_file(self, default=False):
        """
        Read option.yaml and save them to the options object.
        If no options.yaml exists create one with the standard values defined in options_default.yaml
        """
        if default: # read default options
            logging.info('loading default TILDA settings from options_default.yaml')
            self.config_dict = yaml.safe_load(open(self.default_file))
        elif os.path.isfile(self.options_file_path):  # check if yaml-file already exists
            # read options.yaml
            logging.info('loading local TILDA settings from options.yaml')
            self.config_dict = yaml.safe_load(open(self.options_file_path))
        else:  # file does not exist yet, create new one and warn user.
            logging.warning('No options.yaml found, creating a new one with default values.')
            self.load_from_file(default=True)   # load default options
            self.save_to_file()

    def save_to_file(self):
        """
        Save this Options instance to the local yaml file.
        :return:
        """
        stream = open(self.options_file_path, 'w')
        yaml.dump(self.config_dict, stream, default_flow_style=False)
        # default_flow_style=False: Always use block style instead of flow style
        stream.close()
        logging.info('Updated options.yaml')

    def set_freq(self, dic, arith):
        """
        :param dic: dictionary of frequencies
        :param arith: string containing arithmetic to calculate total frequency
        """
        self.config_dict['FREQUENCY']['freq_dict'] = dic
        self.config_dict['FREQUENCY']['arithmetic'] = arith

    def get_freq_settings(self):
        freq_list = self.config_dict['FREQUENCY']['freq_dict']  # dictionary with frequency names and values in MHz
        freq_arith = self.config_dict['FREQUENCY']['arithmetic']  # arithmetic to calculate total frequency
        return freq_list, freq_arith

    def get_abs_freq(self):
        """
        use the settings from FREQUENCY section to determine the absolute laser frequency
        :return: frequency in cm-1
        """
        freq_MHz = eval(self.config_dict['FREQUENCY']['arithmetic'],
                        {'__builtins__': None},
                        self.config_dict['FREQUENCY']['freq_dict'])
        return Physics.wavenumber(freq_MHz)  # Main takes frequency in cm-1
