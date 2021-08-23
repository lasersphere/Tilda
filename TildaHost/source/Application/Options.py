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
        self.format = 'YAML'  # Options 'YAML', 'INI'  # TODO: Just for testing our options. Finally choose one!
        # define path to local options.ini
        self.options_file_path = file_path

        # TODO: We could do it like below or we could create an actual options_template file...
        # create dictionary with sections
        self.config_dict = dict(FREQUENCY={},)
        # populate sections and set standard values
        self.config_dict['FREQUENCY']['freq_dict'] = {'freq1': 374440780}  # dictionary of all (Matisse) frequencies
        self.config_dict['FREQUENCY']['arithmetic'] = '4 * freq1'  # string for arithmetic function to calc frequencies from dictionary

    def load_from_file(self):
        """
        Read option.ini and save them to the options object.
        If no options.ini exists create one with the standard values defined in __init__()
        """
        # TODO: Decide for one format and keep it!
        if self.format == 'YAML':
            if os.path.isfile(self.options_file_path):  # check if ini-file already exists
                # read options.ini
                logging.info('loading local TILDA settings from options.ini')
                self.config_dict = yaml.safe_load(open(self.options_file_path))
            else:  # file does not exist yet, create new one and warn user.
                logging.warning('No options.ini found, creating a new one with default values.')
                self.save_to_file()

        elif self.format == 'INI':
            config = configparser.ConfigParser()
            config.sections()
            if os.path.isfile(self.options_file_path):  # check if ini-file already exists
                # read options.ini
                logging.info('loading local TILDA settings from options.ini')
                config.read(self.options_file_path)
                # convert to config_dict
                for section in config.sections():
                    for key, val in config.items(section):
                        if "{" in val:
                            self.config_dict[section][key] = ast.literal_eval(val)
                        else:
                            self.config_dict[section][key] = val
            else:  # .ini does not exist yet, create new one and warn user.
                logging.warning('No options.ini found, creating a new one with default values.')
                self.save_to_file()

    def save_to_file(self):
        """
        Save this Options instance to the local ini file.
        :return:
        """
        if self.format == 'YAML':
            stream = open(self.options_file_path, 'w')
            yaml.dump(self.config_dict, stream, default_flow_style=False)
            # default_flow_style=False: Always use block style instead of flow style
            stream.close()
            logging.info('Updated options.ini')

        elif self.format == 'INI':
            config = configparser.ConfigParser()
            for section, settings in self.config_dict.items():
                # convert dictionary to config item
                config[section] = settings
            with open(self.options_file_path, 'w') as optionsfile:  # rewrite ini-File
                config.write(optionsfile)
            logging.info('Updated options.ini')

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
