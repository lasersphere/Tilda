"""

Created on '10.05.2021'

@author:'lrenth'

"""

import os


class Options:
    """
    A class for storing local option settings
    """

    def __init__(self):
        self.freq_dict = {}    # dictionary of all (Matisse) frequencies
        self.freq_arith = ""   # string for arithmetic function to calc frequencies from dictionary

    def set_freq(self, dic, arith):
        """
        :param dic: dictionary of frequencies
        :param arith: string containing arithmetic to calculate total frequency
        """
        self.freq_dict = dic
        self.freq_arith = arith
