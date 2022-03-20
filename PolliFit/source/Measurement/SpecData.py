"""
Created on 29.03.2014

@author: hammen
"""

import itertools as it
from datetime import datetime

import numpy as np

from enum import Enum


class SpecDataXAxisUnits(Enum):
    line_volts = 'line voltage / V'  # from DAC
    total_volts = 'total volts / V'  # from  AccVolt + Offset + line_volt * Kepco (usually after preProc)
    dac_register_bits = 'DAC register bits / a.u.'
    # when calibrating an DAC it might be usefull to have DAC register bits as x axis
    frequency_mhz = 'frequency / MHz'  # for plotting etc.
    scraper_mm = 'translation / mm'  # for plotting Scraper translation
    time_mus = 'time / mus'  # for plotting time in mus
    time_ms = 'time / ms'  # for plotting time in ms
    time_s = 'time / s'  # for plotting time in s
    not_defined = 'not defined'  # arbitrary but, maybe useful in rare cases


class SpecData(object):
    """
    This object contains a general spectrum with multiple tracks and multiple scalers.
    """

    def __init__(self):
        self.path = None  # str, path of the file
        self.type = None  # str, isotope name
        self.line = None  # str, lineVar
        self.date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')  #
        self.nrLoops = [1]  # list of integers, len=nrTracks, holds how often the track was repeated
        self.nrTracks = 1  # int, number of tracks in scan.
        self.nrScalers = []  # list, len = nrTracks, holds the number of scaler for each track
        self.accVolt = None  # float, acceleration voltage of the ions
        self.laserFreq = 0  # float, fundamental laser frequency in MHz in the laser lab system
        self.col = False  # bool, collinear = True, anticollinear = False
        self.dwell = 0  # float or list of lists, depending on importer
        self.seq_type = ''  # str where it can be defeinde what type of sequencer was used,
        #  e.g. trs/cs/kepco/csdummy/trsdummy for xml files
        
        self.offset = None  # float, measured offset pre scan, take mean if multiple ones measured
        self.lineMult = None  # float, applied_voltage = (DAC_voltage * lineMult + lineOffset) * voltDivRatio
        self.lineOffset = None  # float, offset of the DAC at 0V set
        self.voltDivRatio = None  # dict, {'accVolt': , 'offset'

        self.x_units_enums = SpecDataXAxisUnits  # to choose from
        self.x_units = SpecDataXAxisUnits.line_volts  # unit of the x axis

        self.scan_dev_dict_tr_wise = []  # list of scan_device_dicts holds the info of the used scanning device

        # Data is organized as list of tracks containing arrays with information
        self.x = []
        self.cts = []
        self.err = []

    def getSingleSpec(self, scaler, track):
        """ Return a tuple with (volt, cts, err) of the specified scaler and track. -1 for all tracks """
        if track == -1:
            return (np.array([i for i in it.chain(*self.x)]),
                    np.array([i for i in it.chain(*(t[scaler] for t in self.cts))]),
                    np.array([i for i in it.chain(*(t[scaler] for t in self.err))]))
        else:
            return np.array(self.x[track]), np.array(self.cts[track][scaler]), np.array(self.err[track][scaler])

    def getArithSpec(self, scaler, track_index, function=None, eval_on=True):
        """
        calculates the arithmetic spectrum from the users function
        :param scaler: list of scalers used for the arithmetic
        :param track_index: list of int, defining used tracks, -1 for all
        :param function: str, users function
        :param eval_on: boolean, defines if evaluation is needed
        :return: 3 ndarrays, dac-voltage, evaluated counts, evaluated errors(only for simple sum or div functions)
        """
        ''' storage for (volt, err) '''
        n = self.getNrSteps(track_index)
        flatx = np.zeros((n,))  # voltage
        flatc = np.zeros((n,))  # counts
        flate = np.zeros((n,))  # uncertainties

        if function is None:
            function = '[' + ''.join(str(e) for e in scaler) + ']'

        if not eval_on or function is None or function[0] == '[':  # check if list mode and if first time
            ''' list mode'''

            ''' get maximum number of scalers '''
            if isinstance(self.nrScalers, list):
                if track_index == -1:
                    nrScalers = self.nrScalers[0]
                else:
                    nrScalers = self.nrScalers[track_index]
            else:
                nrScalers = self.nrScalers

            ''' go throug all scalers used for sum and add counts up'''
            for s in scaler:
                s = int(s)
                if nrScalers > np.abs(s):  # check if scaler exists
                    flatx, c, e = self.getSingleSpec(abs(s), track_index)
                    flatc = flatc + np.copysign(np.ones_like(c), s) * c
                    flate = flate + np.square(e)
                else:
                    pass
            flate = np.sqrt(flate)
        else:
            ''' function mode '''

            var_map_cts = {}    # dict, mapping variables to related array with counts
            if len(scaler) == 0:
                raise Exception('No scaler used')
            for v in scaler:    # go through used variables
                pmt = 's' + str(v)  # create PMT - name (e. g. s1, s3, ...)
                (flatx, var_map_cts[pmt], e) = self.getSingleSpec(abs(v), track_index)    # get voltage, counts and err
                var_map_cts[pmt] = var_map_cts[pmt].astype(int)  # TODO: Is this really necessary?
                flate = flate + np.square(e)  # sum squared errors of each scaler used TODO: Estimate correct error?
            flatc = eval(function, var_map_cts)  # evaluation of counts
            flate = np.sqrt(flate)

        return flatx, flatc, flate

    """Old version of get ArithSpec:"""

    """
    def getArithSpec(self, scaler, track_index):    #TODO new arith
        '''Same as getSingleSpec, but scaler is of type [+i, -j, +k], resulting in s[i]-s[j]+s[k]'''
        l = self.getNrSteps(track_index)
        flatx = np.zeros((l,))
        flatc = np.zeros((l,))
        flate = np.zeros((l,))
        if isinstance(self.nrScalers, list):
            if track_index == -1:
                nrScalers = self.nrScalers[0]
            else:
                nrScalers = self.nrScalers[track_index]
        else:
            nrScalers = self.nrScalers

        for s in scaler:
            s = int(s)
            if nrScalers >= np.abs(s):
                flatx, c, e = self.getSingleSpec(abs(s), track_index)   #TODO use this scheme to calc total spec
                flatc = flatc + np.copysign(np.ones_like(c), s) * c
                flate = flate + np.square(e)
            else:
                pass
        flate = np.sqrt(flate)

        # cut_lower = 50
        # cut_upper = 100
        # flatx = list(flatx)
        # flate = list(flate)
        # flatc = list(flatc)
        # flatx = flatx[0:cut_lower] + flatx[cut_upper:]
        # flatc = flatc[0:cut_lower] + flatc[cut_upper:]
        # flate = flate[0:cut_lower] + flate[cut_upper:]

        return flatx, flatc, flate
    """

    def getNrSteps(self, track):
        if track == -1:
            return sum(map(len, self.x))
        else:
            return len(self.x[track])

    def _normalizeTracks(self):
        """ Check whether a different number of loops was used for the different tracks and correct """
        maxLoops = max(self.nrLoops)
        for i in range(0, self.nrTracks):
            if self.nrLoops[i] < maxLoops:
                self._multScalerCounts(i, maxLoops / self.nrLoops[i])
    
    def _multScalerCounts(self, scaler, mult):
        """ Multiply counts and error of a specific scaler by mult, according to error propagation """
        self.cts[scaler] *= mult
        self.err[scaler] *= mult
        
    def deadtimeCorrect(self, scaler, track):
        for i, cts in enumerate(self.cts[track][scaler]):
            self.cts[track][scaler][i] = (cts * (self.nrLoops[track] * self.dwell)) / (
                1 - (cts * (self.nrLoops[track]*self.dwell)) * 1.65e-8
            ) / (self.nrLoops[track] * self.dwell)
        

if __name__ == '__main__':
    test = SpecData()
    print(test.date)
    print(test.x_units.value)
    test.x_units = test.x_units_enums.frequency_mhz
    print(test.x_units.value)
