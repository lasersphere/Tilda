"""
Created on 

@author: simkaufm

Module Description: Test Module for checking if voltage conversion is ok.
"""

import unittest

import numpy as np

import Service.VoltageConversions.VoltageConversions as VCon

import Service.Formating as Form

scan_d = {'track0': {
    'workingTime': None, 'nOfSteps': 5, 'dacStepSize18Bit': 10, 'dacStartRegister18Bit': 0, 'nOfScans': 30,
    'colDirTrue': False, 'postAccOffsetVoltControl': 2, 'waitAfterReset25nsTicks': 4000,
    'postAccOffsetVolt': 1000, 'dwellTime10ns': 2000000, 'activePmtList': [0, 1],
    'waitForKepco25nsTicks': 40, 'invertScan': False, 'nOfCompletedSteps': 0},
    'track1': {
    'workingTime': None, 'nOfSteps': 5, 'dacStepSize18Bit': -10, 'dacStartRegister18Bit': 40, 'nOfScans': 30,
    'colDirTrue': False, 'postAccOffsetVoltControl': 2, 'waitAfterReset25nsTicks': 4000,
    'postAccOffsetVolt': 1000, 'dwellTime10ns': 2000000, 'activePmtList': [0, 1],
    'waitForKepco25nsTicks': 40, 'invertScan': False, 'nOfCompletedSteps': 0},
    'pipeInternals': {'workingDirectory': None, 'activeTrackNumber': 0, 'curVoltInd': 0, 'activeXmlFilePath': None},
    'isotopeData': {
        'laserFreq': '12568.766', 'isotope': '44Ca', 'accVolt': '9999.8',
        'nOfTracks': '1', 'version': '1.06', 'type': 'cs'},
    'measureVoltPars': {'measVoltTimeout10ns': 100, 'measVoltPulseLength25ns': 400}
    }

volt_array = np.asarray([[0, 10, 20, 30, 40], [40, 30, 20, 10, 0]])


class Test_VoltageConversion(unittest.TestCase):

    def test_array(self):
        np.testing.assert_array_equal(Form.create_x_axis_from_scand_dict(scan_d), volt_array)

    def test_stop_bit(self):
        for i, t in enumerate(['track0', 'track1']):
            start = scan_d[t]['dacStartRegister18Bit']
            step = scan_d[t]['dacStepSize18Bit']
            num_steps = scan_d[t]['nOfSteps']
            self.assertEqual(VCon.calc_dac_stop_18bit(start, step, num_steps), volt_array[i][-1])

    def test_stepsize(self):
        for i, t in enumerate(['track0', 'track1']):
            start = scan_d[t]['dacStartRegister18Bit']
            num_steps = scan_d[t]['nOfSteps']
            step = scan_d[t]['dacStepSize18Bit']
            stop = volt_array[i][-1]
            self.assertEqual(VCon.calc_step_size(start, stop, num_steps), step)

    def test_n_of_steps(self):
        for i, t in enumerate(['track0', 'track1']):
            start = scan_d[t]['dacStartRegister18Bit']
            num_steps = scan_d[t]['nOfSteps']
            step = scan_d[t]['dacStepSize18Bit']
            stop = volt_array[i][-1]
            self.assertEqual(VCon.calc_n_of_steps(start, stop, step), num_steps)
