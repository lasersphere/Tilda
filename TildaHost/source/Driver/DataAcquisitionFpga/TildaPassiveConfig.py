"""
Created on 12.04.2016

@author: simkaufm

Module Description: automatically created with the CApiAnalyser
"""

import ctypes

'''Bitfile Signature:'''
bitfileSignature = '3484C4D672E45F0D11A710760B0AB3E8'
'''Bitfile Path:'''
bitfilePath = 'D:/Workspace/PyCharm/Tilda/TildaTarget/bin/TildaPassive/NiFpga_TildaPassiveMain.lvbitx'
'''FPGA Resource:'''
fpgaResource = 'Rio1'
'''Indicators:'''
TildaPassiveStateInd = {'ref': 0x8112, 'val': ctypes.c_uint(), 'ctr': False}
'''Controls:'''
TildaPassiveStateCtrl = {'ref': 0x810E, 'val': ctypes.c_uint(), 'ctr': True}
delay_10ns_ticks = {'ref': 0x8114, 'val': ctypes.c_ulong(), 'ctr': True}
nOfBins = {'ref': 0x8118, 'val': ctypes.c_ulong(), 'ctr': True}
'''TargetToHostFifos:'''
transferToHost = {'ref': 0, 'val': ctypes.c_ulong(), 'ctr': False}

''' hand filled values '''
transferToHostReqEle = 10000000

tilda_passive_states = {'idle': 0, 'scanning': 1, 'error': 2}
default_nOfBins = 2000  # in units of 10ns -> 20 mus dwell
default_delay = 6000  # in units of 10ns -> 60 mus delay

draft_tipa_scan_pars = {
    'pipeInternals': {'workingDirectory': None,
                      'activeTrackNumber': (0, 'track0'),
                      'curVoltInd': 0,
                      'activeXmlFilePath': None
                      },
    'isotopeData': {'version': '1.06',
                    'type': 'tipa',
                    'isotope': 'Ni',
                    'nOfTracks': 1,
                    'accVolt': 0,
                    'laserFreq': 0},
    'track0': {'dacStepSize18Bit': 1,
               'dacStartRegister18Bit': 0,
               'nOfSteps': 5,
               'nOfScans': 10, 'nOfCompletedSteps': 0, 'invertScan': False,
               'activePmtList': [0, 1, 2, 3],
               'colDirTrue': True,
               'workingTime': None,
               'nOfBins': 1000,
               'softBinWidth_ns': 10,
               'nOfBunches': 1,
               'softwGates': [[-10, 10, 0, 10000], [-10, 10, 0, 10000], [-10, 10, 0, 10000], [-10, 10, 0, 10000]],
               'trigger': {'type': 'SingleHit', 'trigDelay10ns': 5000}
               },
    'measureVoltPars': {'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100}
}
