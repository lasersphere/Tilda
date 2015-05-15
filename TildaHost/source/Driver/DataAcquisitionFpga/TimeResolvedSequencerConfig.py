"""

Created on '12.05.2015'

@author:'simkaufm'

"""

import ctypes


class TRSConfig():
        """
        Indicators, controls and fifos as gained from C-Api generator
        Using CApiAnalyser.py yields:
        """
        '''Bitfile Signature:'''
        bitfileSignature = 'BF31570369009FA00617B7055FD697C8'
        '''Bitfile Path:'''
        bitfilePath = 'D:/Workspace/Eclipse/Tilda/TildaTarget/bin/TimeResolvedSequencer/NiFpga_TRS.lvbitx'
        '''FPGA Resource:'''
        fpgaResource = 'Rio1'
        '''Indicators:'''
        DACQuWriteTimeout = {'ref': 0x8116, 'val': ctypes.c_bool(), 'ctr': False}
        MCSQuWriteTimeout = {'ref': 0x811A, 'val': ctypes.c_bool(), 'ctr': False}
        MCSerrorcount = {'ref': 0x8112, 'val': ctypes.c_byte(), 'ctr': False}
        MCSstate = {'ref': 0x8122, 'val': ctypes.c_uint(), 'ctr': False}
        measVoltState = {'ref': 0x811E, 'val': ctypes.c_uint(), 'ctr': False}
        seqState = {'ref': 0x8146, 'val': ctypes.c_uint(), 'ctr': False}
        '''Controls:'''
        VoltOrScaler = {'ref': 0x810E, 'val': ctypes.c_bool(), 'ctr': True}
        abort = {'ref': 0x813E, 'val': ctypes.c_bool(), 'ctr': True}
        halt = {'ref': 0x813A, 'val': ctypes.c_bool(), 'ctr': True}
        hostConfirmsHzOffsetIsSet = {'ref': 0x8136, 'val': ctypes.c_bool(), 'ctr': True}
        invertScan = {'ref': 0x8162, 'val': ctypes.c_bool(), 'ctr': True}
        timedOutWhileHandshake = {'ref': 0x814A, 'val': ctypes.c_bool(), 'ctr': True}
        MCSSelectTrigger = {'ref': 0x8132, 'val': ctypes.c_ubyte(), 'ctr': True}
        heinzingerControl = {'ref': 0x815E, 'val': ctypes.c_ubyte(), 'ctr': True}
        cmdByHost = {'ref': 0x8142, 'val': ctypes.c_uint(), 'ctr': True}
        waitAfterReset25nsTicks = {'ref': 0x8156, 'val': ctypes.c_uint(), 'ctr': True}
        waitForKepco25nsTicks = {'ref': 0x815A, 'val': ctypes.c_uint(), 'ctr': True}
        measVoltPulseLength25ns = {'ref': 0x8150, 'val': ctypes.c_long(), 'ctr': True}
        measVoltTimeout10ns = {'ref': 0x814C, 'val': ctypes.c_long(), 'ctr': True}
        nOfBunches = {'ref': 0x8124, 'val': ctypes.c_long(), 'ctr': True}
        nOfScans = {'ref': 0x8164, 'val': ctypes.c_long(), 'ctr': True}
        nOfSteps = {'ref': 0x8168, 'val': ctypes.c_long(), 'ctr': True}
        start = {'ref': 0x816C, 'val': ctypes.c_long(), 'ctr': True}
        stepSize = {'ref': 0x8170, 'val': ctypes.c_long(), 'ctr': True}
        delayticks = {'ref': 0x812C, 'val': ctypes.c_ulong(), 'ctr': True}
        nOfBins = {'ref': 0x8128, 'val': ctypes.c_ulong(), 'ctr': True}
        '''TargetToHostFifos:'''
        transferToHost = {'ref': 0, 'val': ctypes.c_ulong(), 'ctr': False}
        
        
        """
        Hand filled Values, for example certain values for enums:
        """
        seqState.update({'init': 0, 'idle': 1, 'measureOffset': 2, 'measureTrack': 3, 'measComplete': 4, 'error': 5})
        transferToHost.update({'nOfReqEle': 100000})


        dummyScanParameters = {'MCSSelectTrigger': 0, 'delayticks': 100, 'nOfBins': 10000, 'nOfBunches': 1,
                               'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100,
                               'VoltOrScaler': False, 'stepSize': int('00000010000000000000', 2),
                               'start': int('00000000001000000000', 2), 'nOfSteps': 20,
                               'nOfScans': 5, 'invertScan': False, 'heinzingerControl': 1, 'waitForKepco25nsTicks': 40,
                               'waitAfterReset25nsTicks': 4000}
