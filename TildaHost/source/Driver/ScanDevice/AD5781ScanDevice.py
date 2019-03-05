"""
Created on 04.03.19

@author: simkaufm

Module Description:  for now, dummy class functionality will be left at the fpga
"""


import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


from Measurement.SpecData import SpecDataXAxisUnits as Units
from Driver.ScanDevice.BaseTildaScanDeviceControl import BaseTildaScanDeviceControl


class AD5781ScanDev(BaseTildaScanDeviceControl):
    """
    overwrite all functions if wanted else dummy
    """

    def __init__(self):
        super(AD5781ScanDev, self).__init__()

    def return_scan_dev_info(self):
        """ return the scan device info """
        draft_scan_dev_dict = {
            'name': 'AD5781_Dummy',
            'type': 'AD5781',  # what type of device, e.g. AD5781(DAC) / Matisse (laser)
            'devClass': 'DAC',  # carrier class of the dev, e.g. DAC / Triton
            'stepUnitName': Units.line_volts.name,
            'start': 0.0,
            'stop': 0.0,
            'stepSize': 1.0,
            'preScanSetPoint': None,  # 0 volts
            'postScanSetPoint': None
        }
        return draft_scan_dev_dict
