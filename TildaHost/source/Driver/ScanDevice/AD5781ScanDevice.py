"""
Created on 04.03.19

@author: simkaufm

Module Description:  for now, dummy class functionality will be left at the fpga
"""


import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

import Service.VoltageConversions.VoltageConversions as VCon

try:
    import Service.VoltageConversions.DAC_Calibration as AD5781Fit
except:
    import Service.VoltageConversions.DacRegisterToVoltageFit as AD5781Fit


from Measurement.SpecData import SpecDataXAxisUnits as Units
from Driver.ScanDevice.BaseTildaScanDeviceControl import BaseTildaScanDeviceControl


class AD5781ScanDev(BaseTildaScanDeviceControl):
    """
    overwrite all functions if wanted else dummy
    """

    def __init__(self):
        super(AD5781ScanDev, self).__init__()

    def return_scan_dev_info(self, dev_type=None, dev_name=None):
        """
        return the scan device info
        -> currently only one dac available.... adapt maybe if needed
        """
        draft_scan_dev_dict = {
            'name': AD5781Fit.dac_name,
            'type': 'AD5781_V2',  # what type of device, e.g. AD5781(DAC) / Matisse (laser)
            'devClass': 'DAC',  # carrier class of the dev, e.g. DAC / Triton
            'stepUnitName': Units.line_volts.name,
            'start': 0.0,
            'stop': 0.0,
            'stepSize': 1.0,
            'preScanSetPoint': None,  # 0 volts
            'postScanSetPoint': None,
            'timeout_s': 10.0,  # timeout in seconds after which step setting is accounted as failure due to timeout,
            # set top 0 for never timing out.
            'setValLimit': (VCon.get_voltage_from_18bit(0), VCon.get_voltage_from_18bit((2 ** 18) - 1)),
            'sepSizeLimit': (VCon.get_stepsize_in_volt_from_18bit(1), 15.0)
        }
        return draft_scan_dev_dict

    def available_scan_dev_names_by_type(self, type):
        """
        return a list of available AD5781DACs, for the given type of this devClass
        :param type: str, type of the device (e.g. AD5781 / Matisse)
        :return: list of strings with available names
        """
        if type == 'AD5781_V2':
            return [AD5781Fit.dac_name]
        else:
            return []

    def available_scan_dev_types(self):
        """
        return a list of available scan device types for this devClass
        :return: list of strings with available types
        """
        return ['AD5781_V2']
