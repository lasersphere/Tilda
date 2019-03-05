"""
Created on 04.03.2019

@author: simkaufm

Module Description:
(blank) base scan device class which holds all required functions
for a device that should be scanned from within TILDA.
"""

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


from Measurement.SpecData import SpecDataXAxisUnits as Units


class BaseTildaScanDeviceControl(QObject):
    """
    overwrite all functions if wanted else dummy
    """
    # whenever the currently connected scan device is sending new values, this will be emitted.
    # dict, see in self.receive() -> 'devPars'
    scan_dev_sends_new_settings_pyqtsig = pyqtSignal(dict)
    # when the scan device has set the scan parameters it will emit a dictionary with the corresponding values
    # dict, see self.receive() -> 'scanParsSet'
    scan_dev_has_setup_these_pars_pyqtsig = pyqtSignal(dict)
    # when the scan device has set a new step, it will emit a dictionary
    # dict, see self.receive() -> 'scanProgress'
    scan_dev_has_set_a_new_step_pyqtsig = pyqtSignal(dict)
    # signal to emit data to the pipeLine
    # will be overwritten if main exists in self.get_existing_callbacks_from_main()
    data_to_pipe_sig = pyqtSignal(np.ndarray, dict)

    def __init__(self):
        super(BaseTildaScanDeviceControl, self).__init__()

    def return_scan_dev_info(self):
        """ return the scan device info """
        draft_scan_dev_dict = {
            'name': 'base',
            'type': 'base',  # what type of device, e.g. AD5781(DAC) / Matisse (laser)
            'devClass': 'base',  # carrier class of the dev, e.g. DAC / Triton
            'stepUnitName': Units.line_volts.name,
            'start': 0.0,
            'stop': 0.0,
            'stepSize': 1.0,
            'preScanSetPoint': None,  # 0 volts
            'postScanSetPoint': None
        }
        return draft_scan_dev_dict

    def setup_scan_in_scan_dev(self, start, stepsize, num_of_steps, num_of_scans, invert_in_odd_scans):
        """
        set these values in the scan device.
        Once this has completed setting this up, it will emit the scan pars
        :param start: float, starting point in units of self.start_step_units
        :param stepsize: float, step size in units of self.start_step_units
        :param num_of_steps: int, number of steps per scan
        :param num_of_scans: int, number of scans
        :param invert_in_odd_scans: bool, True -> invert after step complete
        :return: None
        """
        pass

    def request_next_step(self):
        """ request the next step from the dev """
        pass

    def abort_scan(self):
        """ abort the curently running scan """
        pass

    def set_pre_scan_masurement_setpoint(self, set_val):
        """
        Set the scan device to the setpoint, so a pre scan measurement can be performed with other devices.
        blocks until value is set.
        :param set_val: float, set point for the device
        :return: bool, True for success
        """
        return True
