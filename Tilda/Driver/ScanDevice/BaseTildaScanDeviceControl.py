"""
Created on 04.03.2019

@author: simkaufm

Module Description:
(blank) base scan device class which holds all required functions
for a device that should be scanned from within TILDA.
"""

from PyQt5.QtCore import QObject, pyqtSignal


from Tilda.PolliFit.Measurement.SpecData import SpecDataXAxisUnits as Units


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

    def __init__(self):
        super(BaseTildaScanDeviceControl, self).__init__()
        self.possible_units = Units

    def available_scan_dev_types(self):
        """
        return a list of available scan device types for this devClass
        :return: list of strings with available types
        """
        return ['base']

    def available_scan_dev_names_by_type(self, type):
        """
        return a list of available scan devices, for the given type of this devClass
        :param type: str, type of the device (e.g. AD5781 / Matisse)
        :return: list of strings with available names
        """
        return ['base']

    def return_scan_dev_info(self, dev_type=None, dev_name=None):
        """
        return the scan device info currently subscribed to
        or as requested by dev_type + dev_name
        :return dev_type: str, type of device or None for currently subscribed to
        :return dev_name: str, name of device for which the values are requested, None for currently subscribed to.
        :return dict: {
            'name': 'base',
            'type': 'base',  # what type of device, e.g. AD5781(DAC) / Matisse (laser)
            'devClass': 'base',  # carrier class of the dev, e.g. DAC / Triton
            'stepUnitName': Units.line_volts.name,
            'start': 0.0,
            'stop': 0.0,
            'stepSize': 1.0,
            'preScanSetPoint': None,  # 0 volts
            'postScanSetPoint': None,
            'timeout_s': 10.0,  # timeout in seconds after which step setting is accounted as failure due to timeout,
            # set top 0 for never timing out.
            'setValLimit': (-10.0, 10.0),
            'stepSizeLimit': (7.628880920000002e-05, 15.0)
        }
        """
        draft_scan_dev_dict = {
            'name': 'base',
            'type': 'base',  # what type of device, e.g. AD5781(DAC) / Matisse (laser)
            'devClass': 'base',  # carrier class of the dev, e.g. DAC / Triton
            'stepUnitName': Units.line_volts.name,
            'start': 0.0,
            'stop': 0.0,
            'stepSize': 1.0,
            'preScanSetPoint': None,  # 0 volts
            'postScanSetPoint': None,
            'timeout_s': 10.0,  # timeout in seconds after which step setting is accounted as failure due to timeout,
            # set top 0 for never timing out.
            'setValLimit': (-10.0, 10.0),
            'stepSizeLimit': (7.628880920000002e-05, 15.0)
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

    def deinit_scan_dev(self):
        """
        dio whatever is needed when deinitialising the scan device
        :return: None
        """
        pass

    def get_existing_callbacks_from_main(self):
        """
        get the existing callbacks from the main and overwrite the corresponding pyqsignals if wanted
        :return: None
        """
        pass
