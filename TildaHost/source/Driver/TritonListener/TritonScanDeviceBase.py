"""
Created on 01.03.19

@author: simkaufm

Module Description:  The ScanDeviceBase copied from Triton.
Usually scan devices will stay within Triton, but for testing it is handy to have this class also here.

copied from Triton on 04.03.19 git rev. number: 800d42dc83a9ba7afbdf7f98d27c1d6a598554d1

If changes are made within Triton maybe a copy is needed again.
Required modifications for Tilda are marked with a comment  # changed!
look for them before overwriting again!

"""

import numpy as np  # np types are not supported by serialisation for pyro4! be Careful!
import time
import logging  # changed!


from Driver.TritonListener.TritonDeviceBase import DeviceBase  # changed!
from Measurement.SpecData import SpecDataXAxisUnits as Units  # changed!


class ScanDeviceBase(DeviceBase):
    def __init__(self, name, sql_conf):
        self.type = 'TildaScanDevCtl'  # Specialize device type from 'TritonObject'

        self.possible_start_step_unit = Units

        """ overwrite in device: """
        self.start_step_units = None  # chose one from possible_start_step_unit
        # device limitations:
        self.dev_min_step_size = 0.0  # float, minimum stepsize
        # WARNING float('inf') might be causing problems wehn serializing
        self.dev_max_step_size = 1 * 10 ** 30  #  float('inf')  # float, maximum stepsize
        self.dev_min_val = -1 * 10 ** 30  # float('-inf')  # float, minimal value that can be set with this device
        self.dev_max_val = 1 * 10 ** 30  # float('inf')  # float, minimal value that can be set with this device

        """ scan settings """
        self.sc_start = 0  # float, starting point of scan in units of self.start_step_units
        self.sc_stop = 0  # float, end point of scan in units of self.start_step_units
        self.sc_stepsize = 0  # float, stepsize of scan in units of self.start_step_units
        self.sc_num_of_steps = 0  # int, number of steps per scan
        self.sc_num_of_scans = 0  # int, number of scans
        self.sc_invert_in_odd_scans = False  # bool, True -> invert after step complete
        self.sc_one_scan_vals = np.zeros(0, dtype=np.float)  # numpy array which holds all values which will
        #  be set by the device for one scan
        self._progressing_step = False
        self.scan_statii = {
            'initialized': 'initialized',
            'setupForScan': 'setupForScan',
            'scanning': 'scanning',
            'complete': 'complete',
            'aborted': 'aborted'  # scan was aborted
        }  # list all possible scan statii
        self.scan_status = self.scan_statii['initialized']

        """ live scan settings """
        self.sc_l_cur_step = -1  # -1 pre scan, than it holds the current step
        self.sc_l_cur_scan = 0  # holds the current scan
        self.sc_l_perc_compl = 0  # percent of completed steps will be updated from self.check_if_scan_complete
        self.sc_l_scan_complete = False  # True if scan complete will be updated from self.check_if_scan_complete
        self._internal_scan = False  # will be True if an internal scan has been started
        self._abort_scan = False  # True for abort

        """ constructor """
        super(ScanDeviceBase, self).__init__(name, sql_conf)  # here self.on() is called !

        """ overwrite constants in BaseDevice only after constructor! """
        # self.name = ... etc. but better in another device class ontop of this!

    """ emit parameters etc. """
    # when sending, do not use numpy arrays etc. these cannot be serialized with serpent
    def emit_device_parameters(self):
        """
        emit the device specific parameters,
        might be helpful to learn upstream, what the device can do.
        channel: devPars
        val: {
            'name': self.name,
            'type': self.type,
            'step_size_min_max': (self.dev_min_step_size, self.dev_max_step_size),
            'set_val_min_max': (self.dev_min_val, self.dev_max_val),
            'unitName': self.start_step_units.name
        }
        """
        to_send = {
            'name': self.name,
            'type': self.type,
            'step_size_min_max': (self.dev_min_step_size, self.dev_max_step_size),
            'set_val_min_max': (self.dev_min_val, self.dev_max_val),
            'unitName': self.start_step_units.name
        }
        self.send('devPars', to_send)

    def emit_scan_pars(self):
        """
        emit the scan parameters after they have ben set in the scan device
        channel is: scanParsSet
        val is: {
            'unitName': self.start_step_units.name,
            'start': self.sc_start,
            'stop': self.sc_stop,
            'stepSize': self.sc_stepsize,
            'stepNums': self.sc_num_of_steps,
            'valsArrrayOneScan': self.sc_one_scan_vals
        }
        """
        to_send = {
            'unitName': self.start_step_units.name,
            'start': self.sc_start,
            'stop': self.sc_stop,
            'stepSize': self.sc_stepsize,
            'stepNums': self.sc_num_of_steps,
            'valsArrrayOneScan': self.sc_one_scan_vals
        }
        self.send('scanParsSet', to_send)

    def emit_scan_progress(self):
        """
        will only be emitted after a scan is set!

        channel is scanProgress
        val is: {
            'curStep': self.sc_l_cur_step,
            'curScan': self.sc_l_cur_scan,
            'percentOfScan': self.sc_l_perc_compl,
            'curStepVal': self.sc_one_scan_vals[self.sc_l_cur_step],
            'scanStatus': self.scan_status
        }
        """
        to_send = {
            'curStep': self.sc_l_cur_step,
            'curScan': self.sc_l_cur_scan,
            'curStepVal': self.sc_one_scan_vals[self.sc_l_cur_step],
            'percentOfScan': self.sc_l_perc_compl,
            'scanStatus': self.scan_status
        }
        self.send('scanProgress', to_send)

    ''' scan related '''
    def setup_scan(self, start, stepsize, num_of_steps, num_of_scans, invert_in_odd_scans):
        """
        store the scan parameters coming from elsewhere to the corresponding vars
        and create the array with all values for one scan
        :param start: float, starting point in units of self.start_step_units
        :param stepsize: float, step size in units of self.start_step_units
        :param num_of_steps: int, number of steps per scan
        :param num_of_scans: int, number of scans
        :param invert_in_odd_scans: bool, True -> invert after step complete
        :return:
        """
        self.sc_start = max(start, self.dev_min_val)
        self.sc_stepsize = min(max(stepsize, self.dev_min_step_size), self.dev_max_step_size)
        self.sc_stop = min(self.sc_start + (num_of_steps - 1) * stepsize, self.dev_max_val)
        self.sc_one_scan_vals, self.sc_stepsize = np.linspace(self.sc_start, self.sc_stop, num_of_steps, retstep=True)
        self.sc_one_scan_vals = self.sc_one_scan_vals.tolist()  # np types are not supported by serialisation for pyro4
        self.sc_stepsize = float(self.sc_stepsize)  # np types are not supported by serialisation for pyro4
        self.sc_invert_in_odd_scans = invert_in_odd_scans
        self.sc_num_of_steps = num_of_steps
        self.sc_num_of_scans = num_of_scans
        self.sc_l_cur_step = -1
        self.sc_l_cur_scan = 0
        self.sc_l_perc_compl = 0
        self.sc_l_scan_complete = False
        self._abort_scan = False
        self.scan_status = self.scan_statii['setupForScan']
        self.emit_scan_pars()

    def setup_next_step(self):
        """
        Tell the scan device to set the next step.

        This will return True if the command has been accepted and
        the next step will be called on the next PERIODIC CALL of the scan device.
        It will return False, if the device is currently still setting the last step.
        Once the STEP IS SET, the device will call emit_scan_progess and tell all subscribers.

        This quick operation is non blocking,
        therefore upstream callers will be able to proceed with their code quickly
        and the chance of being stuck while calling this function and simulatineously lose the connection is minimized,
        but still might happen eventually! Be careful here!
        :return bool, True: command accepted, False: command ignored, because busy
        """
        if self._progressing_step:
            # currently still setting up a step
            logging.warning('%s was told to set the next step but is still busy, this step will be ignored! ' % self.name)
            return False
        else:
            invert_pls = self.sc_invert_in_odd_scans and self.sc_l_cur_scan % 2 != 0
            scan_dir = -1 if invert_pls else 1
            # -> -1 for odd scan numbers; +1 for even scan numbers -> sc0: +1, sc1: -1, sc2: +1, ....
            # change direction when scan complete...
            if self.sc_l_cur_step + scan_dir > self.sc_num_of_steps - 1:
                # scan completed in last step -> begin new scan
                self.sc_l_cur_scan += 1
                invert_pls = self.sc_invert_in_odd_scans and self.sc_l_cur_scan % 2 != 0
                self.sc_l_cur_step = self.sc_num_of_steps - 1 if invert_pls else 0
            elif invert_pls and self.sc_l_cur_step + scan_dir < 0:
                # scan completed in last step -> begin new scan
                self.sc_l_cur_scan += 1
                self.sc_l_cur_step = 0
            else:
                self.sc_l_cur_step += scan_dir
            # in the end really set this step in the device
            self._progressing_step = True
            # start periodic if that has not been done yet
            self.scan_status = self.scan_statii['scanning']
            if self._interval == 0:  # start the periodic if that has not been done yet
                self.setInterval(0.1)
            return True

    def check_if_scan_complete(self):
        """ check if the scan is complete. Useful for internal scan etc. """
        num_of_compl_steps = (self.sc_l_cur_step + 1) + self.sc_l_cur_scan * self.sc_num_of_steps
        # plus 1, because step starts at 0
        steps_that_need_to_be_done = self.sc_num_of_steps * self.sc_num_of_scans
        self.sc_l_perc_compl = num_of_compl_steps / steps_that_need_to_be_done
        self.sc_l_scan_complete = num_of_compl_steps >= steps_that_need_to_be_done
        if self.sc_l_scan_complete:
            self.scan_status = self.scan_statii['complete']
        return self.sc_l_scan_complete

    def _periodic(self):
        """ _periodic will handle setting the actual steps therefore it runs in the thread of the scandevice """
        logging.debug('%s periodic is runnning,\n %s \n %s \n %s' % (
            self.name, self._progressing_step, self._abort_scan, self.sc_l_scan_complete))
        if self._progressing_step and not self._abort_scan and not self.sc_l_scan_complete:
            self.set_step_in_dev(self.sc_l_cur_step)  # this might take some time
            self.check_if_scan_complete()
            self.emit_scan_progress()
            self._progressing_step = False
        if self._internal_scan and not self._abort_scan and not self.sc_l_scan_complete:
            # setup the next scan if the internal scan is running
            self.setup_next_step()
        self.periodic()

    def abort_scan(self):
        """ abort the current scan. Block until step was progressed. """
        self._abort_scan = True
        while self._progressing_step:
            time.sleep(0.1)
        self.scan_status = self.scan_statii['aborted']
        self._internal_scan = False  # abort internal scan
        self._abort_scan = False
        self.emit_scan_progress()
        return True

    def start_internal_scan(self):
        """
        call this, if the device should just go from start to stop for number of scans without external input.
        It must have been setup before though!
        """
        self._internal_scan = True
        self.setup_next_step()  # will start the scan


    '''  overwrite from dev: '''
    def set_step_in_dev(self, step_num):
        """ overwrite this with your function were you can actually set this value in the device """
        print('dev would set now step %s with val: %s' % (step_num, self.sc_one_scan_vals[step_num]))
        pass

    def set_pre_scan_measurement_setpoint(self, set_val):
        """
        overwrite this with a function from the device that is able to set the device's scanning parameter,
        to the given set point. Then in Tilda other devices can perform a pre / post scan measurement.
        None if it should do nothing.
        Block until setpoint is reached.
        :param set_val: float / None, value that should be set by the device
        :return: bool, True if successful
        """
        if set_val is None:
            return True
        else:
            print('dev would now set to the desired prescan measurement value %s ' % set_val)
            return True  # in this mode



