"""

Created on '19.05.2015'

@author:'simkaufm'

"""

import time
import logging

import Driver.DataAcquisitionFpga.FindSequencerByType as FindSeq
import Service.Scan.ScanDictionaryOperations as SdOp
import Service.Scan.draftScanParameters as DftScan
import Service.AnalysisAndDataHandling.tildaPipeline as Tpipe
import Driver.Heinzinger.HeinzingerCfg as hzCfg
import Driver.PostAcceleration.PostAccelerationMain as PostAcc


class ScanMain:
    def __init__(self):
        self.sequencer = None
        self.pipeline = None
        self.scan_state = 'initialized'
        self.post_accel_pwr_supplies = None
        self.abort_scan = False
        self.halt_scan = False

        # power supplies can be initialized when starting up.
        self.connect_post_accel_pwr_supplies()

    def connect_post_accel_pwr_supplies(self):
        """
        restarts and connects to the power devices
        """
        self.post_accel_pwr_supplies = PostAcc.PostAccelerationMain()

    def scan_one_isotope(self, scan_dict):
        """
        function to handle the scanning of one isotope, must be interruptable by halt/abort
        """
        logging.info('preparing isotope: ' + scan_dict['isotopeData']['isotope'] +
                     'of type: ' + scan_dict['isotopeData']['type'])
        n_of_tracks, track_list = SdOp.get_number_of_tracks_in_scan_dict(scan_dict)
        self.pipeline = Tpipe.find_pipe_by_seq_type(scan_dict)
        self.prep_seq(scan_dict['isotopeData']['type'])  # should be the same sequencer for the whole isotope
        for tr in track_list:
            if self.start_measurement(scan_dict, tr):
                self.read_data()

    def prep_seq(self, seq_type):
        """
        prepare the sequencer before scanning -> load the correct bitfile to the fpga, etc..
        """
        self.scan_state = 'starting up sequencer of type: ' + seq_type
        if self.sequencer is None:
            logging.debug('loading sequencer of type: ' + seq_type)
            self.sequencer = FindSeq.ret_seq_instance_of_type(seq_type)
        else:
            if seq_type == 'kepco':
                logging.debug('loading sequencer of type: ' + seq_type)
                if self.sequencer.type not in DftScan.sequencer_types_list:
                    logging.debug('loading cs in order to perform kepco scan')
                    self.sequencer = FindSeq.ret_seq_instance_of_type('cs')
            elif self.sequencer.type != seq_type:
                self.sequencer = FindSeq.ret_seq_instance_of_type('cs')

    def start_measurement(self, scan_dict, track_name):
        """
        will start the measurement for one track.
        After starting the measurement, the FPGA runs on its own.
        """
        self.scan_state = 'measuring'
        track_dict = scan_dict.get(track_name)
        if track_dict.get('postAccOffsetVoltControl', False):
            # will not be set for Kepco
            power_supply = 'Heinzinger' + str(track_dict.get('postAccOffsetVoltControl'))
            volt = track_dict.get('postAccOffsetVoltControl', 0)
            self.set_post_accel_pwr_supply(power_supply, volt)
        # figure out how to restart the pipeline with the new parameters here
        start_ok = self.sequencer.measureTrack(scan_dict)
        return start_ok

    def read_data(self):
        meas_state = self.sequencer.config.seqStateDict['measureTrack']
        self.sequencer.getData(self.sequencer.config)


    def set_post_accel_pwr_supply(self, power_supply, volt):
        """
        function to set the desired Heinzinger to the Voltage that is needed.
        """
        self.scan_state = 'set offset volt'
        readback = self.post_accel_pwr_supplies.set_voltage(power_supply, volt)
        return readback

    def get_status_of_pwr_supply(self, power_supply):
        """
        returns a dict containing the status of the power supply,
        keys are: name, programmedVoltage, voltageSetTime, readBackVolt
        """
        return self.post_accel_pwr_supplies.status_of_power_supply(power_supply)

    def measureOffset(self, scanpars):
        """
        Measure the Offset Voltage using a digital Multimeter. Hopefully the NI-4071
        will be implemented in the future.
        :param scanpars: dictionary, containing all scanparameters
        :return: bool, True if success
        """
        return True
