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
        self.post_acc_main = PostAcc.PostAccelerationMain()
        self.abort_scan = False
        self.halt_scan = False

    def init_post_accel_pwr_supplies(self):
        """
        restarts and connects to the power devices
        """
        return self.post_acc_main.power_supply_init()

    def prepare_scan(self, scan_dict, callback_sig=None):
        """
        function to prepare for the scan of one isotope.
        This sets up the pipeline and loads the bitfile on the fpga of the given type.
        """
        logging.info('preparing isotope: ' + scan_dict['isotopeData']['isotope'] +
                     ' of type: ' + scan_dict['isotopeData']['type'])
        self.pipeline = Tpipe.find_pipe_by_seq_type(scan_dict, callback_sig)
        self.pipeline.start()
        self.prep_seq(scan_dict['isotopeData']['type'])  # should be the same sequencer for the whole isotope
        #
        #
        # that's it
        # for track_index, track_num in enumerate(track_num_list):
        #     self.prep_track_in_pipe(track_num, track_index)
        #     if self.start_measurement(scan_dict, track_num):
        #         self.read_data()  # this call blocks.
        # logging.info('Measurement completed of isotope: ' + scan_dict['isotopeData']['isotope'] +
        #              'of type: ' + scan_dict['isotopeData']['type'])
        # # this must be moved to scan complete state or something.
        # self.pipeline.stop()  # halt the pipeline, which will cause the mpl-win to stay open
        # self.pipeline.clear()

    def prep_seq(self, seq_type):
        """
        prepare the sequencer before scanning -> load the correct bitfile to the fpga.
        """
        if self.sequencer is None:  # no sequencer loaded yet, must load
            logging.debug('loading sequencer of type: ' + seq_type)
            self.sequencer = FindSeq.ret_seq_instance_of_type(seq_type)
        else:
            if seq_type == 'kepco':
                if self.sequencer.type not in DftScan.sequencer_types_list:
                    logging.debug('loading cs in order to perform kepco scan')
                    self.sequencer = FindSeq.ret_seq_instance_of_type('cs')
            elif self.sequencer.type != seq_type:  # check if current sequencer type is already the right one
                logging.debug('loading sequencer of type: ' + seq_type)
                self.sequencer = FindSeq.ret_seq_instance_of_type('cs')

    def prep_track_in_pipe(self, track_num, track_index):
        """
        prepare the pipeline for the next track
        """
        track_name = 'track' + str(track_num)
        self.pipeline.pipeData['pipeInternals']['activeTrackNumber'] = (track_index, track_name)

    def start_measurement(self, scan_dict, track_num):
        """
        will start the measurement for one track.
        After starting the measurement, the FPGA runs on its own.
        """
        track_dict = scan_dict.get('track' + str(track_num))
        logging.debug('starting measurement with track_dict: ' +
                      str(sorted(track_dict)))
        # figure out how to restart the pipeline with the new parameters here
        start_ok = self.sequencer.measureTrack(scan_dict, track_num)
        return start_ok

    def read_data(self):
        """
        read the data coming from the fpga.
        The data will be directly fed to the pipeline.
        """
        result = {}
        meas_state = self.sequencer.config.seqStateDict['measureTrack']
        seq_state = self.sequencer.getSeqState()
        while seq_state == meas_state or result.get('nOfEle') > 0:
            seq_state = self.sequencer.getSeqState()
            result = self.sequencer.getData()
            if result.get('nOfEle') == 0:
                if seq_state == meas_state and timed_out_count < max_time_out:
                    time.sleep(sleep_time)
                    timed_out_count += 1
                else:
                    break
            else:
                self.pipeline.feed(result['newData'])
                time.sleep(sleep_time)
        # self.pipeline.clear()

    def set_post_accel_pwr_supply(self, power_supply, volt):
        """
        function to set the desired Heinzinger to the Voltage that is needed.
        """
        readback = self.post_acc_main.set_voltage(power_supply, volt)
        return readback

    def set_post_accel_pwr_spply_output(self, power_supply, outp_bool):
        """
        will set the output according to outp_bool
        """
        self.post_acc_main.set_output(power_supply, outp_bool)

    def get_status_of_pwr_supply(self, power_supply, read_from_dev=True):
        """
        returns a list of dicts containing the status of the power supply,
        keys are: name, programmedVoltage, voltageSetTime, readBackVolt, output
        power_supply == 'all' will return status of all active power supplies
        if read_from_dev is False: only return the last stored status.
        """
        if read_from_dev:
            ret = self.post_acc_main.status_of_power_supply(power_supply)
        else:
            ret = self.post_acc_main.power_sup_status
        return ret

    def measureOffset(self, scanpars):
        """
        Measure the Offset Voltage using a digital Multimeter. Hopefully the NI-4071
        will be implemented in the future.
        :param scanpars: dictionary, containing all scanparameters
        :return: bool, True if success
        """
        return True
