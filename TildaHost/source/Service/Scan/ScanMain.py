"""

Created on '19.05.2015'

@author:'simkaufm'

"""

import logging
from datetime import datetime
import Driver.DataAcquisitionFpga.FindSequencerByType as FindSeq
import Service.Scan.ScanDictionaryOperations as SdOp
import Service.Scan.draftScanParameters as DftScan
import Service.AnalysisAndDataHandling.tildaPipeline as Tpipe
import Driver.PostAcceleration.PostAccelerationMain as PostAcc


class ScanMain:
    def __init__(self):
        self.sequencer = None
        self.pipeline = None
        self.post_acc_main = PostAcc.PostAccelerationMain()

    def close_scan_main(self):
        """
        will deinitialize all active power supplies,
        set 0V on the DAC and turn off all fpga outputs
        """
        self.deinit_post_accel_pwr_supplies()
        self.deinit_fpga()

    def init_post_accel_pwr_supplies(self):
        """
        restarts and connects to the power devices
        """
        return self.post_acc_main.power_supply_init()

    def deinit_post_accel_pwr_supplies(self):
        """
        deinitialize all active power supplies
        """
        self.post_acc_main.power_supply_deinit()

    def prepare_scan(self, scan_dict, callback_sig=None):
        """
        function to prepare for the scan of one isotope.
        This sets up the pipeline and loads the bitfile on the fpga of the given type.
        """
        self.pipeline = None
        logging.info('preparing isotope: ' + scan_dict['isotopeData']['isotope'] +
                     ' of type: ' + scan_dict['isotopeData']['type'])
        self.pipeline = Tpipe.find_pipe_by_seq_type(scan_dict, callback_sig)
        self.prep_seq(scan_dict['isotopeData']['type'])  # should be the same sequencer for the whole isotope

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
                    self.deinit_fpga()
                    self.sequencer = FindSeq.ret_seq_instance_of_type('cs')
            elif self.sequencer.type != seq_type:  # check if current sequencer type is already the right one
                logging.debug('loading sequencer of type: ' + seq_type)
                self.deinit_fpga()
                self.sequencer = FindSeq.ret_seq_instance_of_type(seq_type)

    def deinit_fpga(self):
        """
        deinitilaizes the fpga
        """
        if self.sequencer is not None:
            self.sequencer.DeInitFpga()
            self.sequencer = None

    def prep_track_in_pipe(self, track_num, track_index):
        """
        prepare the pipeline for the next track
        reset 'nOfCompletedSteps' to 0.
        """
        track_name = 'track' + str(track_num)
        self.pipeline.pipeData[track_name]['nOfCompletedSteps'] = 0
        self.pipeline.pipeData['pipeInternals']['activeTrackNumber'] = (track_index, track_name)
        self.pipeline.start()

    def start_measurement(self, scan_dict, track_num):
        """
        will start the measurement for one track.
        After starting the measurement, the FPGA runs on its own.
        """
        track_dict = scan_dict.get('track' + str(track_num))
        logging.debug('starting measurement with track_dict: ' +
                      str(sorted(track_dict)))
        start_ok = self.sequencer.measureTrack(scan_dict, track_num)
        return start_ok

    def read_data(self):
        """
        read the data coming from the fpga.
        The data will be directly fed to the pipeline.
        :return: bool, True if nOfEle > 0 that were read
        """
        result = self.sequencer.getData()
        if result.get('nOfEle', -1) > 0:
            self.pipeline.feed(result['newData'])
            return True
        else:
            return False

    def check_scanning(self):
        """
        check if the sequencer is still in the 'measureTrack' state
        :return: bool, True if still scanning
        """
        meas_state = self.sequencer.config.seqStateDict['measureTrack']
        seq_state = self.sequencer.getSeqState()
        return meas_state == seq_state

    def stop_measurement(self, clear=True):
        """
        stops all modules which are relevant for scanning.
        pipeline etc.
        """
        print('stopping measurement, clear is: ', clear)
        self.pipeline.stop()
        if clear:
            self.pipeline.clear()

    def halt_scan(self, b_val):
        """
        halts the scan after the currently running track is completed
        """
        self.sequencer.halt(b_val)

    def abort_scan(self):
        """
        aborts the scan directly
        """
        self.sequencer.abort()

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

    def calc_scan_progress(self, progress_dict, scan_dict, start_time):
        """
        calculates the scan progress by comparing the given dictionaries.
        progress_dict must contain: {activeIso: str, activeTrackNum: int, completedTracks: list, nOfCompletedSteps: int}
        """
        try:
            return_dict = dict.fromkeys(['activeIso', 'overallProgr', 'timeleft', 'activeTrack', 'totalTracks',
                                         'trackProgr', 'activeScan', 'totalScans', 'activeStep',
                                         'totalSteps', 'trackName'])
            iso_name = progress_dict['activeIso']
            track_num = progress_dict['activeTrackNum']
            track_name = 'track' + str(track_num)
            compl_tracks = progress_dict['completedTracks']
            compl_steps = progress_dict['nOfCompletedSteps']
            n_of_tracks, list_of_track_nums = SdOp.get_number_of_tracks_in_scan_dict(scan_dict)
            track_ind = list_of_track_nums.index(track_num)
            total_steps_list, total_steps = SdOp.get_num_of_steps_in_scan(scan_dict)
            steps_in_compl_tracks = sum(total_steps_list[ind][2] for ind, track_n in enumerate(compl_tracks))
            return_dict['activeIso'] = iso_name
            return_dict['overallProgr'] = float(steps_in_compl_tracks + compl_steps) / total_steps * 100
            # timeleft droht gefahr durch 0 zu teilen
            return_dict['timeleft'] = str(self.calc_timeleft(start_time, (steps_in_compl_tracks + compl_steps),
                                                             (total_steps - (steps_in_compl_tracks + compl_steps)))
                                          ).split('.')[0]
            return_dict['activeTrack'] = track_ind + 1
            return_dict['totalTracks'] = len(list_of_track_nums)
            return_dict['trackProgr'] = float(compl_steps) / total_steps_list[track_ind][2] * 100
            return_dict['activeScan'] = int(compl_steps / total_steps_list[track_ind][1]) + \
                                        (compl_steps % total_steps_list[track_ind][1] > 0)
            return_dict['totalScans'] = total_steps_list[track_ind][0]
            return_dict['activeStep'] = compl_steps - (return_dict['activeScan'] - 1) * total_steps_list[track_ind][1]
            return_dict['totalSteps'] = total_steps_list[track_ind][1]
            return_dict['trackName'] = track_name
            return return_dict
        except Exception as e:
            print('while calculating the scan progress, this happened: ' + str(e))
            return None

    def calc_timeleft(self, start_time, already_compl_steps, steps_still_to_complete):
        """
        calculate the time that is left until the whole scan is completed.
        Therfore measure the expired time since scan start and compare it with remaining steps.
        :return: int, time that is left
        """
        now_time = datetime.now()
        dt = now_time - start_time
        if steps_still_to_complete and already_compl_steps:
            timeleft = dt / already_compl_steps * steps_still_to_complete
        else:
            timeleft = 0
        return timeleft
