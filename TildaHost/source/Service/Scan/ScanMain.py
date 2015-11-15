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
from Driver.Heinzinger.Heinzinger import Heinzinger
import Driver.Heinzinger.HeinzingerCfg as hzCfg


class ScanMain:
    def __init__(self):
        self.sequencer = None
        self.pipeline = None
        self.scan_state = 'initialized'

        self.heinz1 = Heinzinger(hzCfg.comportHeinzinger1)
        self.heinz2 = Heinzinger(hzCfg.comportHeinzinger2)
        # self.heinz3 = Heinzinger(hzCfg.comportHeinzinger2)

    def start_measurement(self, scan_dict):
        self.prep_seq(scan_dict['isotopeData']['type'])
        self.pipeline = Tpipe.find_pipe_by_seq_type(scan_dict)
        self.scan_state = 'measuring'
        track_list = SdOp.get_number_of_tracks_in_scan_dict(scan_dict)[1]
        for track_num in track_list:
            # measure all tracks in order of their tracknumber.
            scan_dict['activeTrackPar'] = scan_dict['track' + track_num]
            self.set_heinzinger(scan_dict['activeTrackPar'])
            # figure out how to restart the pipeline with the new parameters here
            self.measure_one_track(scan_dict)

    def prep_seq(self, seq_type):
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

    def measure_one_track(self, scan_dict):
        self.sequencer.measureTrack(scan_dict)  # this will also set the post acceleration control

    def set_heinzinger(self, track_dict):
        """
        function to set the desired Heinzinger to the Voltage that is needed.
        :param track_dict: dictionary, containing all scanparameters
        :return: bool, True if success, False if fail within maxTries.
        """
        self.scan_state = 'setting Heinzinger'
        active_heinzinger = getattr(self, 'heinz' + str(track_dict['postAccOffsetVoltControl']))
        setVolt = track_dict['postAccOffsetVolt']
        if setVolt != active_heinzinger.getProgrammedVolt():
            # desired Voltage not yet applied
            active_heinzinger.setVoltage(setVolt)
        # compare Voltage with desired Voltage.
        tries = 0
        maxTries = 10
        readback = active_heinzinger.getVoltage()
        while not setVolt * 0.95 < readback < setVolt * 1.05:
            time.sleep(0.1)
            tries += 1
            readback = active_heinzinger.getVoltage()
            if tries > maxTries:
                logging.warning('Heinzinger readback is not within 10% of desired voltage,\n Readback is: ' +
                                str(readback))
                return readback
        logging.info('Heinzinger' + str(track_dict['postAccOffsetVoltControl']) +
                     'readback is: ' + str(readback) + ' V\n' +
                     'last set at: ' + active_heinzinger.time_of_last_volt_set)
        return readback


    def measureOffset(self, scanpars):
        """
        Measure the Offset Voltage using a digital Multimeter. Hopefully the NI-4071
        will be implemented in the future.
        :param scanpars: dictionary, containing all scanparameters
        :return: bool, True if success
        """
        return True
