"""

Created on '13.11.2015'

@author:'simkaufm'


Module for returning an instance of the desired Sequencer
"""

import os
import logging
from PyQt5 import QtWidgets

import Tilda.Application.Config as Cfg


def ret_seq_instance_of_type(seq_type):
    logging.info('searching for sequencer of type: %s' % seq_type)
    if seq_type == 'cs' or seq_type == 'kepco':
        try:
            from Tilda.Driver.DataAcquisitionFpga.ContinousSequencer import ContinousSequencer as Cs
            return Cs()
        except Exception as e:
            reply = QtWidgets.QMessageBox.warning(
                QtWidgets.QWidget(),
                'Hardware not found!',
                'error: could not find hardware for continuous sequencer (cs),'
                'maybe the fpga config file in \n\n'
                f'{os.path.join(Cfg.config_dir, "Driver", "DataAcquisitionFpga", "fpga_config.xml")} \n\n'
                'is not configured correctly.\n'
                'Check if your config file is set for '
                'the right fpga type: \n\n'
                '(PXI-7841R or PXI-7852R currently)\n\n'
                'Also check if the resource is ok, e.g. Rio1 or so, MAX helps to identify this\n\n'
                'Do you want to start the dummy continuous sequencer?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                return ret_seq_instance_of_type('csdummy')
            else:
                return None
    elif seq_type == 'trs':
        try:
            from Tilda.Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer as Trs
            return Trs()
        except Exception as e:
            logging.error('error while loading trs: %s' % e, exc_info=True)
            reply = QtWidgets.QMessageBox.warning(
                QtWidgets.QWidget(),
                'Hardware not found!',
                'error: could not find hardware for time resolved sequencer (trs),'
                'maybe the fpga config file in \n\n'
                f'{os.path.join(Cfg.config_dir, "Driver", "DataAcquisitionFpga", "fpga_config.xml")} \n\n'
                'is not configured correctly.\n'
                'Check if your config file is set for '
                'the right fpga type: \n\n'
                '(PXI-7841R or PXI-7852R currently)\n\n'
                'Also check if the resource is ok, e.g. Rio1 or so, MAX helps to identify this\n\n'
                'Do you want to start the dummy time resolved sequencer?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                logging.info('will load dummy on user request')
                return ret_seq_instance_of_type('trsdummy')
            else:
                logging.info('i will not load any sequencer and abort mission now')
                return None
    elif seq_type == 'csdummy':
        from Tilda.Driver.DataAcquisitionFpga.ContinousSequencerDummy import ContinousSequencer as CsDummy
        return CsDummy()
    elif seq_type == 'trsdummy':
        from Tilda.Driver.DataAcquisitionFpga.TimeResolvedSequencerDummy import TimeResolvedSequencer as TrsDummy
        return TrsDummy()
    else:
        return None
