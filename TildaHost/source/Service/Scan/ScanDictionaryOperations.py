"""

Created on '20.10.2015'

@author:'simkaufm'

"""

import Application.Config as Cfg
import Service.Scan.draftScanParameters as DftSc
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs
from Service.VoltageConversions import VoltageConversions as VCon


def init_empty_scan_dict(type_str=None, version=None, load_default_vals=False):
    """
    returns an empty scan dictionary in the form as defined in Service.Scan.draftScanParameters.
    All values will be set to None, except the Version information.
    Only the version information is already filled in from Application.Config.
    """
    scand = dict.fromkeys(DftSc.scanDict_list)
    for key, val in scand.items():
        scand[key] = dict.fromkeys(getattr(DftSc, key + '_list'))
    scand['track0'] = merge_dicts(scand['track0'], init_seq_specific_dict(type_str))
    if load_default_vals:
        for key, val in scand['track0'].items():
            scand['track0'][key] = DftSc.draftTrackPars.get(key)
        for key, val in scand['isotopeData'].items():
            scand['isotopeData'][key] = DftSc.draftIsotopePars.get(key)
        for key, val in scand['measureVoltPars'].items():
            scand['measureVoltPars'][key] = DftSc.draftMeasureVoltPars.get(key)
    scand['isotopeData']['version'] = Cfg.version
    scand['track0']['trigger'] = {'type': TiTs.no_trigger}
    scand['track0']['nOfCompletedSteps'] = 0
    return scand


def init_seq_specific_dict(type_str):
    """ by a given sequencer type, return a sequencer specific dictionary
     containing all required values for this sequencer. such as 'dwellTime10ns' etc. """
    if type_str in DftSc.sequencer_types_list:
        seq_dict = dict.fromkeys(getattr(DftSc, type_str + '_list'))
    else:
        seq_dict = {}
    return seq_dict


def sequencer_dict_from_track_dict(track_dict, type_str):
    """ return a dictionary which contains all the values inside a track dictionary
     specific for this sequencer type such as 'dwellTime10ns' etc. """
    new_dict = {key: track_dict.get(key) for key in init_seq_specific_dict(type_str)}
    return new_dict


def merge_dicts(d1, d2):
    """ given two dicts, merge them into a new dict as a shallow copy """
    new = d1.copy()
    new.update(d2)
    return new


def get_number_of_tracks_in_scan_dict(scan_dict):
    """
    count the number of tracks in the given dictionary.
    search indicator is 'track' in keys.
    :return: (n_of_tracks, sorted(list_of_track_nums))
    """
    n_of_tracks = 0
    list_of_track_nums = []
    for key, val in scan_dict.items():
        if 'track' in str(key):
            n_of_tracks += 1
            list_of_track_nums.append(int(key[5:]))
    return n_of_tracks, sorted(list_of_track_nums)


def get_available_tracknum(scan_dict):
    """
    will return a tracknumber for the next available track.
    :return: tuple, new_track_num, list_of_track_nums
    """
    n_of_tracks, list_of_track_nums = get_number_of_tracks_in_scan_dict(scan_dict)
    for new_track_num in range(n_of_tracks + 1):
        if new_track_num not in list_of_track_nums:
            return new_track_num, list_of_track_nums


def get_num_of_steps_in_scan(scan_dict):
    """
    go through all tracks and get the number of steps.
    :return: list, each element is a tuple (scans, steps, scans * steps) in this track
    """
    result = []
    all_steps = 0
    n_of_tracks, list_of_track_nums = get_number_of_tracks_in_scan_dict(scan_dict)
    for t in list_of_track_nums:
        steps = scan_dict['track' + str(t)]['nOfSteps']
        scans = scan_dict['track' + str(t)]['nOfScans']
        total = (scans, steps, steps * scans)
        all_steps += total[2]
        result.append(total)
    return result, all_steps


def add_missing_voltages(scan_dict):
    """
    this will calculate 'dacStartVoltage', 'dacStepsizeVoltage', 'dacStopVoltage' and 'dacStartRegister18Bit'
    for each track and will add this to the given scan_dict
    :param scan_dict: dict, containing all informations for a scan.
    :return: dict, the updated scan_dict
    """
    for key, sub_dict in scan_dict.items():
        if 'track' in key:
            dac_stop_18bit = VCon.calc_dac_stop_18bit(sub_dict['dacStartRegister18Bit'],
                                                      sub_dict['dacStepSize18Bit'],
                                                      sub_dict['nOfSteps'])
            sub_dict.update(dacStartVoltage=VCon.get_voltage_from_18bit(sub_dict['dacStartRegister18Bit']))
            sub_dict.update(dacStepsizeVoltage=VCon.get_stepsize_in_volt_from_18bit(sub_dict['dacStepSize18Bit']))
            sub_dict.update(dacStopVoltage=VCon.get_voltage_from_18bit(dac_stop_18bit))
            sub_dict.update(dacStopRegister18Bit=dac_stop_18bit)
    return scan_dict