"""

Created on '20.10.2015'

@author:'simkaufm'

"""

import Service.Scan.draftScanParameters as DftSc
import Application.Config as Cfg
from copy import copy


def init_empty_scan_dict(type_str=None):
    """
    returns an empty scan dictionary in the form as defined in Service.Scan.draftScanParameters.
    All values will be set to None, except the Version information.
    Only the version information is already filled in from Application.Config.
    """
    scand = dict.fromkeys(DftSc.scanDict_list)
    for key, val in scand.items():
        scand[key] = dict.fromkeys(getattr(DftSc, key + '_list'))
    scand['isotopeData']['version'] = Cfg.version
    scand['activeTrackPar'] = merge_dicts(scand['activeTrackPar'],
                                          init_seq_specific_dict(type_str))
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
    """
    n_of_tracks = 0
    for key, val in scan_dict.items():
        if 'track' in str(key):
            n_of_tracks += 1
    return n_of_tracks