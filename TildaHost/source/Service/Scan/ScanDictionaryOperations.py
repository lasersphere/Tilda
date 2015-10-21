"""

Created on '20.10.2015'

@author:'simkaufm'

"""

import Service.Scan.draftScanParameters as DftSc
import Application.Config as Cfg



def init_empty_scan_dict():
    """
    returns an empty scan dictionary in the form as defined in Service.Scan.draftScanParameters.
    Only the version information is already filled in from Application.Config.
    """
    scand = dict.fromkeys(DftSc.scanDict_list)
    for key, val in scand.items():
        scand[key] = dict.fromkeys(getattr(DftSc, key + '_list'))
    scand['isotopeData']['version'] = Cfg.version
    return scand
