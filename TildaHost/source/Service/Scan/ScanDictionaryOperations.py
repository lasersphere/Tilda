"""

Created on '20.10.2015'

@author:'simkaufm'

"""

import Service.Scan.draftScanParameters as DftSc


def init_empty_scan_dict(version):
    scand = dict.fromkeys(DftSc.scanDict_list)
    for key, val in scand.items():
        scand[key] = dict.fromkeys(getattr(DftSc, key + '_list'))
    scand['isotopeData']['version'] = version
    return scand
