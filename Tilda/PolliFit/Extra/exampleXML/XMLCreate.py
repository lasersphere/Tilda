"""
Created on 30.07.2014

@author: hammen
"""

import lxml.etree as et
from datetime import datetime as dt
import numpy as np


def create(file):
    """
    :param file: The test file to create.
    :return: None.
    """
    root = et.Element('TrigaLaserData')

    header = et.SubElement(root, 'header')

    version = et.SubElement(header, 'version')
    version.text = '0.1'

    typ = et.SubElement(header, 'type')
    typ.text = 'StandardScan'

    datetime = et.SubElement(header, 'datetime')
    datetime.text = str(dt.now())

    iso = et.SubElement(header, 'isotope')
    iso.text = 'Idiotium_1002'

    nr_tracks = et.SubElement(header, 'nrTracks')
    nr_tracks.text = '1'

    col = et.SubElement(header, 'colDirTrue')
    col.text = 'True'

    acc_volt = et.SubElement(header, 'accVolt')
    acc_volt.text = '9999.8'

    laser_freq = et.SubElement(header, 'laserFreq')
    laser_freq.text = '1000'

    tracks = et.SubElement(root, 'tracks')

    track0 = et.SubElement(tracks, 'track0')

    left_line0 = et.SubElement(track0, 'leftLine')
    left_line0.text = '-200'

    data0 = et.SubElement(track0, 'data')
    data0.text = repr(np.linspace(0, 50, 100).tolist())

    tree = et.ElementTree(root)
    tree.write(file, pretty_print=True)


if __name__ == '__main__':
    create(r'.\TestXML.xml')
