'''
Created on 30.07.2014

@author: hammen
'''
import lxml.etree as ET
from datetime import datetime as dt
import numpy as np



if __name__ == '__main__':
    root = ET.Element('TrigaLaserData')
    
    header = ET.SubElement(root, 'header')
    
    version = ET.SubElement(header, 'version')
    version.text = '0.1'
    
    typ = ET.SubElement(header, 'type')
    typ.text = 'StandardScan'
    
    datetime = ET.SubElement(header, 'datetime')
    datetime.text = str(dt.now())
    
    iso = ET.SubElement(header, 'isotope')
    iso.text = 'Idiotium_1002'
    
    nrTracks = ET.SubElement(header, 'nrTracks')
    nrTracks.text = '1'
    
    direc = ET.SubElement(header, 'colDirTrue')
    direc.text = 'True'
    
    accVolt = ET.SubElement(header, 'accVolt')
    accVolt.text = '9999.8'
    
    laserFreq = ET.SubElement(header, 'laserFreq')
    laserFreq.text = '1000'
    
    
    tracks = ET.SubElement(root, 'tracks')
    
    track0 = ET.SubElement(tracks, 'track0')
    
    leftLine0 = ET.SubElement(track0, 'leftLine')
    leftLine0.text = '-200'
    
    data0 = ET.SubElement(track0, 'data')
    data0.text = repr(np.linspace(0, 50, 100).tolist())
    
    
    tree = ET.ElementTree(root)
    tree.write('TestXML.tld', pretty_print = True)