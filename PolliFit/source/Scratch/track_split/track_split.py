import xml.etree.ElementTree as ET
import os
from copy import deepcopy
from TildaTools import select_from_db
import numpy


def split_file_into_tracks(path, file):
    filepath = os.path.join(path, file)
    newpath = os.path.join(path, 'splitted')
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    origtree = ET.parse(filepath)
    origroot = origtree.getroot()
    i = 0
    for t in origroot.find('tracks'):
        newtree = deepcopy(origtree)
        newroot = newtree.getroot()
        newroot.find('header').find('nOfTracks').text = '1'
        tracks = newroot.find('tracks')
        tracks.clear()
        track0 = ET.SubElement(tracks, 'track0')

        for ele in t:
            track0.append(ele)

        newfilepath = os.path.join(newpath, os.path.splitext(file)[0]+'_track'+str(i)+'.xml')
        newtree.write(newfilepath)
        i += 1


def fix_dir(path):
    all_files = os.listdir(path)

    for f in all_files:
        split_file_into_tracks(path, f)


fix_dir("C:\\Users\\Laura Renth\\Desktop\\Daten\\masterthesis\\ISLA\\Messungen\\Calcium\\Daten\\col\\data205-214 44Ca mit Blende")
# fix_dir('C:\\Users\\Patrick\\ownCloud\\Projekte\\KOALA\\Calcium\\Auswertung_Laborbuch\\D1\\43\\data')