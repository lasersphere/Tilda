import xml.etree.ElementTree as ET
import os
from copy import deepcopy
#from TildaTools import select_from_db
import numpy


#track-Splitter f�r 43Ca

def split_file_into_tracks(path, file):
    filepath = os.path.join(path, file)
    newpath = os.path.join(path, 'splitted')
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    origtree = ET.parse(filepath)
    origroot = origtree.getroot()
    i = 0
    trackList=[]
    for t in origroot.find('tracks'):
        trackList+=[t]

    while i<4:
        newtree = deepcopy(origtree)
        newroot = newtree.getroot()
        newroot.find('header').find('nOfTracks').text = '2'
        tracks = newroot.find('tracks')
        tracks.clear()
        #track0 = ET.SubElement(tracks, 'track0')

        tracks.append(trackList[0])
        del trackList[0]
        tracks.append(trackList[0])
        del trackList[0]

        newfilepath = os.path.join(newpath, os.path.splitext(file)[0]+'_track'+str(i)+'.xml')
        newtree.write(newfilepath)
        i += 1

def fix_dir(path):
    all_files = os.listdir(path)

    for f in all_files:
        split_file_into_tracks(path, f)


fix_dir("C:\\Users\\pimgram\\IKP ownCloud\\Projekte\\KOALA\\Calcium\\Auswertung_Laborbuch\\D2\\23\\data")