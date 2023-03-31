
import lxml.etree as et
import os
from copy import deepcopy


def split_file_into_tracks(path, file):
    filepath = os.path.join(path, file)
    newpath = os.path.join(path, 'splitted')
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    origtree = et.parse(filepath)
    origroot = origtree.getroot()
    i = 0
    for t in origroot.find('tracks'):
        newtree = deepcopy(origtree)
        newroot = newtree.getroot()
        newroot.find('header').find('nOfTracks').text = '1'
        tracks = newroot.find('tracks')
        tracks.clear()
        track0 = et.SubElement(tracks, 'track0')

        for ele in t:
            track0.append(ele)

        newfilepath = os.path.join(newpath, os.path.splitext(file)[0] + '_track' + str(i) + '.xml')
        newtree.write(newfilepath)
        i += 1


def split_file_into_tracks_mod2(path, file):
    filepath = os.path.join(path, file)
    newpath = os.path.join(path, 'splitted')
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    origtree = et.parse(filepath)
    origroot = origtree.getroot()
    i = 0
    track_list = []
    for t in origroot.find('tracks'):
        track_list += [t]

    while i < 4:
        newtree = deepcopy(origtree)
        newroot = newtree.getroot()
        newroot.find('header').find('nOfTracks').text = '2'
        tracks = newroot.find('tracks')
        tracks.clear()
        # track0 = et.SubElement(tracks, 'track0')

        tracks.append(track_list[0])
        del track_list[0]
        tracks.append(track_list[0])
        del track_list[0]

        newfilepath = os.path.join(newpath, os.path.splitext(file)[0] + '_track' + str(i) + '.xml')
        newtree.write(newfilepath)
        i += 1


def split_dir_into_tracks(path):
    all_files = os.listdir(path)
    for f in all_files:
        split_file_into_tracks(path, f)


def split_dir_into_tracks_mod2(path):
    all_files = os.listdir(path)
    for f in all_files:
        split_file_into_tracks_mod2(path, f)


if __name__ == '__main__':
    split_dir_into_tracks(r'.\data')
