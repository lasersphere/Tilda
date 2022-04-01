import xml.etree.ElementTree as ET
import os
import ast

def freq_clean(path, file):
    filepath = filepath = os.path.join(path, file)
    origtree = ET.parse(filepath)
    root = origtree.getroot()

    freqComb = root.find('tracks').find('track0').find('header').find('triton').find('duringScan')

    if freqComb:
        freqTree = freqComb[0].find('laser_freq_mult').find('data')
        freqList = ast.literal_eval(freqTree.text)

        ref = freqList[0]
        delta = 10
        newFreqList = []
        for f in freqList:
            if f < ref+delta and f > ref-delta:
                newFreqList.append(f)

        nOfChanges = len(freqList) - len(newFreqList)

        freqTree.text = str(newFreqList)

        origtree.write(filepath)
        print('File ' + filepath + ' done. Number of Changes: ' + str(nOfChanges))
    else:
        print('No Freq Measurement in ' + filepath)

def fix_dir(path):
    all_files = os.listdir(path)

    for f in all_files:
        freq_clean(path, f)



fix_dir('C:\\Users\\pimgram\\IKP ownCloud\\Projekte\\KOALA\\C4+\\Online-Auswertung\\2022-03-14\\55\\data\\splitted')