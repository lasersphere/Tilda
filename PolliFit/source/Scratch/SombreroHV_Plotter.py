
import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def plot_files(path, file):
    filepath = os.path.join(path, file)
    tree = ET.parse(filepath)
    root = tree.getroot()
    sombrero_hv = root.find('tracks')[0].find('header').find('triton').find('duringScan').find('SombreroHV').find('voltage').find('data').text
    sombrero_hv = eval('np.array(' + sombrero_hv + ', dtype=float)')

    # plt.title(file)
    plt.plot(np.arange(0, len(sombrero_hv), 1), 10000*sombrero_hv - 15010.5)
    # plt.show()


def fix_dir(path):
    all_files = os.listdir(path)

    plt.title('Sombrero_HV')
    plt.xlabel('number')
    plt.ylabel('Dist. from setpoint in V')
    for f in all_files:
        plot_files(path, f)
    plt.show()


fix_dir("F:\\IKPcloud\\Projekte\\KOALA\\Calcium\\Auswertung_Laborbuch\\D2\\12\\data\\splitted")
