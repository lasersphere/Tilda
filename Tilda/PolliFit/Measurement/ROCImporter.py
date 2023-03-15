# -*- coding: utf-8 -*-
"""
ROCImporter

Created on 05.08.2021

@author: Patrick Mueller

XML converter / data plotter for measurements from the ROC-beamtime.
"""

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.stats as st
from mpl_toolkits.axes_grid1 import make_axes_locatable
import PyQt5.QtWidgets as Qw

font = {'size': 12}
matplotlib.rc('font', **font)
matplotlib.use('Qt5Agg')
plt.style.use('seaborn-white')
# print(plt.style.available)

try:
    from Tilda.PolliFit import TildaTools as Tt, XmlOperations as Xo
except ImportError:
    Xo = None
    Tt = None


PATH = ''


""" Tilda independent """


def gaussian(x, x0, sigma, y0, intensity):
    return np.sqrt(np.pi) * sigma * intensity * st.norm.pdf(x, loc=x0, scale=sigma) + y0


def ratio_model(x, x0, a, b, c, d):
    return a / (x - x0) * np.exp(-b / (x - x0) ** 2 - c * (x - x0) ** 2) + d


def ratio_center(x0, a, b, c, d):
    return


def cal_energies(e_i, e_o, r_oi, cal_o):
    """
    :param e_i: Detected energy of the inner PMT.
    :param e_o: Detected energy of the outer PMT.
    :param r_oi: Energy ratio outer/inner PMT.
    :param cal_o: Calibration factor of the outer PMT.
    :returns: The calibrated energy.
    """
    return cal_o * (r_oi / 3.4 * e_i + e_o)


def calc_cal_i(cal_o, r_oi):
    return cal_o * r_oi / 3.4


def set_path(path):
    global PATH
    PATH = path


def open_dialog(file_filter, single=False):
    _ = Qw.QApplication(sys.argv)
    # app.aboutToQuit.connect(app.deleteLater)
    dial = Qw.QFileDialog()
    if single:
        path_to_file = dial.getOpenFileName(None, 'Open {} file'.format(file_filter),
                                            PATH if PATH != '' else os.path.expanduser('~/Desktop'), file_filter)[0]
    else:
        path_to_file = dial.getOpenFileNames(None, 'Open {} file'.format(file_filter),
                                             PATH if PATH != '' else os.path.expanduser('~/Desktop'), file_filter)[0]
    if not path_to_file:
        quit()
    return path_to_file


def calc_asymmetry(y_atom, y_ion, with_unc=True):
    """
    Calculate the asymmetry.

    :param y_atom: The atom data.
    :param y_ion: The ion data.
    :param with_unc: Whether to also return the uncertainty of the asymmetry.
    :returns: The asymmetry with or without its uncertainty.
    """
    denom = (y_atom + y_ion).astype(float)
    denom[denom != 0] = 1 / denom[denom != 0]
    y_asym = (y_atom - y_ion) * denom
    if with_unc:
        y_asym_unc = np.sqrt((2 * y_ion * np.sqrt(y_atom) * denom ** 2) ** 2
                             + (-2 * y_atom * np.sqrt(y_ion) * denom ** 2) ** 2)
        return y_asym, y_asym_unc
    return y_asym


def load_file(path_to_file=None, atom=None, energy=None, energy_i=None, time=None):
    """
    Load a file stored by the new DAC program.

    :param path_to_file: The path or a list of paths to the ROC .dat files.
    :param atom: Whether to load atom data or ion data.
    :param energy: The energy gates in the format [min, max]. Defaults to [0, infinity].
    :param energy_i:
    :param time: The time gates in the format [min, max]. Defaults to [0, infinity].
    :returns: The name of the Tilda file, the voltages and the gated data.
    """
    data = {}
    voltages = []
    tilda_file = []
    max_step = -1
    if path_to_file is None:
        file_filter = 'ROC (*_{}.dat)'.format('atom' if atom else 'ion')
        path_to_file = open_dialog(file_filter)
    else:
        if isinstance(path_to_file, str):
            path_to_file = [path_to_file]
    for filename in path_to_file:
        # with open(os.path.join(PATH, '{}_{}.dat'.format(filename, 'atom' if atom else 'ion'))) as file:
        e_count = 0
        with open(filename, 'r') as file:
            scan = -1
            step = -1
            for line in file:
                if line[0] == 'T':
                    tilda_file.append(line[10:].replace('\n', '') + '.xml')
                elif line[0] == 'S':
                    step_start = line.find('Step') + 5
                    scan = int(line[5:line.find(',')])
                    next_step = int(line[step_start:(line[step_start:].find(',')+step_start)])
                    if next_step > max_step:
                        max_step = next_step
                        data[next_step] = [np.array([-1, -1, -1])]
                        voltages.append(0)
                    step = next_step
                    voltages[step] += float(line[(line.find('Voltage')+8):])
                else:
                    try:
                        val = np.array(line.split(','), dtype=float)
                        if energy is None:
                            energy = [-np.inf, np.inf]
                        if energy_i is None:
                            energy_i = [-np.inf, np.inf]
                        if time is None:
                            time = [0, np.inf]
                        if energy_i[0] <= val[0] <= energy_i[1] and energy[0] <= val[1] <= energy[1] and time[0] <= val[2] <= time[1]:
                            data[step].append(np.array(line.split(','), dtype=float))
                    except IndexError:
                        e_count += 1
                        continue
        print('# IndexError {}'.format(e_count))
    for step, d in data.items():
        data[step] = np.array(d, dtype=float)
    return tilda_file, np.array(voltages) / (scan + 1), data


def plot_energies(path_to_file=None, bins=100, p0=None, outer_pmt=True,
                  e_min=-np.inf, e_max=np.inf, energy_i=None, plot_multi=False, colors=None):
    if path_to_file is None:
        path_to_file = open_dialog('ROC (*.dat)')
    else:
        if isinstance(path_to_file, str):
            path_to_file = [path_to_file]
    if p0 is None:
        p0 = [-2000, 2000, 0, 400]
    if plot_multi:
        if colors is None:
            colors = ['b', 'g', 'r', 'c']
        for i, file in enumerate(path_to_file[::-1]):
            _, _, data = load_file(file, energy=[e_min, e_max], energy_i=energy_i)
            data = np.concatenate(tuple(d[1:] for d in data.values()), axis=0)
            plt.hist(data[:, int(outer_pmt)], bins=bins, alpha=0.5, label=file[-file[::-1].find('/'):-4],
                     color=colors[i % len(colors)])
            # plt.show()
            # energy = cal_energies(data[:, 0], data[:, 1], 1.2, -0.0085)
            # plt.hist(energy, bins=bins, alpha=0.5, label=file[file.find('26NaAtom'):-4],
            #          color=colors[i % len(colors)])

        plt.title('Muon calibration')
        plt.xlabel('Signal area (arb. units)')
        plt.ylabel('Abundance (counts)')
        plt.legend(loc=2)
        plt.show()
    else:
        t_file, volt, data = load_file(path_to_file, atom='_atom.dat' in path_to_file, energy=None, time=None)
        data = np.concatenate(tuple(d[1:] for d in data.values()), axis=0) * 2
        y_i, x_i, patches_i = plt.hist(data[:, 0], bins=bins, color='b')
        x_i = x_i[:-1] + (x_i[1] - x_i[0]) / 2
        x_if = x_i[x_i >= e_min]
        y_if = y_i[x_i >= e_min]
        y_if = y_if[x_if <= e_max]
        x_if = x_if[x_if <= e_max]
        try:
            popt_i, pcov_i = so.curve_fit(gaussian, x_if, y_if, p0=p0)
            plt.plot(x_i, gaussian(x_i, *popt_i), 'r')
            print('Center inner: {} +- {}'.format(popt_i[0], np.sqrt(pcov_i[0, 0])))
        except RuntimeError:
            pass
        plt.show()

        y_o, x_o, patches_o = plt.hist(data[:, 1], bins=bins, color='b')
        x_o = x_o[:-1] + (x_o[1] - x_o[0]) / 2
        x_of = x_o[x_o >= e_min]
        y_of = y_o[x_o >= e_min]
        y_of = y_of[x_of <= e_max]
        x_of = x_of[x_of <= e_max]
        try:
            popt_o, pcov_o = so.curve_fit(gaussian, x_of, y_of, p0=p0)
            plt.plot(x_o, gaussian(x_o, *popt_o), 'g')
            print('Center outer: {} +- {}'.format(popt_o[0], np.sqrt(pcov_o[0, 0])))
        except RuntimeError:
            pass
        plt.show()


def plot_energy_ratios(path_to_file=None, bins=100, p0=None, r_min=-np.inf, r_max=np.inf):
    if path_to_file is None:
        path_to_file = open_dialog('ROC (*.dat)', single=False)
    else:
        if isinstance(path_to_file, str):
            path_to_file = [path_to_file]
    if p0 is None:
        p0 = [0, 400, 0.5, 0.5, 0]
    _, _, data = load_file(path_to_file, atom='_atom.dat' in path_to_file, energy=None, time=None)
    data = np.concatenate(tuple(d[1:] for d in data.values()), axis=0)
    nonzero = np.nonzero(data[:, 0])[0]
    x = data[nonzero, 1] / data[nonzero, 0]
    x = x[x >= r_min]
    x = x[x <= r_max]
    y_i, x_i, patches_i = plt.hist(x, bins=bins, label=path_to_file[0][path_to_file[0].find('PMT'):-4])
    x_i = x_i[:-1] + (x_i[1] - x_i[0]) / 2
    popt, pcov = so.curve_fit(ratio_model, x_i, y_i, p0)
    x_cont = np.linspace(np.min(x), np.max(x), 10001)
    y_cont = ratio_model(x_cont, *popt)

    plt.title('Atom signal area ratio')
    plt.xlabel('Signal area ratio (outer / inner)')
    plt.ylabel('Abundance (counts)')
    plt.plot(x_cont, ratio_model(x_cont, *popt), 'r-')
    print('popt: {}'.format(popt))
    print('Ratio outer/inner: {}'.format(x_cont[np.argmax(y_cont)]))
    plt.show()


def plot_calibrated_energies(path_to_file, bins=100, e_min=-np.inf, e_max=np.inf,
                             only_inner=False, r_oi=1, cal_o=1, subtract=False, flip_order=False):
    if path_to_file is None:
        path_to_file = open_dialog('ROC (*.dat)', single=False)
    else:
        if isinstance(path_to_file, str):
            path_to_file = [path_to_file]
    if subtract:
        _, _, data = load_file(path_to_file[0], atom='_atom.dat' in path_to_file[0], energy=None, time=None)
        data = np.concatenate(tuple(d[1:] for d in data.values()), axis=0)
        energy = cal_energies(data[:, 0], 0 if only_inner else data[:, 1], r_oi, cal_o)
        energy = energy[energy >= e_min]
        energy = energy[energy <= e_max]
        y_i, x_i, patches_i = plt.hist(energy, bins=bins, label=path_to_file[0][-path_to_file[0][::-1].find('/'):-4])
        _, _, data = load_file(path_to_file[1], atom='_atom.dat' in path_to_file[1], energy=None, time=None)
        data = np.concatenate(tuple(d[1:] for d in data.values()), axis=0)
        energy = cal_energies(data[:, 0], 0 if only_inner else data[:, 1], r_oi, cal_o)
        energy = energy[energy >= e_min]
        energy = energy[energy <= e_max]
        y_j, x_j, patches_j = plt.hist(energy, bins=x_i, label=path_to_file[1][-path_to_file[1][::-1].find('/'):-4])
        x = x_i[:-1] + (x_i[1] - x_i[0]) / 2
        pm = -1 if flip_order else 1
        plt.bar(x, pm * (y_i - y_j), width=x_i[1] - x_i[0], color='r', label='Abs. difference')
        plt.xlim(e_min, e_max)
        plt.title('Beta decay spectrum')
        plt.xlabel('Calibrated energy (MeV)')
        plt.ylabel('Abundance (counts)')
        plt.legend()
        plt.show()
    else:
        _, _, data = load_file(path_to_file, atom='_atom.dat' in path_to_file, energy=None, time=None)
        data = np.concatenate(tuple(d[1:] for d in data.values()), axis=0)
        energy = cal_energies(data[:, 0], 0 if only_inner else data[:, 1], r_oi, cal_o)
        energy = energy[energy >= e_min]
        energy = energy[energy <= e_max]
        plt.title('Beta decay spectrum')
        plt.xlabel('Calibrated energy (MeV)')
        plt.ylabel('Abundance (counts)')
        y_i, x_i, patches_i = plt.hist(energy, bins=bins, label=path_to_file[0][-path_to_file[0][::-1].find('/'):-4])
        plt.legend()  # file[-file[::-1].find('/'):-4]
        plt.show()


def plot_data(path_to_file=None, energy=None, time=None, plot3d=False,
              n_energy_bins=30, e_min=None, e_max=None, show=True):
    """
    Load and plot a pair of atom and ion data with the specified energy and time gates.

    :param path_to_file: The path or a list of paths to the ROC .dat files.
    :param energy: The energy gates in the format [min, max]. Defaults to [0, infinity].
    :param time: The time gates in the format [min, max]. Defaults to [0, infinity].
    :param plot3d: Whether to create a 3d plot with the entire data, ignoring the gates.
     The returned data is still gated.
    :param n_energy_bins: The number of bins used to digitize the energies.
    :param e_min: The center of the lowest energy bin.
    :param e_max: The center of the highest energy bin.
    :param show: Whether to show the plot.
    :returns: The voltages and tuples of the atom, ion and asymmetry (data, uncertainty).
    """
    if path_to_file is None:
        path_to_file = open_dialog('ROC (*_atom.dat)')
    else:
        if isinstance(path_to_file, str):
            path_to_file = [path_to_file]
    path_to_file_ion = [path.replace('_atom.dat', '_ion.dat') for path in path_to_file]
    t_file_atom, volt, data_atom = load_file(path_to_file=path_to_file, atom=True, energy=energy, time=time)
    t_file_ion, volt, data_ion = load_file(path_to_file_ion, atom=False, energy=energy, time=time)
    y_atom = np.array([data_atom[step].shape[0] - 1 for step in range(len(data_atom))])
    y_ion = np.array([data_ion[step].shape[0] - 1 for step in range(len(data_ion))])
    y_asym, y_asym_unc = calc_asymmetry(y_atom, y_ion, with_unc=True)

    if plot3d:
        _, _, complete_atom = load_file(path_to_file=path_to_file, atom=True, energy=None, time=None)
        _, _, complete_ion = load_file(path_to_file=path_to_file_ion, atom=False, energy=None, time=None)

        t_max = int(np.max([np.max(d[:, 2]) for d in complete_atom.values()]
                           + [np.max(d[:, 2]) for d in complete_ion.values()]))
        _time = np.arange(t_max + 1, dtype=int)
        _time_bins = np.linspace(-0.5, t_max + 0.5, t_max + 2)
        
        if e_min is None:
            e_min = np.min([np.min(d[:, 1]) for d in complete_atom.values()]
                           + [np.min(d[:, 1]) for d in complete_ion.values()])
        if e_max is None:
            e_max = np.max([np.max(d[:, 1]) for d in complete_atom.values()]
                           + [np.max(d[:, 1]) for d in complete_ion.values()])
        e_width = (e_max - e_min) / (n_energy_bins - 1) if n_energy_bins > 1 else 0
        _energy_bins = np.linspace(e_min - e_width / 2, e_max + e_width / 2, n_energy_bins + 1)
        # _energy = np.linspace(e_min, e_max, n_energy_bins)

        y_a = np.array([[[np.sum(np.logical_and(complete_atom[step][:, 2] == _t,
                                                np.logical_and(_e_l <= complete_atom[step][:, 1],
                                                               complete_atom[step][:, 1] < _e_u)))
                          for step in range(len(complete_atom))]
                         for _e_l, _e_u in zip(_energy_bins[:-1], _energy_bins[1:])] for _t in _time])
        y_i = np.array([[[np.sum(np.logical_and(complete_ion[step][:, 2] == _t,
                                                np.logical_and(_e_l <= complete_ion[step][:, 1],
                                                               complete_ion[step][:, 1] < _e_u)))
                          for step in range(len(complete_ion))]
                         for _e_l, _e_u in zip(_energy_bins[:-1], _energy_bins[1:])] for _t in _time])
        y = calc_asymmetry(y_a, y_i, with_unc=False)
        y = np.max(y, axis=2) - np.min(y, axis=2)

        # _time, _energy = np.meshgrid(_time, _energy, indexing='ij')
        _time_bins, _energy_bins = np.meshgrid(_time_bins, _energy_bins, indexing='ij')

        # plt.figure(figsize=[5, 5])
        ax = plt.gca()
        # ax.set_aspect('equal')
        plt.title(t_file_atom[-1])
        plt.xlabel('Time (bin #)')
        plt.ylabel('Energy (?)')
        plt.xlim(-1, t_max + 1)
        cmesh = plt.pcolormesh(_time_bins, _energy_bins, y, cmap='viridis', shading='flat')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.1)
        cbar = plt.colorbar(cmesh, cax=cax)
        cbar.set_label(label='Asymmetry signal height (arb. units)')
    else:
        fig, ax1 = plt.subplots(**{'figsize': [8, 6], 'dpi': 96})

        color_atom = 'k'
        color_ion = 'b'
        color_asym = 'r'

        ax1.errorbar(volt, y_atom, yerr=np.sqrt(y_atom),  fmt='{}.'.format(color_atom), label='Atoms')
        ax1.errorbar(volt, y_ion, yerr=np.sqrt(y_ion), fmt='{}.'.format(color_ion), label='Ions')
        ax1.set_xlabel('Kepco-Voltage (V)')
        ax1.set_ylabel('Intensity (counts)')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.legend(loc=2, bbox_to_anchor=(0, 1.15))

        ax2 = ax1.twinx()
        ax2.errorbar(volt, y_asym, yerr=y_asym_unc, fmt='{}.'.format(color_asym), label='Asymmetry')
        ax2.set_ylabel('Asymmetry (arb. units)', color=color_asym)
        ax2.tick_params(axis='y', labelcolor=color_asym)
        ax2.legend(loc=1, bbox_to_anchor=(1, 1.09))

        plt.title(t_file_atom[-1])
        fig.tight_layout()
    if show:
        plt.show()
    return volt, (y_atom, np.sqrt(y_atom)), (y_ion, np.sqrt(y_ion)), (y_asym, y_asym_unc)


""" For Tilda/PolliFit """


def convert_data(data, scaler=0, t_max=300, cs=True):
    """
    Converts the imported ROC data into the Tilda format.

    :param data: The ROC data imported with the 'load_file' function.
    :param scaler: The scaler to be used with the data.
    :param t_max: The maximum time bin.
    :param cs: Whether to convert the data into the cs or the trs file format.
    :returns: The data in the Tilda xml format.
    """
    if cs:
        return [int(d.shape[0] - 1) for step, d in data.items()]
    time_bins = np.arange(t_max + 1, dtype=int)
    return [(scaler, step, t, np.sum(d[:, 2] == t)) for step, d in data.items()
            for t in time_bins if np.sum(d[:, 2] == t) > 0]


def convert_asymmetry(data_atom, data_ion, t_max=300, cs=True):
    """
    Calculates the asymmetry from the imported ROC data and converts it into the Tilda format.

    :param data_atom: The atom data imported with the 'load_file' function.
    :param data_ion: The ion data imported with the 'load_file' function.
    :param t_max: The maximum time bin.
    :param cs: Whether to convert the data into the cs or the trs file format.
    :returns: The asymmetry in the Tilda xml format.
    """
    if cs:
        return [(data_atom[step_a].shape[0] - data_ion[step_i].shape[0])
                / (data_atom[step_a].shape[0] + data_ion[step_i].shape[0] - 2)
                for step_a, step_i in zip(range(len(data_atom)), range(len(data_ion)))]
    time_bins = np.arange(t_max + 1, dtype=int)
    return [(2, step_a, t, (np.sum(d_a[:, 2] == t) - np.sum(d_i[:, 2] == t))
             / (np.sum(d_a[:, 2] == t) + np.sum(d_i[:, 2] == t))) for (step_a, d_a), (step_i, d_i)
            in zip(data_atom.items(), data_ion.items()) for t in time_bins]


def create_scan_dict(path_to_file, t_max=300, voltages=None, tilda_file=None, cs=True):
    """
    Create the 'scan_dict' for the header of the Tilda xml file.

    :param path_to_file: The path to the xml file.
    :param t_max: The maximum time bin.
    :param voltages: The voltages extracted from the ROC files (Needed if the Tilda file is not available).
    :param tilda_file: The name of the Tilda file.
    :param cs: Whether to convert the data into the cs or the trs file format.
    :returns: The 'scan_dict' for the header of the Tilda xml file.
    """
    if tilda_file is not None:
        try:
            print(path_to_file)
            scan_dict, xml_etree = Tt.scan_dict_from_xml_file(path_to_file)
            scan_dict['track0']['activePmtList'] = [0, 1, 2]
            if not cs:
                scan_dict['track0']['activePmtList'] = [0, 1]
                scan_dict['track0']['nOfBins'] = t_max
                scan_dict['track0']['nOfBunches'] = 10000
                scan_dict['track0']['softBinWidth_ns'] = 10
            return scan_dict
        except FileNotFoundError:
            create_scan_dict(path_to_file, t_max=t_max, voltages=voltages, cs=cs)
    elif voltages is not None:
        ret = dict(activePmtList=[0, 1],
                   colDirTrue=True
                   )
        return ret


def export_xml(path_to_file=None, energy=None, time=None, cs=True):
    """
    Export the data as a Tilda xml file either in the cs or the trs format.
    In the latter case the asymmetry cannot be viewed in Tilda.

    :param path_to_file: The path or a list of paths to the ROC .dat files.
    :param energy: The energy gates in the format [min, max]. Defaults to [0, infinity].
    :param time: The time gates in the format [min, max]. Defaults to [0, infinity].
    :param cs: Whether to convert the data into the cs or the trs file format.
    :returns: None.
    """
    root = Xo.xmlCreateIsotope({'accVolt': 30000.0, 'isotope': '54_Ca', 'laserFreq': 0.0,
                                'nOfTracks': 1, 'type': 'cs' if cs else 'trs', 'version': '1.23'})

    if path_to_file is None:
        path_to_file = open_dialog('ROC (*_atom.dat)')
    else:
        if isinstance(path_to_file, str):
            path_to_file = [path_to_file]
    path_to_file_ion = [path.replace('_atom.dat', '_ion.dat') for path in path_to_file]
    t_file_atom, volt, data_atom = load_file(path_to_file=path_to_file, atom=True, energy=energy, time=time)
    t_file_ion, volt, data_ion = load_file(path_to_file=path_to_file_ion, atom=False, energy=energy, time=time)
    t_max = 300
    if not cs:
        t_max = int(np.max([np.max(d[:, 2]) for d in data_atom.values()]
                           + [np.max(d[:, 2]) for d in data_ion.values()]))
    data_asym = convert_asymmetry(data_atom, data_ion, t_max=t_max, cs=cs)
    data_atom = convert_data(data_atom, scaler=0, t_max=t_max, cs=cs)
    data_ion = convert_data(data_ion, scaler=1, t_max=t_max, cs=cs)
    path_to_file_xml = os.path.join(os.path.dirname(path_to_file[-1]), t_file_atom[-1])
    scan_dict = create_scan_dict(path_to_file_xml, t_max=t_max, tilda_file=t_file_atom[-1])
    if cs:
        data = [data_atom, data_ion, process_asymmetry(data_asym)[0].tolist()]
    else:
        data = data_atom + data_ion
    root = Xo.xmlAddCompleteTrack(root, scan_dict, data, 'track0')
    Tt.save_xml(root, path_to_file_xml[:-4] + '_EDITED.xml')


def numpy_array_from_string(string, shape, datatype=np.uint32):
    """
    converts a text array saved in an lxml.etree.Element
    using the function xmlWriteToTrack back into a numpy array
    :param string: str, array
    :param shape: int, or tuple of int, the shape of the output array
    :param datatype: The datatype of the string.
    :return: numpy array containing the desired values
    """
    string = string.replace('\\n', '').replace('[', '').replace(']', '').replace('  ', ' ')
    result = np.fromstring(string, dtype=datatype, sep=' ')
    result = result.reshape(shape)
    return result


def process_asymmetry(y, dtype=None):
    """
    Linearly transform the asymmetry to an 'dtype' array
    without introducing significant numerical errors into its shape.
    
    :param y: The array to transform.
    :param dtype: The type of the returned array. Defaults to 'int'.
    :returns: The transformed integer array.
    """
    y = y - np.min(y)
    d_min = np.min([np.abs(d_f - d_i) for d_i, d_f in zip(y[:-1], y[1:])])
    factor = 1
    if d_min != 0:
        factor = 10 / np.min([np.abs(d_f - d_i) for d_i, d_f in zip(y[:-1], y[1:])])
        y *= factor
    if dtype is None:
        dtype = int
    return y.astype(dtype), factor


def test_process_asymmetry(y=None):
    """
    Plot the transformed asymmetry with type float (exact) and type int (Tilda compatible).

    :returns: None.
    """
    if y is None:
        y = (np.random.random(100) - 0.5) * 5
    y = process_asymmetry(y, dtype=float)
    plt.plot(y)
    y = process_asymmetry(y, dtype=int)
    plt.plot(y)
    plt.show()


def add_asymmetry_to_cs_file(path_to_file=None):
    """
    Add the asymmetry (N(Atom) - N(Ion)) / (N(Atom) + N(Ion))
    to a copy of the specified cs-type xml-file as a third PMT.

    :param path_to_file: The path to the xml file (Including the file itself).
    :returns: None.
    """
    if path_to_file is None:
        file_filter = 'Tilda (*.xml)'
        path_to_file = open_dialog(file_filter)
    for path in path_to_file:
        scan_dict, xml_etree = Tt.scan_dict_from_xml_file(path)
        tracks = Xo.xmlFindOrCreateSubElement(xml_etree, 'tracks')
        track0 = Xo.xmlFindOrCreateSubElement(tracks, 'track0')
        data = Xo.xmlFindOrCreateSubElement(track0, 'data')
        shape = (2, scan_dict['track0']['nOfSteps'])
        scan_dict['track0']['activePmtList'] += [scan_dict['track0']['activePmtList'][-1] + 1]
        scalers = numpy_array_from_string(data[1].text, shape)
        scaler2, uncertainty2 = calc_asymmetry(scalers[0], scalers[1], with_unc=True)
        scaler2, factor = process_asymmetry(scaler2)
        uncertainties = np.concatenate((np.sqrt(scalers), np.expand_dims(uncertainty2 * factor, axis=0)), axis=0)
        scalers = np.concatenate((scalers, np.expand_dims(scaler2, axis=0)), axis=0).astype(int)

        xml_etree = Xo.xmlAddCompleteTrack(xml_etree, scan_dict, scalers, 'track0')
        Xo.xmlWriteDict(data, {'errorArray': uncertainties})
        Tt.save_xml(xml_etree, path[:-4] + '_EDITED.xml')


# plot_data(energy=[0, 4], plot3d=False, n_energy_bins=2)
# export_xml(cs=True)
# add_asymmetry_to_cs_file()  # 'D:\\Users\\Patrick\\Dokumente\\Auswertungen\\ROC2021\\40_Ca_cs_run785_track0.xml')
