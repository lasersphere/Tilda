# -*- coding: utf-8 -*-
"""
ROCImporter

Created on 05.08.2021

@author: Patrick Mueller

Run functions of the Measurement/ROCImporter during the beamtime.
"""

from Measurement.ROCImporter import set_path, open_dialog, plot_energies, plot_energy_ratios, \
    plot_calibrated_energies, plot_data, export_xml, add_asymmetry_to_cs_file, calc_cal_i


def plot_e(path_to_file=None):
    bins = 200  # The number of bins in the energy histogram.
    e_min = -20000  # The minimum "energy" of the outer PMTs to consider for the fits or the combined plot.
    e_max = 2000  # The maximum "energy" of the outer PMTs to consider for the fits or the combined plot.
    e_i_min = -1000000  # The minimum "energy" of the inner PMT to consider for the fits or the combined plot.
    e_i_max = 1000000  # The maximum "energy" of the inner PMT to consider for the fits or the combined plot.
    outer_pmt = True  # Whether to show the outer or the inner PMT in the multi plot.
    plot_multi = True  # Whether to plot all selected calibration files combined
    # or to fit the inner and outer PMT of the selected.
    colors = ['c', 'r', 'g', 'b']  # Colors to cycle through in the multi plot.

    plot_energies(path_to_file, bins=bins, outer_pmt=outer_pmt, e_min=e_min, e_max=e_max, energy_i=[e_i_min, e_i_max],
                  plot_multi=plot_multi, colors=colors[::-1])


def plot_e_ratio(path_to_file=None):
    bins = 100  # The number of bins in the energy histogram.
    r_min = 0  # The minimum ratio to consider for the fits or the plot.
    r_max = 15  # The maximum ratio to consider for the fits or the plot.

    plot_energy_ratios(path_to_file, bins=bins, r_min=r_min, r_max=r_max)


def plot_e_calib(path_to_file=None):
    bins = 80  # The number of bins in the energy histogram.
    e_min = 0  # The minimum "energy" to consider for the fits or the combined plot.
    e_max = 50  # The maximum "energy" to consider for the fits or the combined plot.
    only_inner = False  # Only use the energy of the inner detector.
    subtract = False  # Subtract the second spectrum from the first.
    flip_order = False  # Flip the order of the subtraction.
    r_oi = 1.15  # 1.2  # 1.5
    cal_o = -0.0085
    cal_i = calc_cal_i(cal_o, r_oi)
    print('cal_o: ', cal_o)
    print('cal_i: ', cal_i)

    plot_calibrated_energies(path_to_file, bins=bins, e_min=e_min, e_max=e_max,
                             only_inner=only_inner, r_oi=r_oi, cal_o=cal_o, subtract=subtract, flip_order=flip_order)


def plot(path_to_file=None):
    energy = [0, 99999]  # The energy gates.
    time = [0, 99999]  # The time gates.

    plot_3d = False  # Plot a ROC spectrum with asymmetry (plot_3d=False)
    # or the energy-time distribution of the asymmetry signal (plot_3d=True)
    n_energy_bins = 30  # The number of energy bins.
    e_min = None  # The minimum energy bin for 3d plots.
    e_max = None  # The maximum energy bin for 3d plots.

    show = True  # Show the plot.

    volt, y_atom, y_ion, y_asym = plot_data(path_to_file=path_to_file, energy=energy, time=time, plot3d=plot_3d,
                                            n_energy_bins=n_energy_bins, e_min=e_min, e_max=e_max, show=show)
    return volt, y_atom, y_ion, y_asym


def export(path_to_file=None):
    energy = [0, 99999]  # The energy gates.
    time = [0, 99999]  # The time gates.
    cs = True  # Export the spectrum into a cs or a trs file. In the latter case, the asymmetry is not exported.
    export_xml(path_to_file=path_to_file, energy=energy, time=time, cs=cs)


def plot_and_export():
    path_to_file = open_dialog('ROC (*_atom.dat)')  # Do not change this line.
    volt, y_atom, y_ion, y_asym = plot(path_to_file)
    export(path_to_file=path_to_file)
    return volt, y_atom, y_ion, y_asym


def add_asymmetry():
    add_asymmetry_to_cs_file(path_to_file=None)


""" Uncomment the functions to use them. """
set_path('C:\\Users\\collaps\\Documents\\ROCFiles')
# Set the path to the measurements folder to open the file dialog there.

# plot_e()  # Plot the energy distribution of a ROC data file.

# plot_e_ratio()  # Plot the ratio of the energies of the outer and inner PMT.

plot_e_calib()  # Plot the calibrated energies.

# plot()  # Plot a ROC spectrum with asymmetry (plot_3d=False)
# or the energy-time distribution of the asymmetry signal (plot_3d=True)

# export()  # Export the ROC spectrum to a Tilda xml-file.

# plot_and_export()  # Do both functions above with only one file dialog.

# add_asymmetry()  # Add the asymmetry as another PMT to a cs xml-file. Required for the ROC-Faraday-cup mode.
