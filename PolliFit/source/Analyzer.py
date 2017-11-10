'''
Created on 21.05.2014

@author: hammen, gorges

The Analyzer can extract() parameters from fit results, combineRes() to get weighted averages and combineShift() to calculate isotope shifts.
'''

import ast
import functools
import os
import sqlite3
import logging
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit

import MPLPlotter as plt
import Physics
import TildaTools as TiTs


def getFiles(iso, run, db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''SELECT file, pars FROM FitRes WHERE iso = ? AND run = ? ORDER BY file ''', (iso, run))
    e = cur.fetchall()
    con.close()

    return [f[0] for f in e]


def extract(iso, par, run, db, fileList=[], prin=True):
    '''Return a list of values of par of iso, filtered by files in fileList'''
    print('Extracting', iso, par, )
    if iso == 'all':
        fits = TiTs.select_from_db(db, 'file, pars', 'FitRes', [['run'], [run]], 'ORDER BY file',
                                   caller_name=__name__)
    else:
        fits = TiTs.select_from_db(db, 'file, pars', 'FitRes', [['iso', 'run'], [iso, run]], 'ORDER BY file',
                                   caller_name=__name__)
    if fits is not None:
        if len(fileList):
            fits = [f for f in fits if f[0] in fileList]
        fitres = [eval(f[1]) for f in fits]
        files = [f[0] for f in fits]
        vals = [f[par][0] for f in fitres]
        errs = [f[par][1] for f in fitres]
        date_list = []
        for f, v in zip(files, vals):
            e = TiTs.select_from_db(db, 'date', 'Files', [['file'], [f]], caller_name=__name__)
            if not len(e):  # check if maybe everything is written non capital
                f = os.path.normcase(f)
                e = TiTs.select_from_db(db, 'date', 'Files', [['file'], [f]], caller_name=__name__)
            date = e[0][0]
            if date is not None:
                date_list.append(date)
            if prin:
                print(date, '\t', f, '\t', v, '\t', e)

        if len(fileList):
            for f in fileList:
                if f not in files:
                    print('Warning:', f, 'not found!')
        return vals, errs, date_list, files
    else:
        return None, None, None, None


def weightedAverage(vals, errs):
    '''Return (weighted average, propagated error, rChi^2'''
    weights = 1 / np.square(errs)
    # print(weights)
    average = sum(vals * weights) / sum(weights)
    errorprop = np.sqrt(1 / sum(weights))  # was: 1 / sum(weights)
    if (len(vals) == 1):
        rChi = 0
    else:
        rChi = 1 / (len(vals) - 1) * sum(np.square(vals - average) * weights)

    return (average, errorprop, rChi)


def average(vals, errs):
    average = sum(vals) / len(vals)
    errorprop = np.sqrt(sum(np.square(errs))) / len(vals)

    if len(vals) == 1:
        rChi = 0
    else:
        # rChi = 0
        chiSum = 0
        for i in range(0, len(vals), 1):
            chiSum += np.square(vals[i] - average) * np.square(errs[i])
        rChi = 1 / (len(vals) - 1) * chiSum
    return (average, errorprop, rChi)


def combineRes(iso, par, run, db, weighted=True, print_extracted=True,
               show_plot=False, only_this_files=[], write_to_db=True):
    '''
    Calculate weighted average of par using the configuration specified in the db
    :rtype : object
    '''
    print('Open DB', db)

    con = sqlite3.connect(db)
    cur = con.cursor()

    cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', (iso, par, run))
    con.commit()
    if iso == 'all':
        r = TiTs.select_from_db(db, 'config, statErrForm, systErrForm', 'Combined',
                                [['parname', 'run'], [par, run]],
                                caller_name=__name__)
    else:
        r = TiTs.select_from_db(db, 'config, statErrForm, systErrForm', 'Combined',
                                [['iso', 'parname', 'run'], [iso, par, run]],
                                caller_name=__name__)
    con.close()
    if r is not None:
        (config, statErrForm, systErrForm) = r[0]
    else:
        return [None] * 6
    config = ast.literal_eval(config)

    if len(only_this_files):
        config = only_this_files

    print('Combining', iso, par)
    vals, errs, date, files = extract(iso, par, run, db, config, prin=print_extracted)
    print(files)
    if weighted:
        avg, err, rChi = weightedAverage(vals, errs)
    else:
        avg, err, rChi = average(vals, errs)
    print('rChi is: ', rChi, 'err is: ', err)

    systE = functools.partial(avgErr, iso, db, avg, par)
    statErr = eval(statErrForm)
    systErr = eval(systErrForm)
    print('statErr is: ', statErr)

    print('Statistical error formula:', statErrForm)
    print('Systematic error formula:', systErrForm)
    print('Combined to', iso, par, '=')
    print(str(avg) + '(' + str(statErr) + ')[' + str(systErr) + ']')
    print('Combined rounded to %s %s = %.3f(%.0f)[%.0f]' % (iso, par, avg, statErr * 1000, systErr * 1000))
    if write_to_db:
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?
            WHERE iso = ? AND parname = ? AND run = ?''', (avg, statErr, systErr, rChi, iso, par, run))
        con.commit()
    con.close()
    plt.clear()
    combined_plots_dir = os.path.join(os.path.split(db)[0], 'combined_plots')
    if not os.path.exists(combined_plots_dir):
        os.mkdir(combined_plots_dir)
    avg_fig_name = os.path.join(combined_plots_dir, iso + '_' + run + '_' + par + '.png')
    plotdata = (date, vals, errs, avg, statErr, systErr, ('k.', 'r'),
                False, avg_fig_name, '%s_%s_%s [MHz]' % (iso, par, run))
    ax = plt.plotAverage(*plotdata)
    print('saving average plot to: ', avg_fig_name)
    plt.save(avg_fig_name)
    if show_plot:
        print('showing plot!')
        plt.show(True)
    else:
        plt.clear()

    print('date \t file \t val \t err')
    for i, dt in enumerate(date):
        print(dt, '\t', files[i], '\t', vals[i], '\t', errs[i])

    return avg, statErr, systErr, rChi, plotdata, ax


def combineShift(iso, run, db, show_plot=False):
    """takes an Isotope a run and a database and gives the isotopeshift to the reference!"""
    print('Open DB', db)

    (config, statErrForm, systErrForm) = TiTs.select_from_db(db, 'config, statErrForm, systErrForm', 'Combined',
                                                             [['iso', 'parname', 'run'], [iso, 'shift', run]],
                                                             caller_name=__name__)[0]
    '''config needs to have this shape:
    [
    (['dataREF1.*','dataREF2.*',...],
    ['dataINTERESTING1.*','dataINT2.*',...],
    ['dataREF4.*',...]),
    ([...],[...],[...]),
    ...
    ]
    '''
    config = ast.literal_eval(config)
    print('Combining', iso, 'shift')

    (ref, refRun) = TiTs.select_from_db(db, 'Lines.reference, lines.refRun',
                                        'Runs JOIN Lines ON Runs.lineVar = Lines.lineVar', [['Runs.run'], [run]],
                                        caller_name=__name__)[0]
    # each block is used to measure the isotope shift once
    shifts = []
    shiftErrors = []
    dateIso = []
    for block in config:
        if block[0]:
            preVals, preErrs, date, files = extract(ref, 'center', refRun, db, block[0])
            preVal, preErr, preRChi = weightedAverage(preVals, preErrs)
            preErr = applyChi(preErr, preRChi)
            preErr = np.absolute(preErr)
        else:
            preVal = 0
            preErr = 0

        intVals, intErrs, date, files = extract(iso, 'center', run, db, block[1])
        [dateIso.append(i) for i in date]

        if block[2]:
            postVals, postErrs, date, files = extract(ref, 'center', refRun, db, block[2])
            postVal, postErr, postRChi = weightedAverage(postVals, postErrs)
            postErr = np.absolute(applyChi(postErr, postRChi))
        else:
            postVal = 0
            postErr = 0
        if preVal == 0:
            refMean = postVal
        elif postVal == 0:
            refMean = preVal
        else:
            refMean = (preVal + postVal) / 2

        if preVal == 0 or postVal == 0 or np.absolute(preVal - postVal) < np.max([preErr, postErr]):
            errMean = np.sqrt(preErr ** 2 + postErr ** 2)
        else:
            errMean = np.absolute(preVal - postVal)
        shifts.extend([x - refMean for x in intVals])
        shiftErrors.extend(np.sqrt(np.square(intErrs) + np.square(errMean)))
    val, err, rChi = weightedAverage(shifts, shiftErrors)
    systE = functools.partial(shiftErr, iso, run, db)

    # systErrForm, e.g. systE(accVolt_d=1.5 * 10 ** -4, offset_d=1.5 * 10 ** -4)
    statErr = eval(statErrForm)
    systErr = eval(systErrForm)
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?
        WHERE iso = ? AND parname = ? AND run = ?''', (val, statErr, systErr, rChi, iso, 'shift', run))
    con.commit()
    con.close()
    print('shifts:', shifts)
    print('shiftErrors:', shiftErrors)
    print('Mean of shifts:', val)
    combined_plots_dir = os.path.join(os.path.split(db)[0], 'combined_plots')
    avg_fig_name = os.path.join(combined_plots_dir, iso + '_' + run + '_shift.png')
    plotdata = (
        dateIso, shifts, shiftErrors, val, statErr, systErr, ('k.', 'r'), False, avg_fig_name, '%s_shift [MHz]' % iso)
    if show_plot:
        plt.plotAverage(*plotdata)
        plt.show(True)
    # plt.clear()
    return (shifts, shiftErrors, val, statErr, systErr, rChi)


def combineShiftByTime(iso, run, db, show_plot=False, ref_min_spread_time_minutes=15):
    """
    takes an Isotope a run and a database and gives the isotopeshift to the reference!
    This will perform a linear fit to the references center positions versus time stamp and
    will extrapolate a center position of a the reference at the time of acquiring the desired isotope.
    If the references are only before or after the isotope of interest the mean value is taken.
    :return: list, (shifts, shiftErrors, shifts_weighted_mean, statErr, systErr, rChi)
    """
    print('Open DB', db)

    conf_staterrform_systerrform = TiTs.select_from_db(db, 'config, statErrForm, systErrForm', 'Combined',
                                                       [['iso', 'parname', 'run'], [iso, 'shift', run]],
                                                       caller_name=__name__)
    if conf_staterrform_systerrform:
        (config, statErrForm, systErrForm) = conf_staterrform_systerrform[0]
    else:
        return [None] * 6
    '''config needs to have this shape:
    [
    (['dataREF1.*','dataREF2.*',...],
    ['dataINTERESTING1.*','dataINT2.*',...],
    ['dataREF4.*',...]),
    ([...],[...],[...]),
    ...
    ]
    '''
    config = ast.literal_eval(config)
    print('Combining', iso, 'shift')
    print('config is:')
    for each in config:
        print(each)

    ref_refrun = TiTs.select_from_db(db, 'Lines.reference, lines.refRun',
                                     'Runs JOIN Lines ON Runs.lineVar = Lines.lineVar', [['Runs.run'], [run]],
                                     caller_name=__name__)
    if ref_refrun is not None:
        (ref, refRun) = ref_refrun[0]
    else:
        return [None] * 6
    # each block is used to measure the isotope shift once
    shifts = []
    shiftErrors = []
    dateIso = []  # dates as string for average plot
    print(config)
    for block in config:
        block_shifts = []
        block_shifts_errs = []
        slope = 1
        slope_err = 0
        offset = 0
        offset_err = 0
        pre_ref_files, iso_files, post_ref_files = block
        ref_files = pre_ref_files + post_ref_files
        print('ref_files:')
        for each in ref_files:
            print(each)
        if len(ref_files) == 0:
            logging.warning('warning, no ref files found!')
            return [None] * 6
        ref_centers, ref_errs, ref_dates, ref_files = extract(ref, 'center', refRun, db, ref_files)
        ref_dates_date_time = [datetime.strptime(each, '%Y-%m-%d %H:%M:%S') for each in ref_dates]
        ref_dates_date_time_float = [datetime.strptime(each, '%Y-%m-%d %H:%M:%S').timestamp() for each in ref_dates]
        first_ref = np.min(ref_dates_date_time_float)
        ref_dates_float_relative = [each - first_ref for each in ref_dates_date_time_float]
        refs_elapsed_s = np.max(ref_dates_date_time_float) - first_ref
        iso_centers, iso_errs, iso_dates, iso_files = extract(iso, 'center', run, db, iso_files)
        iso_dates_datetime = [datetime.strptime(each, '%Y-%m-%d %H:%M:%S') for each in iso_dates]
        iso_dates_datetime_float = [datetime.strptime(each, '%Y-%m-%d %H:%M:%S').timestamp() for each in iso_dates]
        iso_date_float_relative = [each - first_ref for each in iso_dates_datetime_float]
        dateIso += iso_dates
        # first assume a constant slope
        slope = 0
        offset, offset_err, rChi = weightedAverage(ref_centers, ref_errs)
        plt_label = 'mean val: %.1f +/- %.1f rChi: %.1f' % (offset, offset_err, rChi)
        if len(ref_files) > 1:
            # fit dates vs center position of refs
            if refs_elapsed_s / 60 > ref_min_spread_time_minutes:
                # if refs are spread sufficiently in time perform linear fit and overwrite slope etc.
                print('ref dates are: ', ref_dates_float_relative)
                use_absoult_sigma = len(ref_files) == 2
                # fit relative to the first reference
                popt, pcov = curve_fit(straight_func, ref_dates_float_relative, ref_centers,
                                       None, ref_errs, absolute_sigma=use_absoult_sigma)
                perr = np.sqrt(np.diag(pcov))
                print('optimized parameters:')
                print(popt, pcov, perr)
                slope, offset = popt
                slope_err, offset_err = perr
                plt_label = 'lin. fit to ref: slope %.1e +/- %.1e\nlin. fit to ref: offset %.1f +/- %.1f' % (
                    slope, slope_err, offset, offset_err)

            else:
                logging.warning('WARNING, while calculating the isotope shift for %s '
                                'in run %s the reference were not spread for more than %.0f minutes.\n '
                                'Therefore the linear fit will be a constants '
                                'value from the weighted average of references.'
                                % (iso, run, ref_min_spread_time_minutes))
        # calc iso shift for each ref
        for i, iso_rel_date in enumerate(iso_date_float_relative):
            # center of iso - center of ref at this time evaluated from linear fit
            block_shifts += [iso_centers[i] - straight_func(iso_rel_date, slope, offset)]
            # gaussian error prop:
            ref_extrapol_err = np.sqrt((slope_err * iso_rel_date) ** 2 + offset_err ** 2)
            block_shifts_errs += [np.sqrt(iso_errs[i] ** 2 + ref_extrapol_err ** 2)]

        # plot and save on disc
        pic_name = 'shift_%s_' % iso
        for file in iso_files:
            pic_name += file.split('.')[0] + '_'
        pic_name = pic_name[:-1] + '.png'
        file_name = os.path.join(os.path.dirname(db), 'shift_pics', pic_name)
        plt.plot_iso_shift_time_dep(
            ref_dates_date_time, ref_dates_date_time_float, ref_centers, ref_errs, ref,
            iso_dates_datetime, iso_dates_datetime_float, iso_centers, iso_errs, iso,
            slope, offset, plt_label, (block_shifts, block_shifts_errs), file_name, show_plot=show_plot)
        shifts += block_shifts
        shiftErrors += block_shifts_errs
    # in the end get the mean value of all shifts:
    shifts_weighted_mean, err, rChi = weightedAverage(shifts, shiftErrors)
    systE = functools.partial(shiftErr, iso, run, db)

    # systErrForm, e.g. systE(accVolt_d=1.5 * 10 ** -4, offset_d=1.5 * 10 ** -4)
    # therefore systE is needed as a "variable"
    statErr = eval(statErrForm)
    systErr = eval(systErrForm)
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?
            WHERE iso = ? AND parname = ? AND run = ?''',
                (shifts_weighted_mean, statErr, systErr, rChi, iso, 'shift', run))
    con.commit()
    con.close()
    print('shifts:', shifts)
    print('shiftErrors:', shiftErrors)
    print('Mean of shifts: %.2f(%.0f)[%.0f] MHz' % (shifts_weighted_mean, statErr * 100, systErr * 100))
    print('rChi: %.2f' % rChi)
    combined_plots_dir = os.path.join(os.path.split(db)[0], 'combined_plots')
    avg_fig_name = os.path.join(combined_plots_dir, iso + '_' + run + '_shift.png')
    plotdata = (
        dateIso, shifts, shiftErrors, shifts_weighted_mean,
        statErr, systErr, ('k.', 'r'), False, avg_fig_name, '%s_shift [MHz]' % iso)
    plt.plotAverage(*plotdata)
    if show_plot:
        plt.show(True)
    plt.clear()

    return shifts, shiftErrors, shifts_weighted_mean, statErr, systErr, rChi


def straight_func(x, slope, offset):
    """
    a straight function that can be used to fit to data.
    :return: x * slope + offset
    """
    return x * slope + offset


def applyChi(err, rChi):
    '''Increases error by sqrt(rChi^2) if necessary. Works for several rChi as well'''
    return err * np.max([1, np.sqrt(rChi)])


def gaussProp(*args):
    '''Calculate sqrt of squared sum of args, as in gaussian error propagation'''
    return np.sqrt(sum(x ** 2 for x in args))


def shiftErr(iso, run, db, accVolt_d, offset_d, syst=0):
    if str(iso)[-1] == 'm' and str(iso)[-2] == '_':
        iso = str(iso)[:-2]
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(
        '''SELECT Lines.reference, lines.frequency FROM Runs JOIN Lines ON Runs.lineVar = Lines.lineVar WHERE Runs.run = ?''',
        (run,))
    (ref, nu0) = cur.fetchall()[0]
    cur.execute('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''', (iso,))
    (mass, mass_d) = cur.fetchall()[0]
    cur.execute('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''', (ref,))
    (massRef, massRef_d) = cur.fetchall()[0]
    deltaM = np.absolute(mass - massRef)
    cur.execute('''SELECT offset, accVolt, voltDivRatio FROM Files WHERE type = ?''', (iso,))
    (offset, accVolt, voltDivRatio) = cur.fetchall()[0]
    voltDivRatio = ast.literal_eval(voltDivRatio)
    if isinstance(voltDivRatio['offset'], float):
        mean_offset_div_ratio = voltDivRatio['offset']
    else:  # if the offsetratio is not a float it is supposed to be a dict.
        mean_offset_div_ratio = np.mean(list(voltDivRatio['offset'].values()))
    if isinstance(offset, str):
        offset = ast.literal_eval(offset)
        if isinstance(offset, list):
            # offset will be list for each track
            offset = np.mean(offset)
    offset = np.abs(offset) * mean_offset_div_ratio
    cur.execute('''SELECT offset FROM Files WHERE type = ?''', (ref,))
    (refOffset,) = cur.fetchall()[0]
    if isinstance(refOffset, str):
        refOffset = ast.literal_eval(refOffset)
        if isinstance(refOffset, list):
            # offset will be list for each track
            refOffset = np.mean(refOffset)
    accVolt = np.absolute(refOffset) * mean_offset_div_ratio + accVolt * voltDivRatio['accVolt']

    fac = nu0 * np.sqrt(Physics.qe * accVolt / (2 * mass * Physics.u * Physics.c ** 2))
    print('systematic error inputs caused by error of...\n...acc Voltage:',
          fac * (0.5 * (offset / accVolt + deltaM / mass) * (accVolt_d)),
          'MHz  ...offset Voltage',
          fac * offset * offset_d / accVolt,
          'MHz  ...masses:',
          fac * (mass_d / mass + massRef_d / massRef),
          'MHz')

    return np.sqrt(np.square(fac * (np.absolute(0.5 * (offset / accVolt + deltaM / mass) * (accVolt_d))
                                    + np.absolute(offset * offset_d / accVolt) + np.absolute(
        mass_d / mass + massRef_d / massRef)))
                   + np.square(syst))


def avgErr(iso, db, avg, par, accVolt_d, offset_d, syst=0):
    print('Building AverageError...')
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''SELECT frequency FROM Lines''')
    (nu0) = cur.fetchall()[0][0]
    cur.execute('''SELECT mass, mass_d, I FROM Isotopes WHERE iso = ?''', (iso,))
    (mass, mass_d, spin) = cur.fetchall()[0]
    if str(iso)[-1] == 'm' and str(iso)[-2] == '_':
        iso = str(iso)[:-2]
    cur.execute('''SELECT offset, accVolt, voltDivRatio FROM Files WHERE type = ?''', (iso,))
    (offset, accVolt, voltDivRatio) = cur.fetchall()[0]
    voltDivRatio = ast.literal_eval(voltDivRatio)
    if isinstance(voltDivRatio['offset'], float):
        mean_offset_div_ratio = voltDivRatio['offset']
    else:  # if the offsetratio is not a float it is supposed to be a dict.
        mean_offset_div_ratio = np.mean(list(voltDivRatio['offset'].values()))
    if isinstance(offset, str):
        offset = ast.literal_eval(offset)
        if isinstance(offset, list):
            offset = np.mean(offset)

    cur.execute('''SELECT Jl, Ju FROM Lines''')
    (jL, jU) = cur.fetchall()[0]
    accVolt = accVolt * voltDivRatio['accVolt'] - offset * mean_offset_div_ratio
    cF = 1
    cF_dist = 1
    '''
    for the A- and B-Factor, the (energy-) distance of the peaks
    can be calculated with the help of the Casimir Factor:
    C_F = F(F+1)-j(j+1)-I(I+1).
    This distance can be converted into an offset voltage
    so we can use the same error formula as for the isotope shift.
    '''
    if par == 'Au':
        cF = (jU + spin) * (jU + spin + 1) - jU * (jU + 1) - spin * (spin + 1)
        cF_dist = cF * 2
    elif par == 'Al':
        cF = (jL + spin) * (jL + spin + 1) - jL * (jL + 1) - spin * (spin + 1)
        cF_dist = cF * 2
    elif par == 'Bu':
        cF = (jU + spin) * (jU + spin + 1) - jU * (jU + 1) - spin * (spin + 1)
        if (spin * jU * (2 * spin - 1) * (2 * jU - 1)) != 0:
            cF_dist = 4 * (3 / 2 * cF * (cF + 1) - 2 * spin * jU * (spin + 1) * (jU + 1)) / (
                spin * jU * (2 * spin - 1) * (2 * jU - 1))
        else:
            cF_dist = 1
    elif par == 'Bl':
        cF = (jL + spin) * (jL + spin + 1) - jL * (jL + 1) - spin * (spin + 1)
        if (spin * jL * (2 * spin - 1) * (2 * jL - 1)) != 0:
            cF_dist = 4 * (3 / 2 * cF * (cF + 1) - 2 * spin * jL * (spin + 1) * (jL + 1)) / (
                spin * jL * (2 * spin - 1) * (2 * jL - 1))
        else:
            cF_dist = 1
    else:
        pass
    print('casimir factor:', cF)
    distance = np.abs(avg * cF_dist)
    print('frequency between left and right edge:', distance)
    distance = distance / Physics.diffDoppler(nu0, accVolt, mass)
    print('voltage between left and right edge:', distance)
    fac = nu0 * np.sqrt(Physics.qe * accVolt / (2 * mass * Physics.u * Physics.c ** 2))
    return np.sqrt(
        np.square(fac * (np.absolute(0.5 * (distance / accVolt) * accVolt_d) +
                         np.absolute(distance * offset_d / accVolt) +
                         np.absolute(mass_d / mass))) +
        np.square(syst)) / cF_dist
    # The uncertainty on the A- & B-Factors needs to be scaled down again
