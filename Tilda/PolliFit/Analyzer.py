"""
Created on 21.05.2014

@author: hammen, gorges

The Analyzer can extract() parameters from fit results, combineRes() to get weighted averages
and combineShift() to calculate isotope shifts.
"""

import ast
import functools
import os
import sqlite3
import logging
from datetime import datetime
from copy import deepcopy

import numpy as np
from scipy.optimize import curve_fit

from Tilda.PolliFit import MPLPlotter as plt
from Tilda.PolliFit import Physics
import Tilda.PolliFit.TildaTools as TiTs
import Tilda.PolliFit.Measurement.MeasLoad as Loader


def getFiles(iso, run, db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('SELECT file, pars FROM FitRes WHERE iso = ? AND run = ? ORDER BY file ', (iso, run))
    e = cur.fetchall()
    con.close()

    return [f[0] for f in e]


def get_date_date_err_to_files(db, filelist):
    """
    get the date and date err of all files in filelist
    :param db: str, bs path to sqlite database
    :param filelist: list, list of strings as listed in table 'Files'
    :return: list of tuples, [(file, date, errDateInS), ..]
    """
    e = []
    con = sqlite3.connect(db)
    cur = con.cursor()
    for f in filelist:
        cur.execute('SELECT file, date, errDateInS FROM main.Files WHERE file = ?  ', (f,))
        e += cur.fetchall()
    con.close()
    return e


def extract(iso, par, run, db, fileList=None, prin=True):
    """
    Return a list of values of par of iso, filtered by files in fileList
    """
    print('Extracting', iso, par, run)
    if iso == 'all':
        fits = TiTs.select_from_db(db, 'file, pars', 'FitRes', [['run'], [run]], 'ORDER BY file',
                                   caller_name=__name__)
    else:
        fits = TiTs.select_from_db(db, 'file, pars', 'FitRes', [['iso', 'run'], [iso, run]], 'ORDER BY file',
                                   caller_name=__name__)
    if fits is not None:
        if fileList is not None:
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

        if fileList is not None:
            for f in fileList:
                if f not in files:
                    print('Warning:', f, r'not found while extracting from db\Files!')
        return vals, errs, date_list, files
    else:
        return None, None, None, None


def weightedAverage(vals, errs):
    """
    Calculates the weighted average, its standard error and the reduced chi-square.

    :param vals: The values of which the weighted average is calculated.
    :param errs: The standard errors of the values. If None, the standard deviation of the values is estimated
     and used as their error. Note that this is the unweighted average with estimated errors.
    :return: The weighted average, its standard error and the reduced chi-square.
    """
    if errs is None:
        return average(vals, None)
    _vals, _errs = np.asarray(vals), np.asarray(errs)

    weights = 1 / _errs ** 2
    _average = np.sum(_vals * weights) / np.sum(weights)
    _av_err = np.sqrt(1 / np.sum(weights))
    if _vals.size == 1:
        r_chi = 0
    else:
        r_chi = np.sum((_vals - _average) ** 2 * weights) / (_vals.size - 1)
    return _average, _av_err, r_chi


def average(vals, errs=None):
    """
    Calculates the average, its standard error and the reduced chi-square.

    :param vals: The values of which the average is calculated.
    :param errs: The standard errors of the values. If None, the standard deviation of the values is estimated
     and used as their error. Note that if errs is None, the reduced chi-square will always be 1.
    :return: The average, its standard error and the reduced chi-square.
    """
    _vals = np.asarray(vals)
    if errs is None:
        _errs = np.full(_vals.shape, np.std(_vals, ddof=1))
    else:
        _errs = np.asarray(errs)

    _average = np.sum(_vals) / _vals.size
    _av_err = np.sqrt(np.sum(_errs ** 2)) / _vals.size

    if _vals.size == 1:
        r_chi = 0
    else:
        r_chi = np.sum(((_vals - _average) / _errs) ** 2) / (_vals.size - 1)
    return _average, _av_err, r_chi


def combineRes(iso, par, run, db, weighted=True, print_extracted=True,
               show_plot=False, only_this_files=None, write_to_db=True,
               combine_from_par='', combine_from_multipl=1.0, combine_from_mult_err=0.0, estimate_err=False):
    """
    Calculate weighted average of par using the configuration specified in the db.

    :param iso:
    :param par:
    :param run:
    :param db:
    :param weighted:
    :param print_extracted:
    :param show_plot:
    :param only_this_files:
    :param write_to_db:
    :param combine_from_par:
    :param combine_from_multipl:
    :param combine_from_mult_err:
    :param estimate_err: Whether to estimate the error of the average solely from the given values.
     This only works for the unweighted average.

    :rtype: tuple.
    :return: Tuple of results.
    """
    print('Open DB', db)

    con = sqlite3.connect(db)
    cur = con.cursor()

    cur.execute('INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)', (iso, par, run))
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

    if only_this_files is not None:
        config = only_this_files

    if combine_from_par == '':
        print('Combining', iso, par)

        vals, errs, date, files = extract(iso, par, run, db, config, prin=print_extracted)
    else:
        print('Combining ', iso, par, ' from par %s with multiplication %s +/- %s'
              % (combine_from_par, combine_from_multipl, combine_from_mult_err))
        vals_temp, errs_temp, date, files = extract(iso, combine_from_par, run, db, config, prin=print_extracted)
        vals = [val * combine_from_multipl for val in vals_temp]
        errs = [np.sqrt(
            (combine_from_mult_err * vals[i]) ** 2 + (err * combine_from_multipl) ** 2
        ) for i, err in enumerate(errs_temp)]

    print(files)
    if weighted:
        avg, err, rChi = weightedAverage(vals, None if estimate_err else errs)
    else:
        avg, err, rChi = average(vals, None if estimate_err else errs)
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
        cur.execute('UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?'
                    'WHERE iso = ? AND parname = ? AND run = ?', (avg, statErr, systErr, rChi, iso, par, run))
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
    """
    config needs to have this shape:
    [
    (['dataREF1.*','dataREF2.*',...],
    ['dataINTERESTING1.*','dataINT2.*',...],
    ['dataREF4.*',...]),
    ([...],[...],[...]),
    ...
    ]
    """
    config = ast.literal_eval(config)
    print('Combining', iso, 'shift')

    (ref, refRun) = TiTs.select_from_db(db, 'Lines.reference, lines.refRun',
                                        'Runs JOIN Lines ON Runs.lineVar = Lines.lineVar', [['Runs.run'], [run]],
                                        caller_name=__name__)[0]
    # each block is used to measure the isotope shift once
    shifts = []
    shiftErrors = []
    dateIso = []
    becola_files = False  # neccessary to adapt for becola files which are identified by run number rather than date
    numIso = []  # necessary for BECOLA files from Ni run, since they don't have reliable dates
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
        for f in files:
            file_name, file_ext = f.split('.', 1)
            if 'BECOLA' in file_name:  # BECOLA files from nickel analysis have naming scheme 'BECOLA_runno.xml'
                becola_files = True
                prefix, iso_num = file_name.split('_', 1)
                numIso.append(int(iso_num))

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
    cur.execute('UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?'
        'WHERE iso = ? AND parname = ? AND run = ?', (val, statErr, systErr, rChi, iso, 'shift', run))
    con.commit()
    con.close()
    print('shifts:', shifts)
    print('shiftErrors:', shiftErrors)
    print('Mean of shifts:', val)
    combined_plots_dir = os.path.join(os.path.split(db)[0], 'combined_plots')
    avg_fig_name = os.path.join(combined_plots_dir, iso + '_' + run + '_shift.png')
    plotdata = (
        dateIso, shifts, shiftErrors, val, statErr, systErr, ('k.', 'r'), False, avg_fig_name, '%s_shift [MHz]' % iso)
    plotdataFiles = (
        numIso, shifts, shiftErrors, val, statErr, systErr, ('k.', 'r'), False, avg_fig_name, '%s_shift [MHz]' % iso)
    if show_plot:
        if becola_files:
            plt.plotAverageBECOLA(*plotdataFiles)
        else:
            plt.plotAverage(*plotdata)
        plt.show(True)
    # plt.clear()
    return (shifts, shiftErrors, val, statErr, systErr, rChi)


def combineShiftByTime(iso, run, db, show_plot=False, ref_min_spread_time_minutes=15,
                       pic_format='.png', font_size=12, default_date_err_s=15 * 60, overwrite_file_num_det=None,
                       store_to_file_in_combined_plots=''):
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
    """
    config needs to have this shape:
    [
    (['dataREF1.*','dataREF2.*',...],
    ['dataINTERESTING1.*','dataINT2.*',...],
    ['dataREF4.*',...]),
    ([...],[...],[...]),
    ...
    ]
    """
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
    fileIso = []  # all isotope files
    all_iso_file_nums = []
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
        ref_date_errs = list(list(zip(*get_date_date_err_to_files(db, ref_files)))[2])
        ref_date_errs = [err_found if err_found > 0 else default_date_err_s for err_found in ref_date_errs]
        first_ref = np.min(ref_dates_date_time_float)
        ref_dates_float_relative = [each - first_ref for each in ref_dates_date_time_float]
        refs_elapsed_s = np.max(ref_dates_date_time_float) - first_ref
        ref_file_nums = TiTs.get_file_numbers(ref_files, user_overwrite=overwrite_file_num_det)

        iso_centers, iso_errs, iso_dates, iso_files = extract(iso, 'center', run, db, iso_files)
        iso_dates_datetime = [datetime.strptime(each, '%Y-%m-%d %H:%M:%S') for each in iso_dates]
        iso_dates_datetime_float = [datetime.strptime(each, '%Y-%m-%d %H:%M:%S').timestamp() for each in iso_dates]
        iso_date_errs = list(list(zip(*get_date_date_err_to_files(db, iso_files)))[2])
        iso_date_errs = [err_found if err_found > 0 else default_date_err_s for err_found in iso_date_errs]
        iso_date_float_relative = [each - first_ref for each in iso_dates_datetime_float]
        iso_file_nums = TiTs.get_file_numbers(iso_files, user_overwrite=overwrite_file_num_det)
        all_iso_file_nums += iso_file_nums,

        dateIso += iso_dates,
        fileIso += iso_files,
        # first assume a constant slope
        slope = 0
        offset, offset_err, rChi = weightedAverage(ref_centers, ref_errs)
        plt_label = 'mean val: %.1f +/- %.1f rChi: %.1f' % (offset, offset_err, rChi)
        cor_sl_off = 0
        if len(ref_files) > 1:
            # fit dates vs center position of refs
            if refs_elapsed_s / 60 > ref_min_spread_time_minutes:
                # if refs are spread sufficiently in time perform linear fit and overwrite slope etc.
                # x-error not relevant for fitting since all references
                # will always take roughly the same time to acquire
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
                cov_sl_off = pcov[0][1]
                cor_sl_off = cov_sl_off / (slope_err * offset_err)

                plt_label = 'lin. fit to ref: slope %.1e +/- %.1e\nlin. fit to ref: offset %.1f +/- %.1f' % (
                    slope, slope_err, offset, offset_err)
                print('optimized parameter from curve fit:')
                print('slope: ', slope, ' +/- ', slope_err)
                print('offset: ', offset, ' +/- ', offset_err)
                print('correlation between slope and offset: ', cor_sl_off)

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
            ref_extrapol_err = np.sqrt((slope_err * iso_rel_date) ** 2 +
                                       offset_err ** 2 +
                                       (slope * iso_date_errs[i]) ** 2)
            # now also with correlation term:
            ref_extrapol_err_cor = np.sqrt((slope_err * iso_rel_date) ** 2 +
                                           offset_err ** 2 +
                                           (slope * iso_date_errs[i]) ** 2 +
                                           2 * (iso_rel_date * slope_err) * offset_err * cor_sl_off)
            print('iso shift errs, uncorrelated %.3f and correlated %.3f, correlation: %.2f' %
                  (ref_extrapol_err, ref_extrapol_err_cor, cor_sl_off))
            block_shifts_errs += [np.sqrt(iso_errs[i] ** 2 + ref_extrapol_err_cor ** 2)]

        # plot and save on disc
        pic_name = 'shift_%s_%s_files_' % (iso, run)
        for fn in iso_file_nums:
            pic_name += fn + '_'
        if isinstance(pic_format, list):
            for pic_form in pic_format:
                pic_name = pic_name[:-1] + pic_form
                file_name = os.path.join(os.path.dirname(db), 'shift_pics', pic_name)
                plt.plot_iso_shift_time_dep(
                    ref_files, ref_file_nums, ref_dates_date_time, ref_dates_date_time_float,
                    ref_date_errs, ref_centers, ref_errs, ref,
                    iso_files, iso_file_nums, iso_dates_datetime, iso_dates_datetime_float,
                    iso_date_errs, iso_centers, iso_errs, iso,
                    slope, offset, plt_label, (block_shifts, block_shifts_errs), file_name, show_plot=show_plot,
                    font_size=font_size)
        else:
            pic_name = pic_name[:-1] + pic_format
            file_name = os.path.join(os.path.dirname(db), 'shift_pics', pic_name)
            plt.plot_iso_shift_time_dep(
                ref_files, ref_file_nums, ref_dates_date_time, ref_dates_date_time_float,
                ref_date_errs, ref_centers, ref_errs, ref,
                iso_files, iso_file_nums, iso_dates_datetime, iso_dates_datetime_float,
                iso_date_errs, iso_centers, iso_errs, iso,
                slope, offset, plt_label, (block_shifts, block_shifts_errs), file_name, show_plot=show_plot,
                font_size=font_size)
        shifts += block_shifts,
        shiftErrors += block_shifts_errs,
    # in the end get the mean value of all shifts:
    print('dates:', dateIso)
    print('shifts:', shifts)
    print('shiftErrors:', shiftErrors)
    date_isos_blockw = deepcopy(dateIso)
    shifts_blockw = deepcopy(shifts)
    shiftErrors_blockw = deepcopy(shiftErrors)
    file_iso_blockw = deepcopy(fileIso)

    # # blockwise logics:
    # w_avg_blockwise = []  # will get one tuple for each block of isotope shifts containing:
    # # (w_avg_block, w_avg_err_block, w_avg_rChi_block, w_avg_appl_chi_err)
    # stdev_blockwise = []  # will hold the weighted mean and the standard deviation for each block as a tuple of each
    # # (w_avg_block, w_avg_appl_chi_err, stdev_block, max(stdev_block, w_avg_appl_chi_err))
    # for files_block, dates_block, shifts_block, shift_errs_block in zip(
    #         file_iso_blockw, date_isos_blockw, shifts_blockw, shiftErrors_blockw):
    #     w_avg_block, w_avg_err_block, w_avg_rChi_block = weightedAverage(shifts_block, shift_errs_block)
    #     w_avg_appl_chi_err = applyChi(w_avg_block, w_avg_rChi_block)
    #     w_avg_blockwise += (w_avg_block, w_avg_err_block, w_avg_rChi_block, w_avg_appl_chi_err),
    #
    #     stdev_block = standard_dev(shifts_block, shift_errs_block, w_avg_block)
    #     stdev_blockwise += (w_avg_block, w_avg_appl_chi_err, stdev_block, max(stdev_block, w_avg_appl_chi_err)),
    #
    # # now calculate the weighted mean of all blocks:
    # w_avg_blocks_list = list(list(zip(*w_avg_blockwise))[0])  # get a list with all w_avgs from above
    # w_avg_appl_chi_err_list = list(list(zip(*w_avg_blockwise))[3])  # same with the applied chi err
    # w_avg_from_blocks, w_avg_err_from_blocks, w_avg_from_blocks_rChi = weightedAverage(
    #     w_avg_blocks_list, w_avg_appl_chi_err_list)
    # w_avg_apply_chi_err_from_blocks = applyChi(w_avg_err_from_blocks, w_avg_from_blocks_rChi)
    #
    # # now the same with the weights from the standard deviation and error from the stdev:
    # # get a list with all stdev uncertainties from above:
    # stdev_max_err_blocks_list = np.array(list(list(zip(*stdev_blockwise))[3]))  # one float for each block
    # stdev_weights_blocks_list = 1 / np.square(stdev_max_err_blocks_list)
    # stdev_from_blocks = standard_dev(w_avg_blocks_list, stdev_max_err_blocks_list, w_avg_from_blocks)
    # w_avg_from_block_stdev_uncert = None  #TODO or not ...

    # for backwards compatibility, flatten everything:
    dateIso = [item for sublist in dateIso for item in sublist]
    shifts = [item for sublist in shifts for item in sublist]
    shiftErrors = [item for sublist in shiftErrors for item in sublist]
    fileIso = [item for sublist in fileIso for item in sublist]
    all_iso_file_nums_flat = [item for sublist in all_iso_file_nums for item in sublist]

    # weighted average over everything:
    shifts_weighted_mean, err, rChi = weightedAverage(shifts, shiftErrors)
    systE = functools.partial(shiftErr, iso, run, db)

    # systErrForm, e.g. systE(accVolt_d=1.5 * 10 ** -4, offset_d=1.5 * 10 ** -4)
    # therefore systE is needed as a "variable"
    statErr = eval(statErrForm)
    systErr = eval(systErrForm)
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?'
                'WHERE iso = ? AND parname = ? AND run = ?',
                (shifts_weighted_mean, statErr, systErr, rChi, iso, 'shift', run))
    con.commit()
    con.close()

    print('date\tfile\tshift / MHz\tshiftError / MHz')
    for i, sh in enumerate(shifts):
        print('%s\t%s\t%.4f\t%.4f' % (dateIso[i], fileIso[i], sh, shiftErrors[i]))
    print('Mean of shifts: %.2f(%.0f)[%.0f] MHz' % (shifts_weighted_mean, statErr * 100, systErr * 100))
    print('rChi: %.2f' % rChi)
    combined_plots_dir = os.path.join(os.path.split(db)[0], 'combined_plots')
    if not os.path.isdir(combined_plots_dir):
        os.mkdir(combined_plots_dir)
    if store_to_file_in_combined_plots:
        file_to_store_to = os.path.join(combined_plots_dir, store_to_file_in_combined_plots)
        write_header = not os.path.isfile(file_to_store_to)
        f_open_append = open(file_to_store_to, 'a+')
        if write_header:
            f_open_append.write(
                '#iso\tfile\tfileNum\tshiftFile\tshiftFileStatErr\tisoMeanShift'
                '\tisoMeanShiftStatErr\tisoMeanShiftSystErr\tisoMeanShiftRChi2\n')
        for sh_i, sh in enumerate(shifts):
            f_open_append.write('%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n'
                                % (iso, fileIso[sh_i], all_iso_file_nums_flat[sh_i], sh, shiftErrors[sh_i],
                                   shifts_weighted_mean, statErr, systErr, rChi))
        f_open_append.close()

    avg_fig_name = os.path.join(combined_plots_dir, iso + '_' + run + '_shift.png')
    plotdata = (
        dateIso, shifts, shiftErrors, shifts_weighted_mean,
        statErr, systErr, ('k.', 'r'), False, avg_fig_name, '%s_shift [MHz]' % iso)
    plt.plotAverage(*plotdata)
    if show_plot:
        plt.show(True)
    plt.clear()

    return shifts, shiftErrors, shifts_weighted_mean, statErr, systErr, rChi


def combineShiftOffsetPerBunchDisplay(iso, run, db, show_plot=False):
    """
    takes an Isotope a run and a database and gives offsets per bunch for all
     Isotopes involved in one IsotopeShift value
        :return: offsets, offsetErrs, config

        offsets: list, [[ref_offset0, ref_offset1, ... ], [iso_offset0, iso_offset1, ...]]
        offsetErrs: list, [[ref_offset_err0, ref_offset_err1, ... ], [iso_offset_err0, iso_offset_err1, ...]]
        config: list, [(ref_file_str0, ref_file_str1, ...), (iso_file_str0, ...), (ref_file_str0, ...)]
    """
    print('Open DB', db)
    # get the shift config for this iso and run:
    conf_staterrform_systerrform = TiTs.select_from_db(db, 'config, statErrForm, systErrForm', 'Combined',
                                                       [['iso', 'parname', 'run'], [iso, 'shift', run]],
                                                       caller_name=__name__)
    if conf_staterrform_systerrform:
        (config, statErrForm, systErrForm) = conf_staterrform_systerrform[0]
    else:
        return [None] * 6
    """
    'config needs to have this shape:
    [
    (['dataREF1.*','dataREF2.*',...],
    ['dataINTERESTING1.*','dataINT2.*',...],
    ['dataREF4.*',...]),
    ([...],[...],[...]),
    ...
    ]
    """
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
    offsets = []
    offsetErrors = []
    dateIso = []  # dates as string for average plot
    print(config)
    for block in config:
        block_offsets = []
        block_offsets_errs = []
        pre_ref_files, iso_files, post_ref_files = block
        TiTs.get_file_numbers(iso_files, )
        ref_files = pre_ref_files + post_ref_files
        print('ref_files:')
        for each in ref_files:
            print(each)
        if len(ref_files) == 0:
            logging.warning('warning, no ref files found!')
            return [None] * 6
        ref_offsets, ref_errs, ref_dates, ref_files = extract(ref, 'offset', refRun, db, ref_files)
        for i, ref_f in enumerate(ref_files):
            fil_path_rel = TiTs.select_from_db(db, 'filePath', 'Files',
                                               [['file'], [ref_f]], caller_name=__name__)
            fil_path_abs = os.path.join(os.path.dirname(db), fil_path_rel[0][0])
            meas = Loader.load(fil_path_abs, db, raw=True)
            scans = meas.nrScans[0]  # take number of scans from track0
            bunches_per_step = meas.nrBunches[0]
            ref_offsets[i] = ref_offsets[i] / scans / bunches_per_step
            ref_errs[i] = ref_errs[i] / scans / bunches_per_step
        ref_dates_date_time = [datetime.strptime(each, '%Y-%m-%d %H:%M:%S') for each in ref_dates]
        ref_dates_date_time_float = [datetime.strptime(each, '%Y-%m-%d %H:%M:%S').timestamp() for each in ref_dates]
        ref_date_errs = list(list(zip(*get_date_date_err_to_files(db, ref_files)))[2])
        first_ref = np.min(ref_dates_date_time_float)
        ref_dates_float_relative = [each - first_ref for each in ref_dates_date_time_float]
        refs_elapsed_s = np.max(ref_dates_date_time_float) - first_ref
        iso_offsets, iso_errs, iso_dates, iso_files = extract(iso, 'offset', run, db, iso_files)
        for i, iso_f in enumerate(iso_files):
            fil_path_rel = TiTs.select_from_db(db, 'filePath', 'Files',
                                               [['file'], [iso_f]], caller_name=__name__)
            fil_path_abs = os.path.join(os.path.dirname(db), fil_path_rel[0][0])
            meas = Loader.load(fil_path_abs, db, raw=True)
            scans = meas.nrScans[0]  # take number of scans from track0
            bunches_per_step = meas.nrBunches[0]
            iso_offsets[i] = iso_offsets[i] / scans / bunches_per_step
            iso_errs[i] = iso_errs[i] / scans / bunches_per_step

        iso_dates_datetime = [datetime.strptime(each, '%Y-%m-%d %H:%M:%S') for each in iso_dates]
        iso_dates_datetime_float = [datetime.strptime(each, '%Y-%m-%d %H:%M:%S').timestamp() for each in iso_dates]
        iso_date_float_relative = [each - first_ref for each in iso_dates_datetime_float]
        iso_date_errs = list(list(zip(*get_date_date_err_to_files(db, iso_files)))[2])
        dateIso += iso_dates

        offset, offset_err, rChi = weightedAverage(ref_offsets, ref_errs)
        plt_label = 'mean val: %.1f +/- %.1f rChi: %.1f' % (offset, offset_err, rChi)
        # calc iso offset for each ref
        for i, iso_rel_date in enumerate(iso_date_float_relative):
            block_offsets += [iso_offsets[i]]
            block_offsets_errs += [iso_errs[i]]

        # plot and save on disc
        pic_name = 'offset_%s_%s_' % (iso, run)
        for file in iso_files:
            pic_name += file.split('.')[0] + '_'
        pic_name = pic_name[:-1] + '.png'
        file_name = os.path.join(os.path.dirname(db), 'offset_pics', pic_name)
        offset_dir = os.path.dirname(file_name)
        if not os.path.isdir(offset_dir):
            os.mkdir(offset_dir)
        plt.plot_iso_shift_time_dep(ref_files, ref_dates_date_time, ref_dates_date_time_float, ref_date_errs,
                                    ref_offsets, ref_errs, ref, iso_files, iso_dates_datetime, iso_dates_datetime_float,
                                    iso_date_errs, iso_offsets, iso_errs, iso, 0, offset, plt_label,
                                    (block_offsets, block_offsets_errs), file_name, show_plot=show_plot,
                                    fig_name='offset', par_name='offset')
        offsets += [[ref_offsets, block_offsets]]
        offsetErrors += [[ref_errs, block_offsets_errs]]
    # in the end get the mean value of all offsets:
    # offsets_weighted_mean, err, rChi = weightedAverage(offsets, offsetErrors)

    print('offsets:', offsets)
    print('offsetErrors:', offsetErrors)
    # print('Mean of offsets: %.2f(%.0f)' % (offsets_weighted_mean, err))
    # print('rChi: %.2f' % rChi)
    combined_plots_dir = os.path.join(os.path.split(db)[0], 'combined_plots')
    avg_fig_name = os.path.join(combined_plots_dir, iso + '_' + run + '_offset.png')
    # plotdata = (
    #     dateIso, offsets, offsetErrors, offsets_weighted_mean,
    #     0, 0, ('k.', 'r'), False, avg_fig_name, '%s_offset [MHz]' % iso)
    # plt.plotAverage(*plotdata)
    # if show_plot:
    #     plt.show(True)
    # plt.clear()

    return offsets, offsetErrors, config


def straight_func(x, slope, offset):
    """
    a straight function that can be used to fit to data.
    :return: x * slope + offset
    """
    return x * slope + offset


def straight_func_vector(slopeOffset, x):
    """
    same as straight but only one parameter as vector, required for fitting using odr
    :param slopeOffset: tuple, (slope_float, offset_float)
    :param x: float, x value
    :return: y
    """
    slope, offset = slopeOffset
    return x * slope + offset


def applyChi(err, rChi):
    'Increases error by sqrt(rChi^2) if necessary. Works for several rChi as well'
    return err * np.max([1, np.sqrt(rChi)])


def standard_dev(vals, errs, w_mean=None):
    """
    calculate the standard deviation as in sqrt(sum_i^n(x_i-X)^2/(n-1))
    with x_i being the vals and X the weighted mean.
    :param vals: list, or numpy array of the values for the standard deviation
    :param errs: list or numpy array with the uncertainties of the vals,
    needed to determine weighted mean if not provided
    :param w_mean: float, (weighted) mean, if not provided will be calculated with vals and errs
    :return: float, the standard deviation of the vals to the w_mean
    """
    if len(vals) > 1:
        if isinstance(vals, list):
            vals = np.array(vals)
        if isinstance(errs, list):
            errs = np.array(errs)
        if w_mean is None:
            # calc w_mean from errs
            w_mean, w_mean_err, w_mean_rChi = weightedAverage(vals, errs)
        stdev = np.sqrt((vals - w_mean) ** 2 / (len(vals) - 1))
        return stdev

    elif len(vals) == 1 and len(errs) == 1:
        return errs[0]
    else:
        return None


def gaussProp(*args):
    'Calculate sqrt of squared sum of args, as in gaussian error propagation'
    return np.sqrt(sum(x ** 2 for x in args))


def shiftErr(iso, run, db, accVolt_d, offset_d, syst=0):
    if str(iso)[-1] == 'm' and str(iso)[-2] == '_':
        iso = str(iso)[:-2]
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('SELECT Lines.reference, lines.frequency FROM Runs JOIN Lines ON Runs.lineVar = Lines.lineVar '
                'WHERE Runs.run = ?', (run,))
    (ref, nu0) = cur.fetchall()[0]
    cur.execute('SELECT mass, mass_d FROM Isotopes WHERE iso = ?', (iso,))
    (mass, mass_d) = cur.fetchall()[0]
    cur.execute('SELECT mass, mass_d FROM Isotopes WHERE iso = ?', (ref,))
    (massRef, massRef_d) = cur.fetchall()[0]
    deltaM = np.absolute(mass - massRef)
    cur.execute('SELECT offset, accVolt, voltDivRatio FROM Files WHERE type = ?', (iso,))
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
    cur.execute('SELECT offset FROM Files WHERE type = ?', (ref,))
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
    cur.execute('SELECT frequency FROM Lines')
    (nu0) = cur.fetchall()[0][0]
    cur.execute('SELECT mass, mass_d, I FROM Isotopes WHERE iso = ?', (iso,))
    (mass, mass_d, spin) = cur.fetchall()[0]
    if str(iso)[-1] == 'm' and str(iso)[-2] == '_':
        iso = str(iso)[:-2]
    cur.execute('SELECT offset, accVolt, voltDivRatio FROM Files WHERE type = ?', (iso,))
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

    cur.execute('SELECT Jl, Ju FROM Lines')
    (jL, jU) = cur.fetchall()[0]
    accVolt = accVolt * voltDivRatio['accVolt'] - offset * mean_offset_div_ratio
    cF = 1
    cF_dist = 1
    """
    for the A- and B-Factor, the (energy-) distance of the peaks
    can be calculated with the help of the Casimir Factor:
    C_F = F(F+1)-j(j+1)-I(I+1).
    This distance can be converted into an offset voltage
    so we can use the same error formula as for the isotope shift.
    """
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
