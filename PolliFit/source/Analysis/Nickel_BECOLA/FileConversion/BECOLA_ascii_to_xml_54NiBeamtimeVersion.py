'''
Created on 30.04.2014
modified on 19.05.2020 to directly convert to xml on the server instead of npy first.

Instructions:
Copy this and the other two scripts into the folder with the .ascii files along with the excel doc of the run.
Using scp this can be done from a command line like this with %FILENAME% and %USER% replaced of course:
    scp -r %FILENAME% %USER%@sfbstore:/raid/archive/becola/e17015/data/asc_with_headers/%FILENAME%
On the server run this python3 script:
    python3 BECOLA_asciiPlus_to_xml_rebinned.py
Then copy the content of the /xml_rebinned folder back to the user folder on linix5:
    scp -r /raid/archive/becola/e17015/data/asc/xml_rebinned %USER%@linix5:/home/%USER%/asc/xml_rebinned
! Might exceed disk quota on linix5. Get quota expanded or remove the pre-run files (e.g. all 5000 files) before copy!
Get the folder from linix5 via WinSCP for example. Maybe there is a direct way to send it to the user pc but I dont know
For some reason txt files are best opened from inside the WinSCP interface and the content copypasted.
Otherwise the linebreaks somehow get lost.

@author: fsommer
'''
import matplotlib
matplotlib.use('Agg')  # make the plotting work on linux server. Must be called before any imports of pylab etc

import os, re

import sys
import warnings

import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from pylab import plot, meshgrid, cm, imshow, colorbar, show
from mpl_toolkits.mplot3d.axes3d import get_test_data
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from scipy.optimize import curve_fit
import scipy.optimize as optimization
from scipy.optimize import fmin as simplex
# "fmin" is not a sensible name for an optimization package. Rename fmin to "simplex"

from DictToXML import writeXMLfromDict  # meant to be executed in same folder as DictToXML.py so we can import this way
from BECOLA_excel_to_sql import BECOLAExelcuter  # meant to be executed in same folder so we can import this way


class BECOLAImporter():
    '''
    This object reads a BECOLA ascii file with space separated values into the SpecData structure

    Data structure for each row is a total of 31 columns:
    1scan# >> 2track# >> 3voltageStep# >> 4- >> 5time_bin >> 6scan_bin >> 7track_bin >> 8volt_bin >>
     9dac_set >> 10- >> 11- >> 12dac_read >> 13- >> 14- >> 15- >> 16time_stamp >> 17scaler1 >> 18scaler 2 >> 19scaler3 >> ...

    Rows with a hash in front are for readability --> drop
    Rows with only 12 columns are trash as well --> drop
    '''

    def __init__(self, path):
        '''Read the file'''

        print("...BECOLA Importer reading", path, end='', flush=True)
        super(BECOLAImporter, self).__init__()

        self.file = os.path.basename(path)
        # Get run number from file name. Used for identification.
        self.runNo = re.split('[._]', self.file)[1]
        self.path = path
        # excel document should be placed in same folder as this file
        self.path_to_excel = os.getcwd()

        self.is_dc_data = False  # Set this to False for time-resolved data!! Also Check standard-gates for trs
        self.divider_ratio = 201.0037  # must be set according to the voltage divider used to measure dac voltage
        self.accVolt = 29847  # buncher high voltage potential set for the runs (not noted on a per-file level)
        self.accVolt_dc = 29855  # in case this is a dc run the voltage may differ
        self.laser_unit = 'THz'  # cm-1 for import from excel. Other option 'THz'. More must be coded

        self.month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Prepare scan variables
        self.isotope = 'XX'  # This is only in the excel files!! Must be updated during excel import
        self.laserFreq = 0  # This is only in the excel files!! Must be updated during excel import
        self.nrOfScalers = 3  # This is not automated for now
        self.nrOfTracks = 1  # standard value. Will be updated if more tracks
        self.nrOfScansCompleted = 0  # variable for counting scans
        self.nrOfScans = 0  # the final number of scans completed in this file
        self.nrOfTimeBins = 1024  # Will be overwritten, but shouldn't change
        self.nrOfSteps = 1  # Will be extracted
        self.number_of_total_steps = 0  # scans + steps
        self.seqs = 0  # This is only in the excel files!!  number of sequences. This is what BECOLA calls the bunches.
        self.softwGates = []  # list for software gates per track and scaler
        # dac related variables
        self.dac_start_set_V = []  # Will be extracted per track
        self.dacStartVolt = []  # will be replaced with corrected start voltage from rebinning per track
        self.dac_stepsize_set_V = []  # Will be extracted per track
        self.dacStepSizeVolt = []   # will be replaced with corrected step size from rebinning per track
        self.dac_last_step_set_V = 0  # for calculating step sizes
        self.data_array = None  # np.zeros((self.nrOfScalers, self.nrOfTimeBins, self.nrOfSteps)) values are: cts
        self.dac_volt_deviation_arr = None  # np.zeros((self.nrOfScans, self.nrOfSteps)) values are: devFromSetVolt
        self.critical_dac_missmatch = 0  # Variable to keep track of DAC errors
        self.fit_success = []  # Variable to track result of dac volt deviation fitting
        self.mda_file_version = 'xx'  # Version of the .mda file from which the .ascii was extracted
        #  file times
        self.scan_date = []  # list of tuples with (start, end)-time for each track

        # run conversion
        self.next_line = ''  # variable in case the description for a line is in the line before that
        self.current_track = 0  # variable to note down the current track for lines where it isn't noted
        self.read_ascii(path)  # import data from ascii

        #create variable for scan voltage analysis
        self.volt_correct = []  # will be list of tuples (one for each scaler)
        self.standard_v_dev = []
        self.max_v_dev = []
        # do scan voltage analysis
        self.plot_and_fit_voltage_devs()

        self.export_read_results()
        # create variables for tilda-like data
        self.data_tuples_arr = np.empty
        self.voltage_projection_arr = np.empty
        self.excel_extraction_failed = False  # keep track whether there were problems with extracting info from excel
        self.xml_dict = {}  # dictionary from which the xml file will be written
        # create xml file from data
        self.to_xml()

    def scan_condition(self, scan_no):
        """
        can define a function here to define which scans to include
        :param scan_no: scan number (should start at 1)
        :return: bool: True or False depending on whether scan satisfies conditions
        """
        # Standard: return True (Include all scans)
        return True

    def read_ascii(self, path):
        with open(path, 'rb') as f:
            # 3D data array with scaler - time - step vs data. Size must be determined

            # Variable to keep track of the file section
            now_reading_section = 'header'

            # Variables to keep track of file progress:
            current_scan_number = 0
            scan_number_in_next_line = False

            # start read from beginning
            f.seek(0)
            for line in f:
                line_text = line.decode('ascii')
                splt = line_text.split()
                if now_reading_section is 'header':
                    # Extract basic properties of Scan
                    if splt and '#' in splt[0]:
                        # This is an info line. See what we can learn
                        if 'MDA File Version' in line_text:
                            self.mda_file_version = splt[-1]
                        elif 'Scan number' in line_text:
                            scan_number = splt[-1]  # not used at the moment
                        elif 'Total requested scan size' in line_text:
                            # The number of tracks, steps and time bins is in here!
                            scan_size = splt[-9:]
                            self.nrOfTracks = int(scan_size[2])
                            for track in range(self.nrOfTracks):
                                self.scan_date.append((None, None))  # scan_date must have dimension of tracks
                            self.nrOfSteps = int(scan_size[4])
                            self.nrOfTimeBins = int(scan_size[8])
                        elif '3-D Scans' in line_text:
                            # Header section finished
                            # The size of the data array is now know. Create it!
                            self.data_array = np.zeros(
                                (self.nrOfTracks, self.nrOfScalers, self.nrOfSteps, self.nrOfTimeBins))
                            # now reading 3-D Scan section
                            now_reading_section = '3-D Scans'

                if now_reading_section is '3-D Scans':
                    # Get a little more Info, but this is nothing really fancy...
                    # Gets start and stop times plus number of runs
                    if splt and '#' in splt[0]:
                        # This is an info line. See what we can learn
                        if 'Scan Divider' in line_text:
                            # New scan number now
                            if self.scan_condition(self.nrOfScansCompleted/self.nrOfTracks+1):
                                self.nrOfScansCompleted += 1
                        elif '4-D Scan Point' in line_text:
                            # next line will be the current track number
                            self.next_line = 'track number'
                        elif self.next_line == 'track number':
                            self.current_track = int(splt[-3])-1  # tracks in asci start at 1, need 0
                            self.next_line = ''
                        elif 'Scan time' in line_text:
                            hashsymb, scan, time, eq, month, day, year, time = splt
                            month = self.month_list.index(month) + 1
                            day = day[:-1]  # there is a comma attached. Get rid of it
                            hour, minute, second = time.split(':')
                            second, millisecond = second.split('.')
                            datetime_found = datetime(int(year), month, int(day),
                                                      int(hour), int(minute), int(second))
                            if self.scan_date[self.current_track][0] is None:
                                # This is the first read scan time. Use as start of scan
                                self.scan_date[self.current_track] = \
                                    (datetime_found, self.scan_date[self.current_track][1])
                            else:
                                # already have a start time. Make this the end time. Will be overwritten if file goes on
                                self.scan_date[self.current_track] = \
                                    (self.scan_date[self.current_track][0], datetime_found)
                        elif '1-D Scans' in line_text:
                            # end of 3-D Scans section. Now comes the real data
                            now_reading_section = '1-D Scans'
                            self.nrOfScans = self.nrOfScansCompleted/self.nrOfTracks
                            self.number_of_total_steps = self.nrOfScansCompleted * self.nrOfSteps
                            # calculate avg step size:
                            for track in range(self.nrOfTracks):
                                self.dac_stepsize_set_V[track] = np.array(self.dac_stepsize_set_V[track]).mean()
                            # prepare array for dac voltage deviations
                            self.dac_volt_deviation_arr = np.zeros((self.nrOfTracks, self.nrOfScans, self.nrOfSteps))
                    elif splt:
                        # this gets us the information about the steps in the current scan
                        # Column Descriptions:
                        #    1  [   5-D Index    ]
                        #    2  [   4-D Index    ]  Track Number
                        #    3  [   3-D Index    ]
                        #    4  [5-D Positioner 1]  DBEC:DRVR:NextRunNum, , LINEAR, , , ,
                        #    5  [4-D Positioner 1]  DBEC:DRVR:NextRegionNum, , LINEAR, , , ,
                        #    6  [3-D Positioner 1]  DBEC:DRVR:NextStepNum, , LINEAR, , , ,
                        #    7  [3-D Detector   1]  DBEC:DRVR:DAC_Volts, , volts
                        #    8  [3-D Detector   2]  DBEC:DRVR:DAC_Raw, ,
                        #    9  [3-D Detector   3]  DBEC:VD_D1285:V_RD_, voltage divider voltage read, Volts
                        #   10  [3-D Detector   4]  DBEC:BCLS:MTER_N0001:V_RD_, DBEC_BCLS:MTER_N0001:V_RD mirror, Volts
                        #   11  [3-D Detector   5]  DBEC:HPTEST1_, HPTEST1 mirror, Volts
                        #   12  [3-D Detector   6]  DBEC:BCLS:MTER_N0002:V_RD_, DBEC_BCLS:MTER_N0002:V_RD mirror, Volts
                        # But since this is coming again, later in the 1-D data, we just want the total number of scans
                        if self.scan_condition(int(splt[0])):
                            self.nrOfScans = int(splt[0])  # updated each scan and in the end == total number of scans
                            current_track = int(splt[1])-1  # starts at 1, make start at 0
                            current_step = int(splt[2])
                            current_step_dac_set_V = float(splt[6]) * 1000  # given in mV, output in V
                            if current_step == 1:  # first dac value we find is the start value
                                self.dac_start_set_V.append(current_step_dac_set_V)
                                self.dac_last_step_set_V = current_step_dac_set_V
                                self.dac_stepsize_set_V.append([])  # new list for new scaler
                            else:
                                self.dac_stepsize_set_V[current_track].append(current_step_dac_set_V-self.dac_last_step_set_V)
                                self.dac_last_step_set_V = current_step_dac_set_V

                if now_reading_section is '1-D Scans':
                    # Get the real data now.
                    if splt and '#' in splt[0]:
                        # This is an info line. See what we can learn
                        if 'Scan Divider' in line_text:
                            # New scan number now
                            self.number_of_total_steps -= 1
                        elif scan_number_in_next_line:
                            # found scan number. Format of line is: # Current point = 3 of 999
                            current_scan_number = int(splt[4])-1
                            scan_number_in_next_line = False
                        elif '5-D Scan Point' in line_text:
                            # in the next line, the scan number is specified:
                            scan_number_in_next_line = True
                    elif splt:
                        # data in long format:
                        # Column Descriptions:
                        #    1  [   5-D Index    ] scan dimension
                        #    2  [   4-D Index    ] region dimension/tracks
                        #    3  [   3-D Index    ] step dimension
                        #    4  [   2-D Index    ] group dimension; not used
                        #    5  [   1-D Index    ] time dimension
                        #    6  [5-D Positioner 1]  DBEC:DRVR:NextRunNum, , LINEAR, , , ,
                        #    7  [4-D Positioner 1]  DBEC:DRVR:NextRegionNum, , LINEAR, , , ,
                        #    8  [3-D Positioner 1]  DBEC:DRVR:NextStepNum, , LINEAR, , , ,
                        #    9  [3-D Detector   1]  DBEC:DRVR:DAC_Volts, , volts
                        #   10  [3-D Detector   2]  DBEC:DRVR:DAC_Raw, ,
                        #   11  [3-D Detector   3]  DBEC:VD_D1285:V_RD_, voltage divider voltage read, Volts
                        #   12  [3-D Detector   4]  DBEC:BCLS:MTER_N0001:V_RD_, DBEC_BCLS:MTER_N0001:V_RD mirror, Volts
                        #   13  [3-D Detector   5]  DBEC:HPTEST1_, HPTEST1 mirror, Volts
                        #   14  [3-D Detector   6]  DBEC:BCLS:MTER_N0002:V_RD_, DBEC_BCLS:MTER_N0002:V_RD mirror, Volts
                        #   15  [2-D Positioner 1]  DBEC:DRVR:NextGroupNum, , LINEAR, , , ,
                        #   16  [1-D Detector   1]  DBEC:DRVR:CountTimes, , secs
                        #   17  [1-D Detector   2]  DBEC:DRVR:CountValues_00, ,
                        #   18  [1-D Detector   3]  DBEC:DRVR:CountValues_01, ,
                        #   19  [1-D Detector   4]  DBEC:DRVR:CountValues_02, ,
                        #   20  [1-D Detector   5]  DBEC:DRVR:CountValues_03, ,
                        #   21  [1-D Detector   6]  DBEC:DRVR:CountValues_04, ,
                        #   22  [1-D Detector   7]  DBEC:DRVR:CountValues_05, ,
                        #   23  [1-D Detector   8]  DBEC:DRVR:CountValues_06, ,
                        #   24  [1-D Detector   9]  DBEC:DRVR:CountValues_07, ,
                        #   25  [1-D Detector  10]  DBEC:DRVR:CountValues_08, ,
                        #   26  [1-D Detector  11]  DBEC:DRVR:CountValues_09, ,
                        #   27  [1-D Detector  12]  DBEC:DRVR:CountValues_10, ,
                        #   28  [1-D Detector  13]  DBEC:DRVR:CountValues_11, ,
                        #   29  [1-D Detector  14]  DBEC:DRVR:CountValues_12, ,
                        #   30  [1-D Detector  15]  DBEC:DRVR:CountValues_13, ,
                        #   31  [1-D Detector  16]  DBEC:DRVR:CountValues_14, ,
                        scanNumber = int(splt[0])  # starts at 1
                        if self.scan_condition(scanNumber):
                            trackNum = int(splt[1])-1  # starts at 1, bring to 0,... for arrays
                            index = int(splt[4])
                            stepNum = int(splt[7])
                            dac_set = float(splt[8])  # in Volt
                            dac_read = float(splt[11])*self.divider_ratio/1000  # in Volt
                            timestamp = float(splt[15])
                            scaler0_data = int(splt[16])
                            scaler1_data = int(splt[17])
                            scaler2_data = int(splt[18])
                            self.data_array[trackNum, :, int(stepNum), index - 1] += [scaler0_data, scaler1_data, scaler2_data]

                            # check dac_set vs dac_read
                            dac_err = (dac_read - dac_set)  # abs error
                            if abs(dac_err) > 0.0005:  # check for critical missmatch. Converts to Volt with factor 1000, so 0.0005 = 0.5V
                                self.critical_dac_missmatch += 1 / (
                                            self.nrOfTimeBins * self.nrOfSteps * self.nrOfScansCompleted)
                            self.dac_volt_deviation_arr[trackNum, current_scan_number, int(stepNum)] = dac_err

    def plot_and_fit_voltage_devs(self):
            # Plot 3D data of dac voltage deviation: np.array (self.nrOfScans, self.nrOfSteps)) values are: devFromSetVolt
            # extract numpy array to X, Y, Z data arrays:
            for track in range(self.nrOfTracks):  # do separate for each track
                try:
                    # in case the file was bad, there might not be good data for plotting and fitting... better try.
                    x = np.arange(self.dac_volt_deviation_arr[track].shape[1])
                    y = np.arange(self.dac_volt_deviation_arr[track].shape[0])
                    X, Y = np.meshgrid(x, y)
                    Z = self.dac_volt_deviation_arr[track]

                    # find outliers and exclude from fit:
                    std_multiplier = 3  # How many std from mean is an outlier?
                    Z_sigma = np.ones(Z.shape)  # good values should have a weight of 1
                    Z_iterator = np.nditer(Z, flags=['multi_index'])
                    for each in Z_iterator:
                        if abs(each - Z.mean()) > std_multiplier * Z.std():
                            Z_sigma[Z_iterator.multi_index] = np.inf  # exclusion by setting sigma to infinity

                    # define fit function (a simple plane)
                    def plane(x, y, mx, my, coff):
                        return x * mx + y * my + coff

                    def _plane(M, *args):
                        """
                        2D function generating a 3D plane
                        :param M: xdata parameter, 2-dimensional array of x and y values
                        :param args: slope and offset passed to plane function
                        :return: array of z-values
                        """
                        x, y = M  # x: steps, y: scans
                        arr = np.zeros(x.shape)
                        arr += plane(x, y, *args)
                        return arr

                    # Define start parameters [mx, my, offset]
                    p0 = [0, 0, Z[0, 0]]
                    # make 1-D data (necessary to use curve_fit)
                    xdata = np.vstack((X.ravel(), Y.ravel()))
                    # fit
                    popt, pcov = curve_fit(_plane, xdata, Z.ravel(), p0, sigma=Z_sigma.ravel())

                    # store results
                    offset_adj = popt[1] * self.nrOfScans/2  # average over time dependence
                    self.volt_correct.append(((popt[2]+offset_adj) * 1000, popt[0] * 1000))  # (avg offset step 0, slope per step)

                    # calculate average and maximum deviation before correction
                    # fit_plane = plane(X, Y, 0,0,0)
                    # self.standard_v_dev = np.sqrt(np.square(1000 * (fit_plane - Z)).mean())
                    # print('standard deviation before correction: ' + str(self.standard_v_dev))
                    # calculate average and maximum deviation after correction
                    fit_plane = plane(X, Y, *popt)
                    self.standard_v_dev.append(np.sqrt(np.square(1000*(fit_plane-Z)).mean()))
                    self.max_v_dev.append(1000*(fit_plane-Z).max())

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_surface(X, Y, np.where(plane(X, Y, *popt)<Z, 1000*plane(X, Y, *popt), np.nan), rstride=1, cstride=1)
                    surf = ax.plot_surface(X, Y, 1000*Z, rstride=1, cstride=1, cmap='hot')
                    ax.plot_surface(X, Y, np.where(plane(X, Y, *popt)>=Z, 1000*plane(X, Y, *popt), np.nan), rstride=1, cstride=1, alpha=0.7)
                    fig.colorbar(surf)
                    plt.xlabel('step number')
                    plt.ylabel('scan number')
                    #plt.show()
                    plt.savefig('xml_rebinned/DBEC_a{}_track{}.png'.format(self.runNo, track))
                    plt.close()
                    plt.clf()

                    plt.imshow(self.dac_volt_deviation_arr[track], cmap='hot', interpolation='nearest')
                    if popt[2] == 1:
                        # error while fitting
                        plt.title('DBEC_' + self.runNo + ': Deviation from dac set voltage.\n'
                                                         '!error while fitting!')
                    else:
                        plt.title('DBEC_' + self.runNo + ': Deviation from dac set voltage.\n'
                                                         'offset: {0:.4f}V\n'
                                                         'step_slope: {1:.4f}V\n'
                                                         'scan_slope: {2:.4f}V\n'
                                                         'standard deviation after correction: {3:.4f}V\n'
                                                         'maximum deviation after correction: {4:.4f}V.'
                                  .format(popt[2] * 1000, popt[0] * 1000, popt[1] * 1000, self.standard_v_dev[track], self.max_v_dev[track]))
                    plt.xlabel('step number')
                    plt.ylabel('scan number')
                    plt.colorbar()
                    plt.savefig('xml_rebinned/DBEC_{}_track{}.png'.format(self.runNo, track))
                    plt.close()
                    plt.clf()

                    self.fit_success.append(True)

                except Warning as w:
                    self.fit_success.append(w)
                except Exception as e:
                    self.standard_v_dev.append(-1)
                    self.volt_correct.append((1, 1))
                    self.fit_success.append(e)

    def export_read_results(self):
            # save to numpy data format .npy
            #np.save('xml_rebinned/DBEC_' + self.runNo, self.data_array)

            # save other information to dict:
            # additional info that may be interesting:
            with open('xml_rebinned/run_pars_summary.txt', 'a+') as sf:  # open summary file in append mode and create if it doesn't exist
                for track in range(self.nrOfTracks):
                    start_time_str = self.scan_date[track][0].strftime('%Y-%m-%d_%H:%M:%S')
                    stop_time_str = self.scan_date[track][1].strftime('%Y-%m-%d_%H:%M:%S')
                    sf.write(
                        '{}\ttrack: {}\tnOfScans: {}\tnOfSteps: {}\tstartVolt: {}\tstepSizeVolt: {}\tnOfTimeBins: {}\tnOfScalers: {}\tstart_time: {}\tstop_time: {}\t dacOffsetV: {:.3e}\t dacStepSlopeV: {:.3e}\t standDevVolt: {:.3e}\t maxVoltDev: {:.3e}\tcritDACerror%: {:.1f}\n' \
                        .format(self.runNo, track, self.nrOfScansCompleted, self.nrOfSteps, self.dac_start_set_V[track],
                                self.dac_stepsize_set_V[track], self.nrOfTimeBins, self.nrOfScalers,
                                start_time_str, stop_time_str, self.volt_correct[track][0], self.volt_correct[track][1],
                                self.standard_v_dev[track], self.max_v_dev[track], 100 * self.critical_dac_missmatch))
                    if self.fit_success[track] is not True:
                        sf.write('problem while fitting: ' + str(self.fit_success[track]) + '\n')

    ''' conversion to xml files '''

    def to_xml(self):
        #print("...now processing " + str(file), end=' ', flush=True)
        self.create_data_tuples()
        self.create_voltage_projection()
        self.extract_excel_info()
        self.extract_time_projection()
        self.prepare_xml_dictionary()
        self.write_to_xml()

    def create_data_tuples(self):
        # Create ScanData Array. Ultimately could use TILDA creat_default_scaler_array_from_scandict function in Formating.py maybe.
        track_data_list = []
        for track in range(self.nrOfTracks):
            data_tuples_list = []
            for index, cts in np.ndenumerate(self.data_array[track]):
                # Loop over data array and extract index + cts and combine them to a tuple.
                data_point_tuple = index + (int(cts),)
                data_tuples_list.append(data_point_tuple) # append tuple to list
            dt = [('sc', 'u2'), ('step', 'u4'), ('time', 'u4'), ('cts', 'u4')]  # data type for npy array
            track_data_list.append(np.array(data_tuples_list, dtype=dt))  # convert list to npy array with given data format
        self.data_tuples_arr = np.stack(track_data_list)

    def create_voltage_projection(self):
        # Create voltage projection
        track_data_list = []
        for track in range(self.nrOfTracks):
            voltage_projections = []
            for scaler in range(self.nrOfScalers):
                proj_list = self.data_array[track].sum(axis=2)[scaler].tolist()
                int_proj = [int(i) for i in proj_list]  # Must be an array of integers!
                voltage_projections.append(int_proj)
            track_data_list.append(np.array(voltage_projections))  # make array of arrays from list of arrays.
        self.voltage_projection_arr = np.stack(track_data_list)

    def extract_excel_info(self):
        ########################################
        # Extract information from excel sheet #
        ########################################
        try:
            exelcuter = BECOLAExelcuter(self.path_to_excel, self.runNo)
            excel_list = exelcuter.run(returnlen=15)
            excel_run_date, excel_runNo, self.isotope, self.laserFreq, excel_laser_backg, excelMode, excel_notes, \
                excel_nOfScans, excel_dacStartVolt, excel_dacEndVolt, excel_fill_time_ms, excel_release_time_ms, \
                excel_nOfSteps, excel_dacStepSizeVolt, self.seqs = excel_list
            if excel_nOfSteps is None:
                excel_nOfSteps = 0
            if excel_nOfScans is None:
                excel_nOfScans = 0
            if self.laser_unit == 'THz':
                self.laserFreq = self.laserFreq /299792458*1E10  # THz/(speedOfLight[m/s]/10^7)
            if 'DC' in excelMode or 'dc' in excelMode or 'Dc' in excelMode:
                self.is_dc_data = True
        except Exception as e:
            self.excel_extraction_failed = True
            with open('xml_rebinned/bad_runs.txt', 'a') as bfile:
                bfile.write('Excel extraction failed for run {} with following exception: {}\n'.format(self.runNo, e))
            excel_nOfSteps = 0
            excel_nOfScans = 0
            self.seqs = 0  # This is only in the excel files!! Will go into xml like this if it's not changed manually

        ######################################
        # compare with information from file #
        ######################################
        missmatch_str = ''
        if excel_nOfSteps != self.nrOfSteps:
            # extracted data is obviously bad. Do not use it.
            missmatch_str += 'nrOfSteps disagree. Excel says {}, from file extracted {}. '.format(excel_nOfSteps, self.nrOfSteps)
        if excel_nOfScans != self.nrOfScans:
            missmatch_str += 'nOfScans disagree! Excel says {}, from file extracted {}. '.format(excel_nOfScans, self.nrOfScans)

        for track in range(self.nrOfTracks):
            # Correct dacStartVolt by offset from read voltage
            self.dacStartVolt.append(self.dac_start_set_V[track] + self.volt_correct[track][0])
            # Correct dacStepSizeVolt by slope of voltage deviation
            self.dacStepSizeVolt.append(self.dac_stepsize_set_V[track] + self.volt_correct[track][1])
        # Check for critical DAC errors
        if self.critical_dac_missmatch >= 0.005:
            missmatch_str += 'critical DAC missmatch: {}%!'.format(100*self.critical_dac_missmatch)
        if missmatch_str != '':
            with open('xml_rebinned/bad_runs.txt', 'a') as bfile:
                    bfile.write('{}\t'.format(self.runNo)+missmatch_str+'\n')

    def extract_time_projection(self):
        ##########################################
        # fit time projection for software gates #
        ##########################################
        for track in range(self.nrOfTracks):
            this_track_gates = []
            dac_start = float(self.dacStartVolt[track])
            dac_step = float(self.dacStepSizeVolt[track])
            dac_stop = dac_start + dac_step*int(self.nrOfSteps)
            v_min = min(dac_start, dac_stop)
            v_max = max(dac_start, dac_stop)
            for scaler in range(self.nrOfScalers):
                if self.is_dc_data:
                    scaler_standard_gates = [v_min, v_max, 0, 10.24]  # time gates are in µs and we for simplicity defined 1bin(BECOLA)=10ns
                    this_track_gates.append(scaler_standard_gates)
                else:
                    # fitpars, successful = self.fit_time_projection_simplex(scaler)
                    fitpars, successful = self.fit_time_projection(track, scaler)
                    if successful:
                        cts_max, FWHM, center, offset = fitpars
                        upper_gate = int(center + abs(FWHM))
                        lower_gate = int(center - abs(FWHM))
                        scaler_softw_gate = [v_min, v_max, lower_gate/100, upper_gate/100]
                        this_track_gates.append(scaler_softw_gate)
                    else:
                        scaler_standard_gates = [v_min, v_max, 5.0, 5.8]
                        this_track_gates.append(scaler_standard_gates)
            self.softwGates.append(this_track_gates)

    def prepare_xml_dictionary(self):
        # write start and stop times
        file_start_time = self.scan_date[0][0]
        file_start_time_str = file_start_time.strftime('%Y-%m-%d %H:%M:%S')
        file_end_time = self.scan_date[-1][0]

        timediff = file_start_time - file_end_time
        midofruntime = file_start_time + timediff/2
        midofruntime_str = midofruntime.strftime('%Y-%m-%d %H:%M:%S')

        ###################################
        #Prepare dicts for writing to XML #
        ###################################
        #TODO: Could add type BECOLA_trs to XML-Importer and use that instead of trs
        header_dict = {'type': 'trs',
                       'isotope': self.isotope,
                       'isotopeStartTime': file_start_time_str,
                       'accVolt': self.accVolt_dc if self.is_dc_data else self.accVolt,
                       'laserFreq': self.laserFreq,  # XMLImporter expects wavenumber
                       'nOfTracks': self.nrOfTracks,
                       'version': 99.0}

        tracks_dict = {}
        for tracknum in range(self.nrOfTracks):
            trackname = 'track{}'.format(tracknum)
            # track times:
            track_start_time_str = self.scan_date[tracknum][0].strftime('%Y-%m-%d %H:%M:%S')
            track_end_time_str = self.scan_date[tracknum][1].strftime('%Y-%m-%d %H:%M:%S')
            # info for track in header
            track_dict_header = {'trigger': {},# Need a trigger dict!
                                'activePmtList': list(range(self.nrOfScalers)),  # Must be in form [0,1,2]
                                'colDirTrue': True,
                                'dacStartRegister18Bit': 0,
                                'dacStartVoltage': self.dacStartVolt[tracknum],
                                'dacStepSize18Bit': None,  # old format xml importer checks whether val or None
                                'dacStepsizeVoltage': self.dacStepSizeVolt[tracknum],
                                'dacStopRegister18Bit': self.nrOfSteps-1,  # not real but should do the trick
                                'dacStopVoltage': float(self.dacStartVolt[tracknum])+(float(self.dacStepSizeVolt[tracknum])*int(self.nrOfSteps-1)),  # nOfSteps-1 bc startVolt is the first step
                                'invertScan': False,
                                'nOfBins': self.nrOfTimeBins,
                                'nOfBunches': self.seqs,  # at BECOLA this corresponds to number of Sequences (Seqs in excel)
                                'nOfCompletedSteps': float(int(self.nrOfSteps)*int(self.nrOfScans)),
                                'nOfScans': self.nrOfScans,
                                'nOfSteps': self.nrOfSteps,
                                'postAccOffsetVolt': 0,
                                'postAccOffsetVoltControl': 0,
                                'SoftBinWidth_us': 1024,  #shrink later!
                                'softwGates': self.softwGates[tracknum],  # For each Scaler: [DAC_Start_Volt, DAC_Stop_Volt, scaler_delay, softw_Gate_width]
                                'workingTime': [track_start_time_str, track_end_time_str],
                                'waitAfterReset1us': 0,  #looks like I need those for the importer
                                'waitForKepco1us': 0  #looks like I need this too
                                }
            track_dict_data = {'scalerArray_explanation': 'time resolved data. List of tuples, each tuple consists of: (scaler_number, line_voltage_step_number, time_stamp, number_of_counts), datatype: np.int32',
                                'scalerArray': self.data_tuples_arr[tracknum]}
            track_dict_projections = {'voltage_projection_explanation': 'voltage_projection of the time resolved data. List of Lists, each list represents the counts of one scaler as listed in activePmtList.Dimensions are: (len(activePmtList), nOfSteps), datatype: np.int32',
                                'voltage_projection': self.voltage_projection_arr[tracknum]}
            tracks_dict[trackname] = {'header': track_dict_header,
                                      'data': track_dict_data,
                                      'projections': track_dict_projections}

        # Combine to xml_dict
        self.xml_dict = {'header': header_dict,
                         'tracks': tracks_dict
                         }

    def write_to_xml(self):
        ################
        # Write to XML #
        ################
        #if not self.excel_extraction_failed:  # actually that is not a big problem in this newer Version...
        xml_name = 'xml_rebinned/BECOLA_'+str(self.runNo)+'.xml'
        writeXMLfromDict(self.xml_dict, xml_name, 'BecolaData')

        ############################
        # Optional: Visualize Data #
        ############################
        # for track in range(self.nrOfTracks):
        #     self.visualizeData(track)

    def visualizeData(self, track):
        """
        Generate graphical output of the data. This can be used to check that everything was imported correctly.
        As is will plot a timeresolved graph bins vs voltage and projections to time and voltage axis.
        """
        data_arr = self.data_array[track]

        print('visualizing Data now')
        # Get sizes of arrays
        scal_data_size, x_data_size, t_data_size = np.shape(data_arr)

        # create mesh for time-resolved plot
        X,Y = meshgrid(np.arange(x_data_size), np.arange(t_data_size))
        # cts data is stored in data_array. Either per scaler [ScalerNo, Y, X]
        # or sum for all scalers: sum(self.data_array)[Y,X]
        Z = data_arr.sum(axis=0)[X, Y]
        # create timeresolved plot
        im = imshow(Z, cmap=cm.RdBu, interpolation='none', aspect='auto')
        colorbar(im)  # create plot legend
        show()

        # create axis-projections
        # either choose one scaler ([0],[1],[2],...) or sum all scalers (.sum(axis=0))
        # Voltage projection
        plot(np.arange(x_data_size), data_arr.sum(axis=2)[0], '.')  # x_step_projection_sc1
        plot(np.arange(x_data_size), data_arr.sum(axis=2)[1], '-')  # x_step_projection_sc2
        plot(np.arange(x_data_size), data_arr.sum(axis=2)[2], '--')  # x_step_projection_sc3
        show()
        # time projection
        plot(np.arange(t_data_size), data_arr.sum(axis=1).sum(axis=0), '.')  # y_time_projection_all
        plot(np.arange(t_data_size), data_arr.sum(axis=1)[0], '-')  # y_time_projection_sc1
        plot(np.arange(t_data_size), data_arr.sum(axis=1)[1], '-')  # y_time_projection_Sc2
        plot(np.arange(t_data_size), data_arr.sum(axis=1)[2], '-')  # y_time_projection_Sc3
        show()
        # fit for time projection
        t_bins_x = np.arange(t_data_size)
        t_cts_y = data_arr.sum(axis=1)[1]
        t_errs_y = np.sqrt(t_cts_y)
        initial_guess = np.array([10*t_cts_y[0], len(t_bins_x)/100, 550, t_cts_y[0]])

        # fit time projection and display
        #pars = self.fitTimePeak(t_bins_x, t_cts_y, t_errs_y, initial_guess)
        #print(pars[0])
        pars_simplex = simplex(self.func_simplex, initial_guess, args=(t_bins_x, t_cts_y, t_errs_y), full_output=0)
        #print(pars_simplex)
        plot(np.arange(t_data_size), data_arr.sum(axis=1)[1])  # y_time_projection_sc2
        plot(np.arange(t_data_size), self.func(np.arange(t_data_size), *pars_simplex), '-')  # y_time_projection_sc2_fit
        show()

    def fit_time_projection_simplex(self, track, scaler):
        data_arr = self.data_array[track]
        # Get sizes of arrays
        scal_data_size, x_data_size, t_data_size = np.shape(data_arr)
        # fit for time projection
        t_bins_x = np.arange(t_data_size)
        t_cts_y = data_arr.sum(axis=1)[scaler]
        t_errs_y = np.sqrt(t_cts_y)
        initial_guess = np.array([10*t_cts_y[0], len(t_bins_x)/100, 550, t_cts_y[0]])

        try:
            # curve_fit fitting
            #pars = self.fitTimePeak(t_bins_x, t_cts_y, t_errs_y, initial_guess)[0]
            # simplex fitting
            pars, fopt, iterations, funcalls, warnflag = simplex(self.func_simplex, initial_guess,
                                                                 args=(t_bins_x, t_cts_y, t_errs_y),
                                                                 full_output=1, disp=False)
            success = False
            if warnflag is 0:
                # maxiterations has not been exceeded
                success = True
            elif warnflag is 1:
                with open('xml_rebinned/bad_runs.txt', 'a') as bfile:
                    bfile.write('TOF-fitting exceeded max number of function evaluations for run {} scaler {}\n'
                                .format(self.runNo, scaler))
            elif warnflag is 2:
                with open('xml_rebinned/bad_runs.txt', 'a') as bfile:
                    bfile.write('TOF-fitting exceeded max number of iterations for run {} scaler {}\n'
                                .format(self.runNo, scaler))
        except Exception as e:
            with open('xml_rebinned/bad_runs.txt', 'a') as bfile:
                bfile.write('TOF-fitting failed for run {} scaler {} with following exception: {}\n'
                            .format(self.runNo, e, scaler))
        except Warning as w:
            pass
        return pars, success

    def func_simplex(self, params, X, Y, Err):
        # extract current values of fit parameters from input array
        c = params[0]  # hight factor
        w = params[1]  # FWHM
        t0 = params[2]  # max position
        o = params[3]  # offset

        #compute chi-square
        chi2 = 0.0
        for n in range(len(X)):
            x=X[n]
            # lorentz function
            # for these functions could refer to pollifits physics class...
            y = o + c*(np.power(w/2,2))/(np.power(x-t0,2)+np.power(w/2,2))
            # lorentz function (old)
            #y = o + c*(w*w/(np.power(x-t0,2)+np.power(w, 2)))

            chi2 = chi2 + (Y[n]-y)*(Y[n]-y)/(Err[n]*Err[n])
        return chi2

    def fit_time_projection(self, track, scaler):
        data_arr = self.data_array[track]
        try:
            # Get sizes of arrays
            scal_data_size, x_data_size, t_data_size = np.shape(data_arr)
            # fit for time projection
            t_bins_x = np.arange(t_data_size)
            t_cts_y = data_arr.sum(axis=1)[scaler]

            # estimates:: amplitude: sigma*sqrt(2pi)*(max_y-min_y), sigma=10, center:position of max_y, offset: min_y
            start_pars = np.array([10*2.51*(max(t_cts_y)-min(t_cts_y)), 10, np.where(t_cts_y == max(t_cts_y))[0], min(t_cts_y)])
            #print(start_pars)
            ampl, sigma, center, offset = curve_fit(self.fitfunc, t_bins_x, t_cts_y, start_pars)[0]
            #print(ampl, sigma, center, offset)
            success = True
        except Exception as e:
            ampl, sigma, center, offset = 1, 1, 1, 1
            success = False
            with open('xml_rebinned/bad_runs.txt', 'a') as bfile:
                bfile.write('TOF-fitting failed for run {} scaler {} with following exception: {}\n'
                            .format(self.runNo, scaler, e))
        return (ampl/sigma/2.51+offset, 2.355*sigma, center, offset), success

    def fitfunc(self, t, a, s, t0, o):
        """
        fitfunction for time projection
        t: time
        t0: mid-tof
        a: cts_max
        s: sigma
        o: offset
        """
        # Gauss function
        return o + a * 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*np.power((t-t0)/s, 2))

    def func(self, t,c,w,t0, o):
        """
        t: time
        t0: mid-tof
        a: cts_max
        w: width
        o: offset
        """
        # Gauss
        # return o + c*(np.power(w,2)/4)/(np.power(t-t0,2)+np.power(w,2)/4)
        # Lorentz
        return o + c*(w*w/(np.power(t-t0,2)+np.power(w, 2)))

    def fitTimePeak(self, t_bins_x, t_cts_y, t_errs_y, initial_guess):
        """
        Function to fit a gauss curve onto the time projection of a scaler
        returns a suggestion for gates: mid-tof, with of time window
        """
        pars = ()
        try:
            pars = optimization.curve_fit(self.func, t_bins_x, t_cts_y, initial_guess, t_errs_y)
        except Exception as e:
            with open('xml_rebinned/bad_runs.txt', 'a') as bfile:
                bfile.write('TOF-fitting failed for run {} with following exception: {}\n'.format(self.runNo, e))
        return pars


if __name__ == '__main__':
    # disable warning messages
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    directory = os.getcwd()

    print('Checking for output directory...')
    if not os.path.exists('xml_rebinned'):
        os.makedirs('xml_rebinned')
        print('Output folder created.')
    else:
        print('Output folder already exists.')

    print('Importing files. Please wait...')

    files_done = 0
    total_files = len(os.listdir(directory))
    percentage = 0

    # BECOLAImporter('DBEC_6502.asc')

    for file in os.listdir(directory):  # crawl through the folder
        if os.path.isfile(file) and '.asc' in file:  # Pick only .asc files
            print("\r"+'{:.0f}%'.format(percentage), end='', flush=True)
            imper = BECOLAImporter(file)  # Convert the .asc file to numpy collection
            imper = None  # make sure there's nothing left?
            files_done += 1
            percentage = files_done/total_files*100
        else:
            total_files -= 1
            percentage = files_done/total_files*100
            print("\r"+'{:.0f}%'.format(percentage), end='', flush=True)
    print("\r"+'{:.0f}%'.format(percentage), end='', flush=True)
    print("...BECOLA Importer done reading all files in folder", end='')