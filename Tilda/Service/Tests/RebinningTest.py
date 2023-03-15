import Tilda.Service.Formating as Form
from Tilda.Service.Formating import create_default_scaler_array_from_scandict
from Tilda.Service.Formating import time_rebin_all_data_slow
from Tilda.Service.Formating import time_rebin_all_data
import Tilda.Service.Scan.draftScanParameters as DftScpars
import numpy as np
import timeit
scanpars = DftScpars.draftScanDict
tracks = ['track0']
sc_arr1 = [np.mgrid[0:100, 0:100]]
sc_arr = create_default_scaler_array_from_scandict(scanpars, 1)
# print(sc_arr1[0])
bins = 9
time_axis = Form.create_time_axis_from_scan_dict(scanpars, rebinning=False)


# def wrapper(func, *args, **kwargs):
#     def wrapped():
#         return func(*args, **kwargs)
#     return wrapped
#
# wrapped1 = wrapper(time_rebin_all_data, sc_arr, bins)
# wrapped2 = wrapper(time_rebin_all_data_slow, sc_arr, bins)
#
# print(timeit.timeit(wrapped1, number=100))
# print(timeit.timeit(wrapped2, number=100))

rebin = time_rebin_all_data(sc_arr, scanpars)
print(len(rebin[0][0][1]))
print(time_axis[0])
# rebin2 = time_rebin_all_data_slow(sc_arr, bins)

# for i, j in enumerate(rebin[0][1]):
#     # print('sc_arr1[0][1][i], rebin2[0][1][i], j')
#     print(sc_arr1[0][1][i], j)

# for i, j in enumerate(rebin[0][0]):
#     print('sc_arr1[0][0][i], rebin2[0][0][i], j')
#     print(sc_arr[0][0][i], rebin2[0][0][i], j)
