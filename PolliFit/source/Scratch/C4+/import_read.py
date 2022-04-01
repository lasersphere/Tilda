import numpy
import Analyzer
import matplotlib.pyplot as plt

# import
# file = "C:\\Users\\pimgram\\IKP ownCloud\\Projekte\\KOALA\\C4+\\Online-Auswertung\\2022-02-18\\900uW.txt"
#
# pmt01 = numpy.loadtxt(file, delimiter=',', skiprows=1, usecols=(1,))
# pmt01_d = numpy.loadtxt(file, delimiter=',', skiprows=1, usecols=(2,))
# pmt23 = numpy.loadtxt(file, delimiter=',', skiprows=1, usecols=(3,))
# pmt23_d = numpy.loadtxt(file, delimiter=',', skiprows=1, usecols=(4,))
#
# wAvg01, wError01, chi01 = Analyzer.weightedAverage(pmt01, pmt01_d)
# avgError01 = numpy.std(pmt01) / (len(pmt01))**0.5
# avg01 = numpy.mean(pmt01)
#
# wAvg23, wError23, chi23 = Analyzer.weightedAverage(pmt23, pmt23_d)
# avgError23 = numpy.std(pmt23) / (len(pmt23))**0.5
# avg23 = numpy.mean(pmt23)
#
# print("Weighted Avg01: ", wAvg01, " +- ", wError01)
# print("Mean01: ", avg01, ", Error of mean: ", avgError01)
#
# print("Weighted Avg23: ", wAvg23, " +- ", wError23)
# print("Mean23: ", avg23, ", Error of mean: ", avgError23)


#import zsm
file = "C:\\Users\\pimgram\\IKP ownCloud\\Projekte\\KOALA\\C4+\\Online-Auswertung\\2022-03-01\\33\\zsm.txt"

pmt01 = numpy.loadtxt(file, delimiter=',', skiprows=1, usecols=(1,))
pmt01_d = numpy.loadtxt(file, delimiter=',', skiprows=1, usecols=(2,))


wAvg01, wError01, chi01 = Analyzer.weightedAverage(pmt01, pmt01_d)
avgError01 = numpy.std(pmt01) / (len(pmt01))**0.5
avg01 = numpy.mean(pmt01)


print("Weighted Avg01: ", wAvg01, " +- ", wError01)
print("Mean01: ", avg01, ", Error of mean: ", avgError01)

# plt.plot(x_data_orig, y_data_orig)
# plt.show()