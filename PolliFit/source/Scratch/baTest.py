import Tools
seeD1 = True
col = False
pathD1 = 'C:\\Users\\pimgram\\IKP ownCloud\\Projekte\\ALIVE\\Auswertungen\\Barium\\Barium_Data_D1.sqlite'

lD1col = 303870539.68
laserD1col = 2 * lD1col
lD1acol = 303555802.99
laserD1acol = 2 * lD1acol
#lD1 = 303569472
#lD1 = 303580000 #um 132_Ba zu trennen
#lD1 = 303855080
#laserD1 = 2*lD1
as_freqD1 = False #False = x in Energie;

pathD2 = 'C:\\Users\\pimgram\\IKP ownCloud\\Projekte\\ALIVE\\Auswertungen\\Barium\\Barium_Data_D2.sqlite'
lD2acol = 328892500
laserD2acol = 2 * lD2acol
lD2col = 329214500
laserD2col = 2 * lD2col
as_freqD2 = False

# Tools.createDB(pathD1)
#Tools.createDB(pathD2)

if seeD1:
    #D1 Linie
     #Tools.isoPlot(pathD1, '130_Ba', linevar='BaRef',col=False, laserfreq=laserD1acol, show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '130_Ba', linevar='BaRef', col=True, laserfreq=laserD1col, show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '132_Ba', linevar='BaRef',col=False, laserfreq=laserD1acol, show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '132_Ba', linevar='BaRef', col=True, laserfreq=laserD1col, show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '134_Ba', linevar='BaRef',col=False, laserfreq=laserD1acol,show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '134_Ba', linevar='BaRef', col=True, laserfreq=laserD1col, show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '135_Ba', linevar='BaRef',col=False, laserfreq=laserD1acol,show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '135_Ba', linevar='BaRef', col=True, laserfreq=laserD1col, show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '136_Ba', linevar='BaRef',col=False, laserfreq=laserD1acol,show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '136_Ba', linevar='BaRef', col=True, laserfreq=laserD1col, show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '137_Ba', linevar='BaRef',col=False, laserfreq=laserD1acol,show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '137_Ba', linevar='BaRef', col=True, laserfreq=laserD1col, show=False, as_freq=as_freqD1)
     #Tools.isoPlot(pathD1, '138_Ba', linevar='BaRef', col=False, laserfreq=laserD1acol, show=False, as_freq=as_freqD1)
     Tools.isoPlot(pathD1, '138_Ba', linevar='BaRef', col=True, laserfreq=laserD1col, as_freq=as_freqD1)
else:
    #D2 Linie
    #Tools.isoPlot(pathD2, '130_Ba', linevar='BaD2', col=False, laserfreq=laserD2acol,show=False, as_freq=as_freqD2)
    Tools.isoPlot(pathD2, '130_Ba', linevar='BaD2', col=True, laserfreq=laserD2col, show=False, as_freq=as_freqD2)
    #Tools.isoPlot(pathD2, '132_Ba', linevar='BaD2', col=False, laserfreq=laserD2acol,show=False, as_freq=as_freqD2)
    Tools.isoPlot(pathD2, '132_Ba', linevar='BaD2', col=True, laserfreq=laserD2col, show=False, as_freq=as_freqD2)
    #Tools.isoPlot(pathD2, '134_Ba', linevar='BaD2', col=False, laserfreq=laserD2acol,show=False, as_freq=as_freqD2)
    Tools.isoPlot(pathD2, '134_Ba', linevar='BaD2', col=True, laserfreq=laserD2col, show=False, as_freq=as_freqD2)
    #Tools.isoPlot(pathD2, '135_Ba', linevar='BaD2', col=False, laserfreq=laserD2acol,show=False, as_freq=as_freqD2)
    Tools.isoPlot(pathD2, '135_Ba', linevar='BaD2', col=True, laserfreq=laserD2col, show=False, as_freq=as_freqD2)
    #Tools.isoPlot(pathD2, '136_Ba', linevar='BaD2', col=False, laserfreq=laserD2acol,show=False, as_freq=as_freqD2)
    Tools.isoPlot(pathD2, '136_Ba', linevar='BaD2', col=True, laserfreq=laserD2col, show=False, as_freq=as_freqD2)
    #Tools.isoPlot(pathD2, '137_Ba', linevar='BaD2', col=False, laserfreq=laserD2acol,show=False, as_freq=as_freqD2)
    Tools.isoPlot(pathD2, '137_Ba', linevar='BaD2', col=True, laserfreq=laserD2col, show=False, as_freq=as_freqD2)
    #Tools.isoPlot(pathD2, '138_Ba', linevar='BaD2', col=False, laserfreq=laserD2acol, as_freq=as_freqD2)
    Tools.isoPlot(pathD2, '138_Ba', linevar='BaD2', col=True, laserfreq=laserD2col, as_freq=as_freqD2)