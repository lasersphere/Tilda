import os

dir = 'D:\Temp\Python_install\\'

filenames = os.listdir(dir)



string = 'conda install --use-local --offline '

for file in filenames:
    if file.endswith('.tar.bz2'):
        string = string + dir + file + '\t'

# print(string)
userinp = input('the following will be executed: \n' + string + '\n \n press y to proceed, n to exit:')
if userinp == 'y':
    os.system(string)