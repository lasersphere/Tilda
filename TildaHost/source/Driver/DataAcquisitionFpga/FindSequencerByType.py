import os


path = os.path.realpath(__file__)
for file in os.listdir(os.path.split(path)[0]):
    if 'Config' in file:
        print(os.path.split(path)[0] + file)