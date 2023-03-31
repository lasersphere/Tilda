"""
Created on 2023-03-16

@author: Patrick Mueller

Import the config modules.
"""

import os
import importlib

import Tilda.Application.Config as Cfg

def loadSourceFile(path,name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    SQLConfig = loadSourceFile(os.path.join(Cfg.config_dir, 'Driver', 'SQLStream', 'SQLConfig.py'),"SQLConfig")
    #SQLConfig = import_module(os.path.join(Cfg.config_dir, 'Driver', 'SQLStream', 'SQLConfig.py'))
except (FileNotFoundError):
    import Tilda.Driver.SQLStream.SQLDraftConfig as SQLConfig

try:
    InfluxConfig = loadSourceFile(os.path.join(Cfg.config_dir, 'Driver', 'InfluxStream', 'InfluxConfig.py'),"InfluxConfig")
except (FileNotFoundError):
    import Tilda.Driver.InfluxStream.InfluxConfigDraft as InfluxConfig

try:
    TritonConfig = loadSourceFile(os.path.join(Cfg.config_dir, 'Driver', 'TritonListener', 'TritonConfig.py'),"TritonConfig")
    #TritonConfig = import_module(os.path.join(Cfg.config_dir, 'Driver', 'TritonListener', 'TritonConfig.py'))
except (FileNotFoundError):
    import Tilda.Driver.TritonListener.TritonDraftConfig as TritonConfig

try:
    DAC_Calibration = loadSourceFile(os.path.join(Cfg.config_dir, 'Service', 'VoltageConversions', 'DAC_Calibration.py'),"DAC_Calibration")
    #DAC_Calibration = import_module(os.path.join(Cfg.config_dir, 'Service', 'VoltageConversions', 'DAC_Calibration.py'))
except (FileNotFoundError):
    import Tilda.Service.VoltageConversions.DacRegisterToVoltageFit as DAC_Calibration
