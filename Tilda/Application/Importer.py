"""
Created on 2023-03-16

@author: Patrick Mueller

Import the config modules.
"""

import os
from importlib import import_module

import Tilda.Application.Config as Cfg

try:
    SQLConfig = import_module(os.path.join(Cfg.config_dir, 'Driver', 'SQLStream', 'SQLConfig.py'))
except (FileNotFoundError, ImportError):
    import Tilda.Driver.SQLStream.SQLDraftConfig as SQLConfig

try:
    InfluxConfig = import_module(os.path.join(Cfg.config_dir, 'Driver', 'SQLStream', 'InfluxConfig.py'))
except (FileNotFoundError, ImportError):
    import Tilda.Driver.InfluxStream.InfluxConfigDraft as InfluxConfig

try:
    TritonConfig = import_module(os.path.join(Cfg.config_dir, 'Driver', 'TritonListener', 'TritonConfig.py'))
except (FileNotFoundError, ImportError):
    import Tilda.Driver.TritonListener.TritonDraftConfig as TritonConfig

try:
    DAC_Calibration = import_module(os.path.join(Cfg.config_dir, 'Service', 'VoltageConversions', 'DAC_Calibration.py'))
except (FileNotFoundError, ImportError):
    import Tilda.Service.VoltageConversions.DacRegisterToVoltageFit as DAC_Calibration
