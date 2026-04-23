"""
Classes for the exchange of data between programs on multiple computers.

The package contains classes for the creation of data structures, the
control of devices and a PyQt5-based interface. The default communica-
tion backend is based on ZeroMQ, the development of other backends is
possible. For every distinct program, an Instance is required (see the
BaseInstance class in the Instance module). This object centralises
communication and is required for nearly all classes to function.
Because of this, for nearly every program using this framework, the
first step in the execution is the creation of a suitable Instance.

The functions get_default_instance and get_default_main_window can be
used to easily create programs that can start other components based on
configuration files.

The subpackage ZeroMQ contains all classes specific to the ZeroMQ imple-
mentation of the backend.

Modules:
Utility: Data structures and helper functions.
Processing: Queues and Loops.
Instance: Base class for backend implementations.
Device: Containers for groups of variables and execution of shared code.
Variable: Network-accessible data.
Connection: Remote access to variables.
Interface: Base classes for UI creation.
Initializer: Wrappers for creating variables as properties.
"""
# Hint for developers: To follow PEP 8, all docstrings and comments except
# TODOs and similarly temporary notes are to be limited to 72 symbols
# (disregarding indentation shared by the whole block).
# Due to developer preference, code is not limited in width, though with
# the exception of function definitions it is usually limited to a width
# of 120 symbols.
# TODO limit comments to 72 symbols? (docstrings are done)
#  make hints w.r.t. sub-objects in docstrings shorter
#  move things to sub-object docstring (requires making properties explicit?)

__version__ = '1.0.7'

from .Devices import ActiveDevice as Device
from .Encoder import JSONEncoder as JSON, NumpyArrayEncoder as npArray, Datatype
from .FactoryFunctions import get_default_instance as Instance, get_default_window as Window, create_variable_or_descriptor as Variable, create_trigger_or_descriptor as Trigger, read_configuration