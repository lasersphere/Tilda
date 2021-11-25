# Data-Acquisition and Data-Analysis Software for Collinear Laser Spectroscopy
The data-acquisition software TILDA and the data-analysis tools of Pollifit together provide 10ns-timed experiment 
control and a framework to load, view and analyze the corresponding data from Collinear Laser Spectroscopy (CLS) 
experiments. The code is written and maintained by members of the LaserSpHERe group.
---
## Content
1. [General](#data-acquisition-and-data-analysis-software-for-collinear-laser-spectroscopy)
   * [License](#license)
   * [Requirements](#Requirements)
2. [TILDA](#tilda)
3. [Pollifit](#pollifit)

## License
For now, TILDA does not have a license. This is not good... Lets have a look at 
[EUPL 1.2](https://joinup.ec.europa.eu/collection/eupl/eupl-guidelines-faq-infographics) (used a lot at GSI), 
[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)
and [MIT License](https://opensource.org/licenses/mit-license.php).
More info on these and many more licenses can be found [here](https://spdx.org/licenses/).

## Requirements
TILDA and Pollifit are distributed via GIT.

The software is currently tested to be compatible with python versions 3.4 (out of support), 3.8 and 3.9 on both 32- and
64-bit architectures.
Using python 3.8 or 3.9 all required packages can be installed using *python -m pip install -r 
[requirements.txt](requirements.txt)*, preferably using a virtual environment (.venv). All packages for python 3.4 are
available on the ownCloud group server under "\Projekte\TRIGA\Triton\PythonPackages\full_installation".

LabView 2014 SP1 (32bit), LabView 2014 FPGA, Ni RIO 14.0.1 Driver, Xilinx Compilation tools 14.7 and FPGA C-APi 14.0 are
required for development of the FPGA code. Newer Versions of LabView may or may not work.

For now, Windows 10 is the only actively supported operating system. The python code should run on any other OS, but
FPGA-hardware support may be lacking.

## Documentation
The TILDA and Pollifit software project is documented in the OneNote 
[Tilda-Manual](https://espace.cern.ch/labbooks-lasersphere/_layouts/OneNote.aspx?id=%2Flabbooks-lasersphere%2FDarmstadt%2FTILDA-Manual).
This includes the legacy notes from the old lab book as well as newer User Guides, Programmer Guides, Documentation and 
Discussion. The manual is still work-in-process but can answer many questions already.

A comprehensive and coherent description of the software is also included in the 
[PhD Thesis](http://tuprints.ulb.tu-darmstadt.de/9286) 
of Simon Kaufmann. 

Finally, all python code should be properly annotated.

---
# TILDA
TILDA is a software for the control and data-acquisition of CLS experiments. It provides a complex user interface that 
gives control over settings for a measurement and also provides a display of the measurement itself in (almost) 
real-time.

Currently there are three measurement modes: 
* The "SimpleCounter" (SC) allows a continuous observation of detector count rates.
* The "ContinuousSequencer" (CS) builds on the SC and enables scanning measurements with (voltage-)step-resolution.
* The "TimeResolvedSequencer" (TRS) provides step- and time-resolution for scanning measurements. 

The measurements are performed on an FPGA card to allow high resolution timing 8including triggers) and data is 
constantly transferred to the python program where the raw-data is first saved, then processed in the analysis-pipeline
(as a copy) and finally displayed and saved again in the processed format. A second FPGA is set up as a 
pulse-pattern-generator (PPG) and enables the creation of advanced timing patterns.

Dummy devices are available for all three measurement modes and allow testing of the program without FPGA hardware.

---
# Pollifit
Pollifit provides the backbone for data- and file-handling as well as for data-analysis of CLS measurements. 
It is used within TILDA but also works as a stand-alone program. A SQLite database for each measurement campaign is 
used to manage files, isotopes, scan parameters, line shapes and results.

Pollifit provides an interface for various analysis-operations including
(but not limited to) fitting of single files, batch-fitting and the extraction of isotope shifts and nuclear charge 
radii.

The Pollifit code is also an excellent basis for advanced analysis scripts that go beyond the possibilities of the GUI.