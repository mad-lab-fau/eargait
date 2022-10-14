[![PyPI](https://img.shields.io/pypi/v/eargait)](https://pypi.org/project/eargait/)
[![Documentation Status](https://readthedocs.org/projects/eargait/badge/?version=latest)](https://eargait.readthedocs.io/en/latest/?badge=latest)
[![Test and Lint](https://github.com/mad-lab-fau/eargait/actions/workflows/test-and-lint.yml/badge.svg?branch=main)](https://github.com/mad-lab-fau/eargait/actions/workflows/test-and-lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/eargait)

# Eargait - The Gait Analysis Package for Ear-Worn IMU Sensors !

*Eargait* provides a set of algorithms and functions to process IMU data recorded with ear-worn IMU sensors and to 
estimate characteristic gait parameters. 

analyze your IMU data recorded with hearing aid integrated IMU sensors.

<center> <img src=./docs/_static/logo/WalkingHearingAid.pdf height="200"/></center>

[![docs](https://img.shields.io/badge/docs-online-green.svg)](http://MadLab.mad-pages.informatik.uni-erlangen.de/qu94hoxi/eargait/README.html)
(Link for docs online is currently not working)
## Getting started

### Prerequisites
*Eargait* only supports Python 3.7 and newer.
First, install a compatible version of Python.

### Set up athe the virtual environment
We recommend installing the packages in a virtual environment (e.g. conda/Anaconda).
For more information regarding Anaconda, please visit [Anaconda.com](https://docs.anaconda.com/anaconda/install/index.html). <br />
If you want to install the packages directly on the local python version, directly go to [Install Packages](#install-packages)  <br />

If you are familiar with virtual environments you can ``also use any other type of virtual environment. 
Furthermore, you can also directly install the python packages on the local python version, however, we would not recommend doing so.

**In PyCharm** <br />
See [documentation](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html).

**Shell/Terminal** <br /> 
First, verify that you have a working conda installation. Open a terminal/shell and type
```
conda env list
```
If an error message similar to the one below is displayed, you probably do not have a working conda version installed. 
```
conda: command not found
```
In the shell/terminal:
```
conda create --no-default-packages -n gait_analysis python=3.8
```
*gait_analysis* is the name of the virtual environment. This environment can now also be included in PyCharm, 
as described See [here](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html) 
by using the existing environment option. <br /> 
To check, whether the virtual environment has been created successfully, run again:
```
conda env list
```
The environment *gait_analysis* should now be displayed.  <br /> 
Activate conda environment and install packages (see below).
 
```
conda activate gait_analysis
```

For more help: [Conda Documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)


### Install Packages
If you are using the conda environment, activate environment (in shell/terminal) (see above).
You can also install the packages directly on the local python version of the computer.

#### Update pip
```
pip install --upgrade pip 
```
#### *signialib*
Within the shell/terminal, go to the Signia directory by
```
cd my_local_path/signialib
```
*my_local_path* should be changed to the local path of the computer, where the Signia package is saved. <br /> 
Then install the package with:
```
pip install .
```

####  *eargait*
Change to the local directory of Eargait with:
```
cd my_local_path/eargait
```
Then install the package with:
```
pip install -e .
```

## Check successful installation

To check whether the installation was successful, run the following line directly after installing `eargait` in the same shell/terminal: 
```
python examples/check_installation/check_installation.py

```
Should return: `Installation was successful!`

## Poetry Environment

If you are using poetry and want add the package to you existing poetry environment, run: 
```
poetry add eargait
```

## Dev Setup
We are using poetry to manage dependencies and poethepoet to run and manage dev tasks. 

To set up the dev environment including the required dependencies for using eargait run the following commands:
```
git clone https://github.com/mad-lab-fau/eargait
cd eargait
poetry install
```
Afterwards you can start to develop and change things. 
If you want to run tests, format your code, build the docs, ..., 
you can run one of the following poethepoet commands

```
CONFIGURED TASKS
  format         
  lint           Lint all files with Prospector.
  check          Check all potential format and linting issues.
  test           Run Pytest with coverage.
  docs           Build the html docs using Sphinx.
  bump_version   
```
by calling
```
poetry run poe <command name>
```

## Contribution

The entire development is managed via [GitHub](https://github.com/mad-lab-fau/eargait).
If you run into any issues, want to discuss certain decisions, want to contribute features or feature requests, just 
reach out to us by [opening a new issue](https://github.com/mad-lab-fau/eargait/issues/new/choose).