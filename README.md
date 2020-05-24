# Python Robotics

A toolbox for robot dynamic simulation, analysis, control and planning.

## Installation ##

### Method 1: Using PIP ###

To install the package, use: 
```bash

pip install git+https://github.com/SherbyRobotics/pyro
```

It will install the files from the project. It will also check and install the required dependencies. Warning: don't use this option if you are using an Anacondo distribution.

### Method 2: Clone repo and install ###

The following dependencies must be installed.

Required libraries:

* numpy
* scipy
* matplotlib
* pytest (optionnal, only to run tests)

Clone the pyro repo from git, and install the pyro library to your python
environment:

```bash

git clone https://github.com/SherbyRobotics/pyro.git
cd pyro
python setup.py install
```
or to install in develop mode (having the `pyro` module automatically updated as you edit the code in this
repository) use:

```bash

git clone https://github.com/SherbyRobotics/pyro.git
cd pyro
python setup.py develop
```


### Method 3: Clone repo and add to python path ###

Another option for development is simply to clone the repo:
```bash
git clone https://github.com/SherbyRobotics/pyro.git
```
then add the pyro folder to the pythonpath variable of your environment.



