# Python Robotics

A toolbox for robot dynamic simulation, analysis, control and planning.

## Installation ##

### Method 1: Using PIP ###

To install the package, use: 
```bash

pip install git+https://github.com/SherbyRobotics/pyro
```

It will install the files from the project. It will also check and install the required dependencies.

### Method 2: Clone repo and install ###

The following dependencies must be installed.

Required libraries:

* numpy
* scipy
* matplotlib
* pytest (only to run tests)

Clone the pyro repo from git, and install the pyro library to your python
environment:

```bash

git clone https://github.com/SherbyRobotics/pyro.git
cd pyro
python setup.py install
```

## Development ##

Use `python setup.py develop` to install in develop mode, which will
create a link (.egg-link file) to this code. The `pyro` module
will therefore be automatically updated as you edit the code in this
repository.

Run tests: `pytest -ra ./tests` (from repository root)

Run all examples: `pytest -ra ./examples` (from repository root)
