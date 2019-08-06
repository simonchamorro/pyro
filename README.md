# Python Robotics

A toolbox for robot dynamic simulation, analysis, control and planning.

## Installation ##

Required libraries:

* numpy
* scipy
* matplotlib
* pytest (only to run tests)

Run the following command to install the pyro library to your python
environment:

`python setup.py install`

## Development ##

Use `python setup.py develop` to install in develop mode, which will
create a link (.egg-link file) to this code. The `pyro` module
will therefore be automatically updated as you edit the code in this
repository.

Run tests: `pytest -ra ./tests` (from repository root)

Run all examples: `pytest -ra ./examples` (from repository root)