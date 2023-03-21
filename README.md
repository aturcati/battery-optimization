# Battery Optimization
This repository contains code examples to optimize energy trading from a battery.

## Repository Structure
- `battery_optimization` contains the libraries for data handling and for optimizing,
- `notebooks` contains example jupyter notebooks to illustrate the use of the libraries,
- `data` contains the raw and processed electricity price data.

## Requirements
The code is in python and the dependency management is done with `poetry`. Follow the installation steps
from https://python-poetry.org/docs/ to install it on your machine.
Add the following plugin to automatically load `.env` files:
```shell
$ poetry self add poetry-dotenv
```

## Installation
Clone the repository on your machine and install the dedicated environment with all the dependencies
```shell
$ git clone https://github.com/aturcati/battery-optimization.git
$ poetry install
```

## Run the code
To run the code one needs to activate the python virtual environment with `poetry shell` or `poetry run <command>`.
For example, to start a jupyterlab server and test the example notebooks:
```shell
$ poetry run jupyter-lab
```
