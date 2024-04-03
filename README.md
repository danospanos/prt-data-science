# Electricity price forecasts
This project includes a notebook data-model-exploration.ipynb and a Python script (train_model.py) for training a cross-validated Lasso and LGB model. The script outputs results as PNG plots and displays evaluation metrics in the terminal.

The work is summarized in the notebook.

## Prerequisites

 - Python 3.8.5
 - Pipenv (recommended for creating a virtual environment)
 
## Installation

 1) Clone this repository to your local machine.
 2) Setting up the virtual environment.
You can initialize using pipenv, in the project directory with:

> pipenv --python 3.8.5

> pipenv install -r requirements.txt

## Example Usage

In you virtual environment run:

> python3 train_model.py --n_ahead 1 --dayhour 12 --model LGBM

 - n_ahead (int): indicates the number of hours ahead. (1, ...)
 - dayhour (int): specifies the hour of the day for which the predictions are made. (0, ..., 23)
 - model (str): name of the model, either LGBM or Lasso.

## Output

 - PNG Plot: The script will save PNG plot in the project directory.
 - Terminal Output: Evaluation metrics and other outputs will be displayed in the terminal.
