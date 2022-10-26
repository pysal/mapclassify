from os.path import abspath, dirname

import pandas as pd


def load():
    """
    Load the data and return a DataSeries instance.

    """

    df = _get_data()

    return df["emp/sq km"]


def _get_data():
    filepath = dirname(abspath(__file__))
    filepath += "/calempdensity.csv"
    df = pd.read_csv(filepath)
    return df
