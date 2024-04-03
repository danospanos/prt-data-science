#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utils to load and format the CSV data
"""

# == Libraries
import pandas as pd


def format_df(df):
    """
    Format prt-legacy csv file to a data frame with datetime index
    :param df: Dataframe to format
    :return: formatted dataframe
    """

    dates = df.apply(lambda x: pd.to_datetime(
        f"{int(2000 + x[0])} {int(x[1])} {int(x[2])}",
        format="%Y %m %d"), axis=1)
    df.drop([0, 1, 2], axis=1, inplace=True)
    df.columns = list(range(df.shape[1]))
    df.set_index(dates, inplace=True)

    return df


def load_prt_legacy_file(file):
    """
    Load legacy file into dataframe
    """
    try:
        df = pd.read_csv(file, header=None, sep=r"\s+")
        df = format_df(df)
    except:
        raise IOError("File could not be loaded %s" % file)

    return df


def daily_to_hourly(df: pd.DataFrame):
    """
    Daily to hourly dataframe transformation
    :param df:
    :return:
    """
    # reindex df
    re_ind = pd.date_range(df.index[0], df.index[-1], freq="1d")
    df = df.reindex(re_ind)
    # flatten
    flat = pd.DataFrame(df.values.flatten())
    dates = pd.date_range(df.index[0], periods=len(flat), freq="1h")

    return flat.set_index(dates)
