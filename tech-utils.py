import pandas as pd
import os


def saver(df: pd.DataFrame, name: str) -> None:
    """
    Saving function for your DataFrames

    Please Be Creative with those names, or i will be forced to add exception!

    :raises FileExistsError: It means that this file name was USED!

    :param df: DataFrame
    :type df: pd.DataFrame
    :param name: Name for .csv file - BE CREATIVE!!!
    :type name: str
    """

    if os.path.isfile(f"../kc459385_saves/{name}"):
        raise FileExistsError(f"You moron! {f"kc459385_saves/{name}.csv"} EXISTS!!!")

    df.to_csv(f"../kc459385_saves/{name}.csv")