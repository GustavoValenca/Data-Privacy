import pandas as pd
import numpy as np

def get_mae(
    real_data: pd.Series,
    noised_data: pd.Series
) -> float:

    error = np.abs(real_data - noised_data)

    return error.mean()


def get_tae(
    real_data: pd.Series,
    noised_data: pd.Series
) -> float:

    error = np.abs(real_data - noised_data)

    return error.sum()


def get_largest_bin_error(
    real_data: pd.Series,
    noised_data: pd.Series
) -> float:
    
    real_winner = real_data.idxmax()

    error = abs(
        real_data[real_winner] -
        noised_data[real_winner]
    )

    return error


def is_winner_preserved(
    real_data: pd.Series,
    noised_data: pd.Series
) -> bool:

    real_winner = real_data.idxmax()
    noised_winner = noised_data.idxmax()

    return real_winner == noised_winner

def gap_first_and_second_candidates(data: pd.Series) -> float:
    
    data = data.sort_values(ascending=False)

    first = data.iloc[0]
    second = data.iloc[1]

    gap = first - second

    return gap