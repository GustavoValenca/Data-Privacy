import numpy as np
import pandas as pd

def get_mae(real_counts, est_counts):
    est_counts = est_counts.reindex(real_counts.index, fill_value=0)
    
    return np.mean(np.abs(real_counts - est_counts))

def precision_at_k(real_counts, est_counts, k=10):    
    # top-k reais
    topk_real = set(real_counts.sort_values(ascending=False).head(k).index)
    
    est_counts = est_counts.reindex(real_counts.index, fill_value=0)
    topk_est = set(est_counts.sort_values(ascending=False).head(k).index)
    
    intersection = len(topk_real & topk_est)
    
    return intersection / k

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