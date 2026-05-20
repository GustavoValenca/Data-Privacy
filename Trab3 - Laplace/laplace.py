import pandas as pd
import numpy as np

def laplace_mechanism(
    data: pd.Series,
    epsilon: float,
    sensibility: float = 1.0
) -> pd.Series:

    scale = sensibility / epsilon

    noise = np.random.laplace(
        loc=0,
        scale=scale,
        size=len(data)
    )

    noised_data = data + noise

    return pd.Series(
        noised_data,
        index=data.index
    )