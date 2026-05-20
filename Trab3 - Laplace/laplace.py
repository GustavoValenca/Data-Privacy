import pandas as pd
import numpy as np

def laplace_mechanism(
    data: pd.Series,
    epsilon: float,
    sensibility: float = 1.0
) -> pd.Series:

    scale = sensibility / epsilon

    u = np.random.uniform(
        low=-0.5,
        high=0.5,
        size=len(data)
    )

    noise = -scale * np.sign(u) * np.log(
        1 - 2 * np.abs(u)
    )

    noised_data = data + noise

    return pd.Series(
        noised_data,
        index=data.index
    )