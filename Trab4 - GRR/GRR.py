import numpy as np
import pandas as pd

def apply_grr(df, column, epsilon):
    domain = np.array(df[column].unique())

    k = len(domain)

    exp_eps = np.exp(epsilon)
    p = exp_eps / (exp_eps + k - 1)

    value_to_idx = {v: i for i, v in enumerate(domain)}

    data_idx = df[column].map(value_to_idx).to_numpy()

    n = len(data_idx)

    keep = np.random.rand(n) < p

    replacement = np.random.randint(0, k - 1, size=n)

    replacement += (replacement >= data_idx)

    perturbed_idx = np.where(
        keep,
        data_idx,
        replacement
    )

    perturbed = domain[perturbed_idx]

    return perturbed, domain


def estimate_counts(perturbed, domain, epsilon):
    n = len(perturbed)
    k = len(domain)

    exp_eps = np.exp(epsilon)

    p = exp_eps / (exp_eps + k - 1)
    q = 1 / (exp_eps + k - 1)

    value_to_idx = {v: i for i, v in enumerate(domain)}

    perturbed_idx = np.array(
        [value_to_idx[v] for v in perturbed],
        dtype=np.int32
    )

    counts = np.bincount(
        perturbed_idx,
        minlength=k
    )

    lambda_j = counts / n

    phi_hat = (lambda_j - q) / (p - q)

    est_counts = phi_hat * n

    return pd.Series(
        est_counts,
        index=domain
    )


def post_process(counts):
    counts = counts.copy()
    counts[counts < 0] = 0
    return counts