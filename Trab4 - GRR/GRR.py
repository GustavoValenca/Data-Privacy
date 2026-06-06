import numpy as np

def grr_perturb(value, domain, epsilon):
    k = len(domain)

    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    q = 1 / (np.exp(epsilon) + k - 1)

    if np.random.rand() < p:
        return value
    else:
        other_values = [v for v in domain if v != value]
        return np.random.choice(other_values)
    
def apply_grr(df, column, epsilon):
    domain = df[column].unique().tolist()

    perturbed = df[column].apply(lambda x: grr_perturb(x, domain, epsilon))

    return perturbed, domain

def estimate_counts(perturbed_series, domain, epsilon):
    n = len(perturbed_series)
    k = len(domain)

    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    q = 1 / (np.exp(epsilon) + k - 1)

    counts = perturbed_series.value_counts().reindex(domain, fill_value=0)

    lambda_j = counts / n

    phi_hat = (lambda_j - q) / (p - q)

    est_counts = phi_hat * n

    return est_counts

def post_process(counts):
    counts[counts < 0] = 0
    return counts