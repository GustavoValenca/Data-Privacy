import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def generate_histogram(votes_by_candidate, save_path = None, zones = False, title = "", show = False):
    sorted_items = sorted(
        votes_by_candidate.items(),
        key=lambda item: item[1],
        reverse=True
    )

    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    if zones:
        names = list(map(lambda x: f"Zona {x}", names))

    plt.figure(figsize=(16, 8))

    # Gráfico horizontal
    bars = plt.barh(names, values, height=0.5)

    # Labels nas barras
    plt.bar_label(bars)

    # Maior valor no topo
    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi = 200)

    if show:
        plt.show()
    else:
        plt.close()

def _annotate_points(x_values, y_values):
    """
    Adiciona o valor de x próximo de cada ponto.
    """
    for x, y in zip(x_values, y_values):
        plt.annotate(
            f"{round(y,2)}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

def _configure_log_x_axis(epsilons):
    """
    Configura o eixo X em escala logarítmica,
    mostrando grades verticais APENAS nos pontos plotados.
    """
    ax = plt.gca()

    # Escala log no eixo X
    ax.set_xscale("log")

    # Mostrar ticks apenas nos epsilons existentes
    ax.set_xticks(epsilons)
    ax.set_xticklabels([str(eps) for eps in epsilons])

    # Remover ticks menores automáticos do log scale
    ax.minorticks_off()

    # Grid apenas nos ticks principais
    ax.grid(True, which="major", linestyle="--", alpha=0.6)


def plot_mae_by_epsilon(results: dict, save_path = None):
    """
    Plota o erro absoluto médio por categoria em função de epsilon.
    """
    epsilons = sorted(results.keys())
    mae_values = [results[eps]["mae_mean"] for eps in epsilons]

    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, mae_values, marker="o")

    _configure_log_x_axis(epsilons)
    _annotate_points(epsilons, mae_values)

    plt.xlabel("ε (log scale)")
    plt.ylabel("Erro Absoluto Médio")
    plt.title("Erro Absoluto Médio por Categoria vs ε")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi = 200)
    plt.show()


def plot_tae_by_epsilon(results: dict, save_path = None):
    """
    Plota o erro absoluto total em função de epsilon.
    """
    epsilons = sorted(results.keys())
    tae_values = [results[eps]["tae_mean"] for eps in epsilons]

    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, tae_values, marker="o")

    _configure_log_x_axis(epsilons)
    _annotate_points(epsilons, tae_values)

    plt.xlabel("ε (log scale)")
    plt.ylabel("Erro Absoluto Total")
    plt.title("Erro Absoluto Total vs ε")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi = 200)
    plt.show()


def plot_winner_preservation_by_epsilon(results: dict, save_path = None):
    """
    Plota a probabilidade empírica de preservar o vencedor
    em função de epsilon.
    """
    epsilons = sorted(results.keys())

    preservation_values = [
        results[eps]["winner_preserved_percentage"]
        for eps in epsilons
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, preservation_values, marker="o")

    _configure_log_x_axis(epsilons)
    _annotate_points(epsilons, preservation_values)

    plt.xlabel("ε (log scale)")
    plt.ylabel("Probabilidade de Preservar o Vencedor (%)")
    plt.title("Preservação do Vencedor vs ε")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi = 200)
    plt.show()



def _mean_and_ci(values, confidence=0.95):
    """
    Retorna média e intervalo de confiança usando
    aproximação normal.
    """

    if isinstance(values, pd.DataFrame):
        values = values.to_numpy().ravel()
    elif isinstance(values, pd.Series):
        values = values.to_numpy()

    values = np.asarray(values, dtype=float).ravel()

    n = len(values)
    mean = np.mean(values)

    if n <= 1:
        return mean, mean, mean

    std = np.std(values, ddof=1)
    sem = std / np.sqrt(n)

    z_values = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }

    z = z_values.get(confidence, 1.96)

    margin = z * sem

    return mean, mean - margin, mean + margin


def plot_metric_by_epsilon(
    results: dict,
    metric: str,
    save_path=None,
    show=False,
    confidence=0.95,
):
    epsilons = sorted(results.keys())

    means = []
    lower_bounds = []
    upper_bounds = []
    has_ci = False

    for eps in epsilons:
        value = results[eps][metric]

        if np.isscalar(value):
            means.append(value)
            lower_bounds.append(value)
            upper_bounds.append(value)
        else:
            mean, lower, upper = _mean_and_ci(
                value,
                confidence=confidence,
            )
            means.append(mean)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            has_ci = True

    plt.figure(figsize=(8, 5))

    plt.plot(
        epsilons,
        means,
        marker="o",
        label="Mean"
    )

    if has_ci:
        plt.fill_between(
            epsilons,
            lower_bounds,
            upper_bounds,
            alpha=0.2,
            label=f"{int(confidence * 100)}% CI"
        )
        plt.legend()

    _configure_log_x_axis(epsilons)
    _annotate_points(epsilons, means)

    plt.xlabel("ε (log scale)")
    plt.ylabel(metric)
    plt.title(f"{metric} vs ε")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()