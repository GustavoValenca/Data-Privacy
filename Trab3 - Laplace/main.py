import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from laplace import laplace_mechanism
from utils import *

df = pd.read_csv('votacao_prefeito_CE_2024.csv', encoding='cp1252', sep=';')

# filtering data
df_fortaleza = df[df['NM_MUNICIPIO'] == 'FORTALEZA']

df_fortaleza_1st_term = df_fortaleza[df_fortaleza['NR_TURNO'] == 1]


grouped_candidates = df_fortaleza_1st_term.groupby('NM_VOTAVEL')

votes_by_candidate = {}
for name, group in grouped_candidates:
    if name in votes_by_candidate:
        votes_by_candidate[name] += group.QT_VOTOS.sum().item()
    else:
        votes_by_candidate[name] = group.QT_VOTOS.sum().item()

# amount of votes of each candidate
generate_histogram(votes_by_candidate, save_path="graficos/histograma_original.png")
original_series = pd.Series(votes_by_candidate).sort_values(ascending = False)

tabela_estatisticas = original_series.describe()
tabela_estatisticas["sum"] = original_series.sum()

print("Tabela com estatísticas do histograma real:")
print(tabela_estatisticas.round(2))

candidato_mais_votado = original_series.index[0]
qnt_votos_mais_votado = original_series.iat[0]

seg_candidato_mais_votado = original_series.index[1]
gap_entre_top2_candidatos = qnt_votos_mais_votado - original_series.iat[1]

print()
print("Candidato mais votado:", candidato_mais_votado)
print("Contagem de votos do candidato mais votado:", qnt_votos_mais_votado)
print("Segundo candidato mais votado:", seg_candidato_mais_votado)
print("Gap entre os dois candidatos mais votados:", gap_entre_top2_candidatos)


dict_experiments = {}

epsilon_list = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

for epsilon in epsilon_list:
    mae_list = []
    tae_list = []
    biggest_bin_list = []
    bool_winner_preserved_list = []
    noised_gap_list = []
    ruidous_series_list = []

    for j in range(20):
        current_laplace_series = laplace_mechanism(original_series.copy(), epsilon)

        mae = get_mae(original_series.copy(), current_laplace_series)
        tae = get_tae(original_series.copy(), current_laplace_series)
        biggest_bin = get_largest_bin_error(original_series.copy(), current_laplace_series)
        bool_winner_preserved = is_winner_preserved(original_series.copy(), current_laplace_series)
        noised_gap = gap_first_and_second_candidates(current_laplace_series)

        ruidous_series_list.append(current_laplace_series)

        mae_list.append(mae.item())
        tae_list.append(tae.item())

        biggest_bin_list.append(biggest_bin.item())
        bool_winner_preserved_list.append(bool_winner_preserved)

        noised_gap_list.append(noised_gap.item())

    dict_experiments[epsilon] = {
        'mae_mean': np.mean(mae_list),
        'tae_mean': np.mean(tae_list),
        'biggest_bin_mean': np.mean(biggest_bin_list),
        'winner_preserved_percentage': np.mean(bool_winner_preserved_list),
        'is_winner_preserved_list': bool_winner_preserved_list,
        'mae_list': mae_list,
        'tae_list': tae_list,
        'biggest_bin_list': biggest_bin_list,
        'noised_gap_list' : noised_gap_list
    }


plot_mae_by_epsilon(dict_experiments, save_path="graficos/mae_metric.png")

plot_tae_by_epsilon(dict_experiments, save_path="graficos/tae_metric.png")

plot_winner_preservation_by_epsilon(dict_experiments, save_path="graficos/winner_preservation.png")