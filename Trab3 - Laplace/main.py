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
plt.figure(figsize=(12, 6))
bars = plt.bar(votes_by_candidate.keys(), votes_by_candidate.values(), width=0.5)
plt.bar_label(bars)
plt.xticks(rotation=85) 
plt.tight_layout() 
plt.show()

dict_experiments = {}

epsilon_list = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

for epsilon in epsilon_list:
    mae_list = []
    tae_list = []
    biggest_bin_list = []
    bool_winner_preserved_list = []
    noised_gap_list = []

    for j in range(20):
        current_laplace_series = laplace_mechanism(pd.Series(votes_by_candidate), epsilon)

        mae = get_mae(pd.Series(votes_by_candidate), current_laplace_series)
        tae = get_tae(pd.Series(votes_by_candidate), current_laplace_series)
        biggest_bin = get_largest_bin_error(pd.Series(votes_by_candidate), current_laplace_series)
        bool_winner_preserved = is_winner_preserved(pd.Series(votes_by_candidate), current_laplace_series)
        noised_gap = gap_first_and_second_candidates(current_laplace_series)

        mae_list.append(mae.item())
        tae_list.append(tae.item())

        biggest_bin_list.append(biggest_bin.item())
        bool_winner_preserved_list.append(bool_winner_preserved)

        noised_gap_list.append(noised_gap.item())

    dict_experiments[epsilon] = {
        'mae_list': mae_list,
        'tae_list': tae_list,
        'biggest_bin_list': biggest_bin_list,
        'is_winner_preserved_list': bool_winner_preserved_list,
        'noised_gap_list' : noised_gap_list
    }

print(dict_experiments)