import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from GRR import *
from metrics import *

df = pd.read_csv('Trab3 - Laplace/votacao_prefeito_CE_2024.csv', encoding='cp1252', sep=';')

df_fortaleza = df[df['NM_MUNICIPIO'] == 'FORTALEZA']
df_fortaleza_1st_term = df_fortaleza[df_fortaleza['NR_TURNO'] == 1]
df_fortaleza_1st_term_expanded = df_fortaleza_1st_term.loc[df_fortaleza_1st_term.index.repeat(df_fortaleza_1st_term["QT_VOTOS"])].reset_index(drop=True)
df_fortaleza_1st_term_expanded_filtered = df_fortaleza_1st_term_expanded[['NR_ZONA', 'NM_VOTAVEL', 'QT_VOTOS']]

df_fortaleza_1st_term_expanded_filtered.to_csv('votacao_prefeito_CE_2024_fortal_1st_term_expanded_filtered.csv')

dict_experimentos = {}

epsilon_list = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

real_candidate_counts = df_fortaleza_1st_term_expanded_filtered["NM_VOTAVEL"].value_counts()
real_zone_counts = df_fortaleza_1st_term_expanded_filtered["NR_ZONA"].value_counts()

for epsilon in epsilon_list:
    mae_list = []
    biggest_bin_list = []
    is_winner_preserved_list = []
    noised_gap_list = []
    precision_list = []

    print(f'Epsilon: {epsilon}')
    for j in range(20):
        # GRR
        perturbed, domain = apply_grr(df_fortaleza_1st_term_expanded_filtered, "NM_VOTAVEL", epsilon)
        est_candidate_counts = estimate_counts(perturbed, domain, epsilon)
        est_candidate_counts = post_process(est_candidate_counts)

        # MAE
        mae_value = get_mae(real_candidate_counts, est_candidate_counts)

        perturbed_zona, domain_zona = apply_grr(df_fortaleza_1st_term_expanded_filtered, "NR_ZONA", epsilon)
        est_zone_counts = estimate_counts(perturbed_zona, domain_zona, epsilon)
        est_zone_counts = post_process(est_zone_counts)

        precision = precision_at_k(real_zone_counts, est_zone_counts, k=10)

        biggest_bin = get_largest_bin_error(pd.Series(real_candidate_counts), est_candidate_counts)
        is_winner_preserved = is_winner_preserved(pd.Series(real_candidate_counts), est_candidate_counts)
        noised_gap = gap_first_and_second_candidates(est_candidate_counts)

        precision_list.append(precision)
        mae_list.append(mae_value)
        biggest_bin_list.append(biggest_bin.item())
        is_winner_preserved_list.append(is_winner_preserved)
        noised_gap_list.append(noised_gap.item())

        print(precision)
        print(mae_value)

    dict_experimentos[epsilon] = {
        'mae_list': mae_list,
        'biggest_bin_list': biggest_bin_list,
        'is_winner_preserved_list': is_winner_preserved_list,
        'noised_gap_list' : noised_gap_list,
        'precision_list': precision_list
    }
