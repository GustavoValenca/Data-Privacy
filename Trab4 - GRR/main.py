import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from GRR import *
from metrics import *
from utils import *
from tqdm import tqdm


df = pd.read_csv('../Trab3 - Laplace/votacao_prefeito_CE_2024.csv', encoding='cp1252', sep=';')


APPLY_SAMPLE = False

df_fortaleza = df[df['NM_MUNICIPIO'] == 'FORTALEZA']
df_fortaleza_1st_term = df_fortaleza[df_fortaleza['NR_TURNO'] == 1]
df_fortaleza_1st_term_expanded = df_fortaleza_1st_term.loc[df_fortaleza_1st_term.index.repeat(df_fortaleza_1st_term["QT_VOTOS"])].reset_index(drop=True)
df_fortaleza_1st_term_expanded_filtered = df_fortaleza_1st_term_expanded[['NR_ZONA', 'NM_VOTAVEL', 'QT_VOTOS']]

df_fortaleza_1st_term_expanded_filtered.to_csv('votacao_prefeito_CE_2024_fortal_1st_term_expanded_filtered.csv')

if APPLY_SAMPLE:
    df_fortaleza_1st_term_expanded_filtered = df_fortaleza_1st_term_expanded_filtered.sample(100000)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!! SAMPLE APLICADO, RETIRAR POSTERIORMENTE !!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print()

###########################################################################################

real_candidate_counts = df_fortaleza_1st_term_expanded_filtered["NM_VOTAVEL"].value_counts()
print("### ANALYSIS BY CANDIDATES ###")

print("Real histogram candidates statistics:")
tabela_estatisticas = real_candidate_counts.describe()
tabela_estatisticas["sum"] = real_candidate_counts.sum()
print(tabela_estatisticas.round(2))

candidato_mais_votado = real_candidate_counts.index[0]
qnt_votos_mais_votado = real_candidate_counts.iat[0]

seg_candidato_mais_votado = real_candidate_counts.index[1]
gap_entre_top2_candidatos = qnt_votos_mais_votado - real_candidate_counts.iat[1]

print()
print("Candidato mais votado:", candidato_mais_votado)
print("Contagem de votos do candidato mais votado:", qnt_votos_mais_votado)
print("Segundo candidato mais votado:", seg_candidato_mais_votado)
print("Gap entre os dois candidatos mais votados:", gap_entre_top2_candidatos)

generate_histogram(real_candidate_counts, title = "Votos por candidato", save_path = "graficos/candidates_histogram.png")

###########################################################################################

real_zone_counts = df_fortaleza_1st_term_expanded_filtered["NR_ZONA"].value_counts()

print()
print()

print("### ANALYSIS BY ZONES ###")
print("Top 10 zonas com mais votos")
print(real_zone_counts.sort_values(ascending=False).head(10))

generate_histogram(real_zone_counts, title = "Votos por zona eleitoral", zones = True, save_path = "graficos/zones_histogram.png")

###########################################################################################

dict_experimentos = {}
epsilon_list = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

for epsilon in epsilon_list:
    postprocesssed_mae_list = []
    postprocesssed_is_winner_preserved_list = []
    postprocesssed_precision_list = []

    raw_mae_list = []
    raw_is_winner_preserved_list = []
    raw_precision_list = []

    print(f'Epsilon: {epsilon}')
    for j in tqdm(range(20), desc=f"Processing epsilon {epsilon}"):
        # GRR
        perturbed, domain = apply_grr(df_fortaleza_1st_term_expanded_filtered, "NM_VOTAVEL", epsilon)
        est_candidate_counts_no_postprocess = estimate_counts(perturbed, domain, epsilon)
        est_candidate_counts = post_process(est_candidate_counts_no_postprocess)

        # Candidates
        ### With Postprocessing
        mae_value = get_mae(real_candidate_counts, est_candidate_counts)
        is_winner_preserved_value = is_winner_preserved(pd.Series(real_candidate_counts), est_candidate_counts)
        
        ### Without Postprocessing
        mae_value_no_postprocess = get_mae(real_candidate_counts, est_candidate_counts_no_postprocess)
        is_winner_preserved_value_no_postprocess = is_winner_preserved(pd.Series(real_candidate_counts), est_candidate_counts_no_postprocess)

        # Zones
        perturbed_zona, domain_zona = apply_grr(df_fortaleza_1st_term_expanded_filtered, "NR_ZONA", epsilon)
        est_zone_counts_no_postprocess = estimate_counts(perturbed_zona, domain_zona, epsilon)
        est_zone_counts = post_process(est_zone_counts_no_postprocess)

        ### With Postprocessing
        precision = precision_at_k(real_zone_counts, est_zone_counts, k=10)

        ### Without Postprocessing
        precision_no_postprocess = precision_at_k(real_zone_counts, est_zone_counts_no_postprocess, k=10)


        # Postprocessed metrics
        postprocesssed_precision_list.append(precision)
        postprocesssed_mae_list.append(mae_value)
        postprocesssed_is_winner_preserved_list.append(is_winner_preserved_value)

        # Raw (without postprocessing) metrics
        raw_mae_list.append(mae_value_no_postprocess)
        raw_is_winner_preserved_list.append(is_winner_preserved_value_no_postprocess)
        raw_precision_list.append(precision_no_postprocess)


    dict_experimentos[epsilon] = {
        'mae_mean': np.mean(postprocesssed_mae_list),
        'precision_mean': np.mean(postprocesssed_precision_list),
        'winner_preserved_percentage': np.mean(postprocesssed_is_winner_preserved_list),

        'mae_mean_no_postprocess': np.mean(raw_mae_list),
        'precision_mean_no_postprocess': np.mean(raw_precision_list),
        'winner_preserved_percentage_no_postprocess': np.mean(raw_is_winner_preserved_list),

        'mae_list': postprocesssed_mae_list,
        'is_winner_preserved_list': postprocesssed_is_winner_preserved_list,
        'precision_list': postprocesssed_precision_list,

        'mae_list_no_postprocess': raw_mae_list,
        'is_winner_preserved_list_no_postprocess': raw_is_winner_preserved_list,
        'precision_list_no_postprocess': raw_precision_list
    }


pd.DataFrame(dict_experimentos).to_csv('metricas/experimentos_grr.csv', index=False)

plot_metric_by_epsilon(dict_experimentos, metric='mae_list', save_path="graficos/postprocessed_mae_metric.png")
plot_metric_by_epsilon(dict_experimentos, metric='precision_list', save_path="graficos/postprocessed_precision_metric.png")
plot_metric_by_epsilon(dict_experimentos, metric='is_winner_preserved_list', save_path="graficos/postprocessed_winner_preserved_metric.png")

plot_metric_by_epsilon(dict_experimentos, metric='mae_list_no_postprocess', save_path="graficos/raw_mae_metric.png")
plot_metric_by_epsilon(dict_experimentos, metric='precision_list_no_postprocess', save_path="graficos/raw_precision_metric.png")
plot_metric_by_epsilon(dict_experimentos, metric='is_winner_preserved_list_no_postprocess', save_path="graficos/raw_winner_preserved_metric.png")
