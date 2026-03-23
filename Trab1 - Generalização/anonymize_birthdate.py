import json
import os

import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def map_data_nascimento_nivel_0(x):
    """
    Função de mapeamento para o nível 0 de generalização da data nascimento. Dados originais.
    """
    return x

def map_data_nascimento_nivel_1(x):
    """
    Função de mapeamento para o nível 1 de generalização da data nascimento. Apenas ano e mês.
    """
    if type(x) != str: return x
    return "-".join(x.split("-")[:2])

def map_data_nascimento_nivel_2(x):
    """
    Função de mapeamento para o nível 2 de generalização da data nascimento. Apenas ano.
    """
    if type(x) != str: return x
    return x.split("-")[0]

def gera_json_nivel(data_nascimento_original, nivel):
    """
    Função que gera os json com mapeamento dos valores originais para os generalizados de acordo com o nível fornecido
    """
    data_nascimento = pd.to_datetime(data_nascimento_original, format="%Y-%m-%d", errors="coerce")
    datas_unicas_ordenadas = data_nascimento.drop_duplicates().sort_values().dt.strftime("%Y-%m-%d")
    map_json = {}

    for data in datas_unicas_ordenadas:
        if nivel == 0: map_json[data] = data
        if nivel == 1: map_json[data] = map_data_nascimento_nivel_1(data)
        if nivel == 2: map_json[data] = map_data_nascimento_nivel_2(data)

    return map_json

def gera_json_final(data_nascimento, output_path):
    json_final = dict()
    for i in range(3):
        json_final[f"nivel {i}"] = gera_json_nivel(data_nascimento, nivel=i)

    with open(os.path.join(output_path, f"anonymized_birthdate.json"), "w") as file:
        json.dump(json_final, file, indent=4)

    return json_final

def gera_histograma(coluna_modificada, nivel, output_path):
    """
    Função de geração do histograma de acordo com o nível de generalização aplicado na coluna de data nascimento.
    """
    if nivel == 0:
        coluna_modificada = pd.to_datetime(coluna_modificada, format="%Y-%m-%d", errors="coerce")
        coluna_filtrada = coluna_modificada[
            (coluna_modificada >= '1900-01-01') &
            (coluna_modificada <= '2026-12-31')
            ]
        n_bins = (coluna_filtrada.max() - coluna_filtrada.min()).days // 16
        coluna_filtrada.hist(bins=n_bins)

    if nivel == 1:
        coluna_modificada = pd.to_datetime(coluna_modificada, format="%Y-%m", errors="coerce")
        coluna_filtrada = coluna_modificada[
            (coluna_modificada >= '1900-01-01') &
            (coluna_modificada <= '2026-12-31')
            ]
        n_bins = (coluna_filtrada.max().to_period("M") - coluna_filtrada.min().to_period("M")).n // 4
        coluna_filtrada.hist(bins=n_bins)

    if nivel == 2:
        coluna_modificada = coluna_modificada.astype(float)
        coluna_filtrada = coluna_modificada.clip(lower=1900, upper=2026)
        n_bins = (int(coluna_filtrada.max() - coluna_filtrada.min()) + 1) // 2
        coluna_filtrada.plot.hist(bins=n_bins)

    plt.title(f"Histograma do atributo data nascimento\npara nível de generalização {nivel}")
    plt.xlabel("Data de nascimento")
    plt.ylabel("Contagem de registros")
    plt.savefig(os.path.join(output_path, f"datebirth_histogram_level_{nivel}.png"))
    plt.show()




# # Exemplo de execução do código
# dados = pd.read_csv("data/Covid1.csv", sep = ";")
# NIVEL = 1
#
# data_nascimento = dados.dataNascimento
# funcao_map = [map_data_nascimento_nivel_0, map_data_nascimento_nivel_1, map_data_nascimento_nivel_2][NIVEL]
# map_json = gera_json_nivel(data_nascimento, nivel = NIVEL)
#
# coluna_modificada = data_nascimento.apply(funcao_map)
# gera_histograma(coluna_modificada, nivel = NIVEL)
# gera_json_final(data_nascimento)