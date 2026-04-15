import pandas as pd
from utils_tarefa2 import k_l_anonymity_greedy_month, generalizacao_minima_grupo
import matplotlib.pyplot as plt

def plota_histograma_tamanhos_grupos(dataframe_agrupado_k_l, k, l):
    tamanhos_grupos_k_l = dataframe_agrupado_k_l.groupby("grupo")["idadeCaso"].count().rename({"idadeCaso" : "count"})
    tamanho_bins = tamanhos_grupos_k_l.max() - tamanhos_grupos_k_l.min()
    tamanhos_grupos_k_l.plot.hist(bins = tamanho_bins)
    #tamanhos_grupos_k_l.value_counts().plot.bar()
    plt.title(f"Distribuição dos tamanhos das classe de equivalência\npara o K = {k} e L = {l}")
    plt.ylabel("Contagem")
    plt.xlabel("Tamanho do grupo")
    plt.show()


def plota_grafico_variacao_k(df, l):
    lista_k = [2, 4, 6, 8, 10, 12, 14, 16]
    lista_tamanhos = []
    for k in lista_k:
        dataframe_agrupado_k_l = k_l_anonymity_greedy_month(df, col_data_nasc="dataNascimento",
                                                            col_sensivel="racaCor",
                                                            k=k, l=l)
        tamanhos_grupos_k_l = dataframe_agrupado_k_l.groupby("grupo")["idadeCaso"].count().rename(
            {"idadeCaso": "count"})
        tamanho_medio = tamanhos_grupos_k_l.mean()
        lista_tamanhos.append(tamanho_medio)

    plt.bar(lista_k, lista_tamanhos)
    plt.title(f"Tamanho médio dos grupos com variação do k\ne com L = {l}")
    plt.ylabel("Tamanho médio")
    plt.xlabel("K")
    plt.show()

def plota_valores_distintos(dataframe_agrupado_k_l, k, l):
    lista_valores_distintos = []
    for i in dataframe_agrupado_k_l.grupo.unique():
        grupo = dataframe_agrupado_k_l[dataframe_agrupado_k_l.grupo == i].copy()
        valores_distintos = grupo.racaCor.nunique()
        lista_valores_distintos.append(valores_distintos)
    distribuicao_distintos = pd.Series(lista_valores_distintos).value_counts().sort_index()
    print(distribuicao_distintos)
    distribuicao_distintos.plot.bar()
    plt.title(f"Distribuição de valores distintos entre as classes de equivalência\npara k = {k} e l = {l}")
    plt.ylabel("Contagem")
    plt.xlabel("Quantidade de distintos")
    plt.show()