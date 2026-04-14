import pandas as pd
import numpy as np

import utils_tarefa2

dados = pd.read_csv("Covid1.csv", sep = ";")

dados_filtrados = dados[["idadeCaso", "dataNascimento", "racaCor"]].dropna()
dados_filtrados = dados_filtrados[ (dados_filtrados.idadeCaso > 0) &  (dados_filtrados.idadeCaso <= 100) ]

print( dados_filtrados.racaCor.unique() )
print(dados_filtrados.racaCor.value_counts())

def k_l_anonymity_greedy_month(
        df,
        col_data_nasc,
        col_sensivel,
        k,
        l
):
    df = df.copy()
    df[col_data_nasc] = pd.to_datetime(df[col_data_nasc])

    # Extrair ano e ano_mes
    df["ano"] = df[col_data_nasc].dt.year
    df["ano_mes"] = df[col_data_nasc].dt.to_period("M")

    # Inicia a coluna 'grupo' no dataframe principal com -1
    df["grupo"] = -1

    # Variável global de grupos para não repetir IDs entre anos diferentes
    grupo_id_global = 0

    # Agrupar e processar por ano isoladamente
    for ano, df_ano in df.groupby("ano"):
        # Ordenar o sub-dataframe SEM resetar o index
        df_ano = df_ano.sort_values([col_data_nasc])

        # Cria uma coluna local de grupos para este ano
        df_ano["grupo"] = -1


        # Contador de grupos locais para o ajuste final
        grupos_locais_criados = 0

        i = 0
        n = len(df_ano)
        # Armazena os índices originais na ordem em que foram classificados
        indices_ordenados = df_ano.index

        while i < n:
            grupo_labels = []  # Substitui grupo_indices, agora armazena o index original
            ano_mes_set = set()
            valores_sensiveis = set()

            # -------------------------
            # FASE 1: garantir k
            # -------------------------
            while len(grupo_labels) < k and i < n:
                idx = indices_ordenados[i]
                grupo_labels.append(idx)
                ano_mes_set.add(df_ano.loc[idx, "ano_mes"])
                valores_sensiveis.add(df_ano.loc[idx, col_sensivel])
                i += 1

            # -------------------------
            # FASE 2: garantir L-diversidade
            # -------------------------
            while i < n and len(valores_sensiveis) < l:
                idx = indices_ordenados[i]
                grupo_labels.append(idx)
                ano_mes_set.add(df_ano.loc[idx, "ano_mes"])
                valores_sensiveis.add(df_ano.loc[idx, col_sensivel])
                i += 1

            # -------------------------
            # FASE 3: verificação de semelhança do grupo
            # -------------------------
            if len(grupo_labels) > 0:
                idade_vals = df_ano.loc[grupo_labels, "idadeCaso"]
                data_vals = df_ano.loc[grupo_labels, col_data_nasc]

                mesma_idade = idade_vals.nunique() == 1
                mesma_data = data_vals.nunique() == 1

                if mesma_idade and mesma_data:
                    while i < n:
                        idx = indices_ordenados[i]
                        prox_mesma_idade = df_ano.loc[idx, "idadeCaso"] == idade_vals.iloc[0]
                        prox_mesma_data = df_ano.loc[idx, col_data_nasc] == data_vals.iloc[0]

                        if prox_mesma_idade and prox_mesma_data:
                            grupo_labels.append(idx)
                            valores_sensiveis.add(df_ano.loc[idx, col_sensivel])
                            ano_mes_set.add(df_ano.loc[idx, "ano_mes"])
                            i += 1
                        else:
                            break
                else:
                    # -------------------------
                    # FASE 4: expansão restrita
                    # -------------------------
                    while i < n:
                        if l == 1:
                            break

                        idx = indices_ordenados[i]
                        current_ano_mes = df_ano.loc[idx, "ano_mes"]

                        if current_ano_mes in ano_mes_set:
                            grupo_labels.append(idx)
                            valores_sensiveis.add(df_ano.loc[idx, col_sensivel])
                            i += 1
                        else:
                            break

            # Atribuir o ID de grupo global para os índices processados
            df_ano.loc[grupo_labels, "grupo"] = grupo_id_global

            grupo_id_global += 1
            grupos_locais_criados += 1

        # -------------------------
        # Ajuste final (Por Ano)
        # -------------------------
        if grupos_locais_criados > 0:
            last_group = grupo_id_global - 1
            mask_last = df_ano["grupo"] == last_group

            last_size = mask_last.sum()
            last_l = df_ano.loc[mask_last, col_sensivel].nunique()

            if (last_size < k or last_l < l):
                if grupos_locais_criados > 1:
                    df_ano.loc[mask_last, "grupo"] = last_group - 1
                else:
                    pass

        # Mapeia os grupos computados neste ano de volta para o dataframe principal
        # Isso funciona perfeitamente porque os índices originais foram mantidos
        df.loc[df_ano.index, "grupo"] = df_ano["grupo"]

    # Limpeza de colunas temporárias
    df = df.drop(columns=["ano", "ano_mes"])

    return df


amostra_pequena = dados_filtrados.sample(100000, random_state = 23)

mostrar_grupos_k_l = True
if mostrar_grupos_k_l:
    print("Rodando execução do K-anonimato integrado com L-diversidade")
    k = 16
    l = 4
    dataframe_agrupado_k_l = k_l_anonymity_greedy_month(amostra_pequena, col_data_nasc = "dataNascimento", col_sensivel = "racaCor",
                                                  k = k, l = l)
    tamanhos_grupos_k_l = dataframe_agrupado_k_l.groupby("grupo")["idadeCaso"].count().rename({"idadeCaso" : "count"})
    print(amostra_pequena.shape, dataframe_agrupado_k_l.shape)
    print(tamanhos_grupos_k_l.describe())

    print()
    for i in range(1000, 1015):
        print("Grupo", i)
        grupo = dataframe_agrupado_k_l[dataframe_agrupado_k_l.grupo == i].copy()
        grupo_generalizado, niveis = utils_tarefa2.generalizacao_minima_grupo(grupo)
        print(grupo)
        print(niveis)
        print()

    for i in dataframe_agrupado_k_l.grupo.unique():
        grupo = dataframe_agrupado_k_l[dataframe_agrupado_k_l.grupo == i].copy()
        grupo_generalizado, niveis = utils_tarefa2.generalizacao_minima_grupo(grupo)
        if niveis is None or niveis["groupSize"] < k or grupo_generalizado.racaCor.nunique() < l:
            print("Grupo",i,"problemático")
            impossibilidade_generalizacao = (niveis is None)
            grupo_pequeno = (niveis["groupSize"] < k)
            sem_l_diversidade = (grupo_generalizado.racaCor.nunique() < l)
            if impossibilidade_generalizacao: print("- Grupo não é possível generalizar")
            if grupo_pequeno: print("- Grupo pequeno")
            if sem_l_diversidade: print("- Grupo sem l-diversidade")
            print(grupo)