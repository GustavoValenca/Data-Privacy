import pandas as pd
import numpy as np
from anonymize_age import anonymize_age
from anonymize_birthdate import map_data_nascimento

# -------------------------
# Funções de generalização
# -------------------------

def generalizar_idade(valor, nivel):
    if pd.isna(valor):
        return np.nan

    valor = int(valor)

    # Remover valores inválidos
    if valor < 1 or valor > 100:
        return None

    if nivel == 0:
        return valor

    elif nivel == 1:
        # Intervalos de 5
        inicio = ((valor - 1) // 5) * 5 + 1
        fim = inicio + 4
        return f"{inicio}-{fim}"

    elif nivel == 2:
        if 1 <= valor <= 10:
            return "crianca"
        elif 11 <= valor <= 17:
            return "adolescente"
        elif 18 <= valor <= 35:
            return "adulto_jovem"
        elif 36 <= valor <= 59:
            return "adulto"
        else:
            return "idoso"

    elif nivel == 3:
        return "1-100"


def generalizar_data(data, nivel):
    if pd.isna(data):
        return np.nan

    data = pd.to_datetime(data)

    if nivel == 0:
        return data.strftime("%Y/%m/%d")

    elif nivel == 1:
        return data.strftime("%Y/%m")

    elif nivel == 2:
        return data.strftime("%Y")


# -------------------------
# Função principal
# -------------------------

def generalizacao_minima_grupo(df_grupo,
                               col_idade="idadeCaso",
                               col_data="dataNascimento"):
    df = df_grupo.copy()

    niveis_idade = [0, 1, 2, 3]
    niveis_data = [0, 1, 2]

    # testar combinações (garante mínima generalização)
    for ni in niveis_idade:
        for nd in niveis_data:

            temp = df.copy()

            temp[col_idade] = temp[col_idade].apply(lambda x: anonymize_age(x, ni))
            temp[col_data] = temp[col_data].apply(lambda x: map_data_nascimento(x, nd))

            # Verificar anonimato: todos iguais
            if (temp[col_idade].nunique() == 1) and (temp[col_data].nunique() == 1):
                melhor_df = temp
                melhor_niveis = {
                    "idadeCaso": ni,
                    "dataNascimento": nd,
                    "groupSize": df.shape[0],
                    "groupIndexes": list(df.index)
                }
                return melhor_df, melhor_niveis

    # Caso não seja possível anonimizar o grupo com base nas generalizações fornecidas
    return None, None

def verifica_inconsistencias(dataframe_agrupado_k_l, k, l):
    for i in dataframe_agrupado_k_l.grupo.unique():
        grupo = dataframe_agrupado_k_l[dataframe_agrupado_k_l.grupo == i].copy()
        grupo_generalizado, niveis = generalizacao_minima_grupo(grupo)
        if niveis is None or niveis["groupSize"] < k or grupo_generalizado.racaCor.nunique() < l:
            print("Grupo",i,"problemático")
            impossibilidade_generalizacao = (niveis is None)
            grupo_pequeno = (niveis["groupSize"] < k)
            sem_l_diversidade = (grupo_generalizado.racaCor.nunique() < l)
            if impossibilidade_generalizacao: print("- Grupo não é possível generalizar")
            if grupo_pequeno: print("- Grupo pequeno")
            if sem_l_diversidade: print("- Grupo sem l-diversidade")
            print(grupo)


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