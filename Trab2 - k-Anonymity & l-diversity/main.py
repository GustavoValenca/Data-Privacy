import pandas as pd
import numpy as np
import os
import time

import utils_tarefa2

if not os.path.isfile("Covid_clean.csv"):
    df = pd.read_csv('../Trab1 - Generalization/Covid1.csv', sep=';')
    df_clean = df[df['idadeCaso'].between(1, 100)]
    df_clean = df_clean.reset_index(drop=True)
    df_clean.to_csv(f'Covid_clean.csv', index=False, sep=';')

df = pd.read_csv('Covid_clean.csv', sep=';')

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


amostra_pequena = df.sample(100000, random_state = 23)
amostra_pequena = amostra_pequena.reset_index(drop=True)

# just to manage attributes' hierarchies
class SIAttribute:
    def __init__(self, name, maximum_hierarchy):
        self.name = name
        self.maximum_hierarchy = maximum_hierarchy
        self.hierarchy_list = None

class ManagerSIAttributes:
    def __init__(self, length):
        self.attributes = {}
        self.data_length = length
    def add(self, name, maximum_hierarchy):
        self.attributes[name] = SIAttribute(name, maximum_hierarchy)
        self.attributes[name].hierarchy_list = [0] * self.data_length


def calculate_dataset_precision(manager):
    total_sum = 0
    data_length = manager.data_length
    for _, attribute in manager.attributes.items():
        current_sum = 0
        for i in range(data_length):
            current_sum += (attribute.hierarchy_list[i] / attribute.maximum_hierarchy)
        total_sum += current_sum
    precision = 1 - total_sum / (data_length * len(manager.attributes))
    return precision

def static_menu():
    print('-=' * 20 + '-')
    print('Bem vindo.')
    print('Escolha um valor k para o k-anonimato e um valor l para o l-diversidade.')
    print('Lembre-se que o valor de l sempre deverá ser menor ou igual a k.')
    print('-' * 20)
    print('k: {2, 4, 8, 16}')
    print('l: {2, 3, 4}')
    print('-' * 20)
    print('Para sair do programa, digite -1 em ambos os casos.')



mostrar_grupos_k_l = True
if mostrar_grupos_k_l:
    while True:
        df_copy = df.copy()

        possible_k = [2, 4, 8, 16]
        possible_l = [2, 3, 4]

        static_menu()


        try:
            k = int(input('Valor k: '))
            l = int(input('Valor l: '))
        except:
            print('Entrada em formato errado. Tente novamente.')
            time.sleep(1)
            continue

        if k == -1 and l == -1:
            break
        elif k == -1 or k == -1:
            print('Tente novamente.')
            time.sleep(1)
            continue
        elif k not in possible_k:
            print("Valor não disponível para k.")
            time.sleep(1)
            continue
        elif l > k:
            print("O valor de l deve ser menor que k.")
            time.sleep(1)
            continue
        elif l not in possible_l:
            print("valor não disponível para l.")
            time.sleep(1)
            continue
        print(f'Seus k e l escolhidos foram {k} e {l}, respectivamente.')
        print("Rodando execução do K-anonimato integrado com L-diversidade")
        # k = 1
        # l = 1
        dataframe_agrupado_k_l = k_l_anonymity_greedy_month(amostra_pequena, col_data_nasc = "dataNascimento", col_sensivel = "racaCor",
                                                    k = k, l = l)
        tamanhos_grupos_k_l = dataframe_agrupado_k_l.groupby("grupo")["idadeCaso"].count().rename({"idadeCaso" : "count"})
        print(amostra_pequena.shape, dataframe_agrupado_k_l.shape)
        print(tamanhos_grupos_k_l.describe())

        data_length = len(dataframe_agrupado_k_l)

        manager = ManagerSIAttributes(data_length)
        manager.add('idadeCaso', 3)
        manager.add('dataNascimento', 2)

        df_copy['idadeCaso'] = df_copy['idadeCaso'].astype(str)

        for i in dataframe_agrupado_k_l.grupo.unique():
            grupo = dataframe_agrupado_k_l[dataframe_agrupado_k_l.grupo == i].copy()
            grupo_generalizado, niveis = utils_tarefa2.generalizacao_minima_grupo(grupo)

            # print(niveis)
            indices = niveis['groupIndexes']
            # print(niveis)
            # print(grupo_generalizado)

            for j in indices:
                # print(j)
                manager.attributes['idadeCaso'].hierarchy_list[j] = niveis['idadeCaso']
                manager.attributes['dataNascimento'].hierarchy_list[j] = niveis['dataNascimento']

                valor_idade = grupo_generalizado.loc[j, 'idadeCaso']
                valor_nascimento = grupo_generalizado.loc[j, 'dataNascimento']

                df_copy.loc[j, 'idadeCaso'] = str(valor_idade)
                df_copy.loc[j, 'dataNascimento'] = str(valor_nascimento)
            
        df_copy = df_copy[["idadeCaso", "dataNascimento", "racaCor"]]  
        print(f'Tamanho do dataset: {manager.data_length}')
        dataset_precision = calculate_dataset_precision(manager)
        print(f'A precisão do seu dataset é { round(dataset_precision, 4)}')
        df_copy.to_csv(f"covid_{k}_{l}.csv")
        print()
        # for i in range(1000, 1015):
        #     print("Grupo", i)
        #     grupo = dataframe_agrupado_k_l[dataframe_agrupado_k_l.grupo == i].copy()
        #     grupo_generalizado, niveis = utils_tarefa2.generalizacao_minima_grupo(grupo)
        #     print(grupo)
        #     print(niveis)
        #     print(grupo_generalizado)
        #     print()

