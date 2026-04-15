import pandas as pd
import numpy as np
import os
import time

import utils_tarefa2
from utils_tarefa2 import k_l_anonymity_greedy_month

if not os.path.isfile("Covid_clean.csv"):
    df = pd.read_csv('../Trab1 - Generalization/Covid1.csv', sep=';')
    df_clean = df[df['idadeCaso'].between(1, 100)]
    df_clean = df_clean.reset_index(drop=True)
    df_clean.to_csv(f'Covid_clean.csv', index=False, sep=';')

df = pd.read_csv('Covid_clean.csv', sep=';')

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
        df_copy = df_copy[["idadeCaso", "dataNascimento", "racaCor"]]


        possible_k = [1, 2, 4, 8, 16]
        possible_l = [1, 2, 3, 4]

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
        dataframe_agrupado_k_l = k_l_anonymity_greedy_month(df_copy, col_data_nasc = "dataNascimento", col_sensivel = "racaCor",
                                                    k = k, l = l)
        tamanhos_grupos_k_l = dataframe_agrupado_k_l.groupby("grupo")["idadeCaso"].count().rename({"idadeCaso" : "count"})
        print(df_copy.shape, dataframe_agrupado_k_l.shape)
        print(tamanhos_grupos_k_l.describe())

        data_length = len(dataframe_agrupado_k_l)

        manager = ManagerSIAttributes(data_length)
        manager.add('idadeCaso', 3)
        manager.add('dataNascimento', 2)

        print("Aguarde, seu dataset está sendo preparado...")  
        time.sleep(1)
        df_copy['idadeCaso'] = df_copy['idadeCaso'].astype(str)

        if k != 1 and l != 1:
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

