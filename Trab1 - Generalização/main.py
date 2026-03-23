import pandas as pd
import time
import os

from anonymize_birthdate import *
from anonymize_age import *

print("Inicializando o programa...")

os.makedirs('./Generated Datasets/', exist_ok=True)
os.makedirs('./Generated JSONs/', exist_ok=True)
os.makedirs('./Generated Plots/', exist_ok=True)


df = pd.read_csv('Covid1.csv', sep=';')

idadeCaso_raw = df.idadeCaso
dataNascimento_raw = df.dataNascimento

# json for anonymized attributes
generate_json_age(idadeCaso_raw, './Generated JSONs')
gera_json_final(dataNascimento_raw, './Generated JSONs')

def static_menu():
    print('-=' * 20 + '-')
    print('Bem vindo.')
    print('Escolha um nível de anonimização para idade e para nascimento.')
    print('-' * 20)
    print('Idade: 0 a 3')
    print('Nascimento: 0 a 2')
    print('-' * 20)
    print('Para sair do programa, digite -1 em ambos os casos.')

class SIAttribute:
    def __init__(self, name, maximum_hierarchy):
        self.name = name
        self.maximum_hierarchy = maximum_hierarchy
        self.current_hierarchy = 0

class ManagerSIAttributes:
    def __init__(self):
        self.attributes = {}
    def add(self, name, maximum_hierarchy):
        self.attributes[name] = SIAttribute(name, maximum_hierarchy)

manager = ManagerSIAttributes()
manager.add('idadeCaso', 3)
manager.add('dataNascimento', 2)

def calculate_dataset_precision(data_length, manager):
    total_sum = 0
    for _, attribute in manager.attributes.items():
        current_sum = data_length * (attribute.current_hierarchy / attribute.maximum_hierarchy)
        total_sum += current_sum
    precision = 1 - total_sum / (data_length * len(manager.attributes))
    return precision

while True:
    df_copy = df.copy()

    # user's input
    static_menu()
    try:
        ni = int(input('Nível de anonimização para idade: '))
        nd = int(input('Nível de anonimização para nascimento: '))
    except:
        print('Entrada em formato errado. Tente novamente.')
        time.sleep(1)
        continue

    if ni == -1 and nd == -1:
        break
    elif ni == -1 or nd == -1:
        print('Tente novamente.')
        time.sleep(1)
        continue
    print(f'Seus níveis escolhidos foram {ni} para idade e {nd} para nascimento.')

    manager.attributes['idadeCaso'].current_hierarchy = ni
    manager.attributes['dataNascimento'].current_hierarchy = nd

    # dataset precision
    data_length = len(df_copy)
    dataset_precision = calculate_dataset_precision(data_length, manager)

    print('-' * 20)
    print(f'A precisão do seu dataset é {dataset_precision}')

    # modifying dataset
    new_age = anonymize_age(idadeCaso_raw, ni)

    map_function = [map_data_nascimento_nivel_0,
                    map_data_nascimento_nivel_1,
                    map_data_nascimento_nivel_2][nd]
    new_birth_date = dataNascimento_raw.apply(map_function)


    df_copy['idadeCaso'] = new_age
    df_copy['dataNascimento'] = new_birth_date


    gera_histograma(new_birth_date, nd, './Generated Plots/')

    age_for_hist = idadeCaso_raw[(idadeCaso_raw >= 1) & (idadeCaso_raw <= 100)]
    result_for_hist = anonymize_age(age_for_hist, ni)
    generate_histogram_age(result_for_hist, ni, './Generated Plots')

    print('Salvando Dataset...')

    df_copy.to_csv(f'./Generated Datasets/DT_{ni}_{nd}.csv', index=False, sep=';')
