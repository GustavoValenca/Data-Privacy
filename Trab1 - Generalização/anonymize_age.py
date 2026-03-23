import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import Counter


def anonymize_age(data, level):
    if level == 0:
        return data
    elif level == 1:
        classes = {}
        for i in range(1, 101, 5):
            class_name = f'[{i}-{i + 5 - 1}]'
            classes[class_name] = (i, i + 5 - 1)

    elif level == 2:
        classes = {'Criança': (1, 10), 'Adolescente': (11, 17), 'Adulto Jovem': (18, 35),
               'Adulto': (36, 59), 'Idoso': (60, 100)}

    elif level == 3:
        classes = {'[1-100]': (1, 100)}

    choices = []
    conditions = []
    for key, values in classes.items():
        choices.append(key)
        conditions.append((data >= values[0]) & (data <= values[1]))

    result = np.select(conditions, choices, default=None)
    return result

def generate_json_age(data, output_path):
    json_memory = {}

    data_length = len(data)
    for level in range(4):
        json_memory[f'nível {level}'] = {}
        data_level = anonymize_age(data, level)
        for i in range(data_length):
        # writing data by checking if register has 'item' method
            json_memory[f'nível {level}'][str(i)] = data_level[i].item() if hasattr(data_level[i], "item") else data_level[i]

    with open(os.path.join(output_path, f"anonymized_age.json"), "w", encoding="utf-8") as f:
        json.dump(json_memory, f, ensure_ascii=False, indent=2)
    return json_memory

def generate_histogram_age(data, level, output_path): # data must contain values in the [1-100] interval
    if level == 0:
        plt.hist(data, bins=100, edgecolor='black')
        plt.title(f"Histograma do atributo Idade\npara nível de generalização {level}")
        plt.xlabel("Idade")
        plt.ylabel("Contagem de registros")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"age_histogram_level_{level}.png"))
        plt.show()

        return
    
    count = Counter(data)

    if level == 1:
        classes_in_order = []
        for i in range(1, 101, 5):
            class_name = f'[{i}-{i + 5 - 1}]'
            classes_in_order.append(class_name)
        
    elif level == 2:
        classes_in_order = ['Criança', 'Adolescente', 'Adulto Jovem', 'Adulto', 'Idoso']
    else:
        classes_in_order = ['[1-100]']

    dict_in_order = {chave: count[chave] for chave in classes_in_order}
    plt.bar(dict_in_order.keys(), dict_in_order.values())
    plt.title(f"Histograma do atributo Idade\npara nível de generalização {level}")
    plt.xlabel("Faixa")
    plt.ylabel("Contagem de registros")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"age_histogram_level_{level}.png"))

    plt.show()
    return