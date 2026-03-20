import pandas as pd
import numpy as np
import random
import json
import os

print(os.listdir('.'))

df = pd.read_csv('Covid1.csv', sep=';')

idadeCaso_raw = df.idadeCaso

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

def generate_json_age(data):
    json = {}

    data_length = len(data)
    for level in range(4):
        json[f'nível {level}'] = {}
        data_level = anonymize_age(data, level)
        for i in range(data_length):
        # writing data by checking if register has 'item' method
            json[f'nível {level}'][str(i)] = data_level[i].item() if hasattr(data_level[i], "item") else data_level[i]

    return json

json_memory = generate_json_age(idadeCaso_raw)

with open("anonymized_age.json", "w", encoding="utf-8") as f:
    json.dump(json_memory, f, ensure_ascii=False, indent=2)