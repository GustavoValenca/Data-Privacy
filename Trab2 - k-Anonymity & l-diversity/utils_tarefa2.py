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