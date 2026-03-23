# Trabalho 1: Generalização - Privacidade de Dados 2026

Disciplina: Privacidade de Dados  
Professor: Javam Machado  
Equipe: Gustavo Valença, João Paulo Lima

## Objetivo
O trabalho consiste em implementar métodos de anonimização no dataset `Covid1.csv` por meio da técnica de generalização. O foco da anonimização está nos atributos `idadeCaso` e `dataNascimento`, que são tratados como semi-identificadores.

## Especificações de Generalização

O projeto implementa hierarquias de generalização específicas para dois atributos:

* Atributo `idadeCaso` (4 níveis de hierarquia):**
    * **Nível 0:** Dado original.
    * **Nível 1:** Intervalos de 5 valores (ex: 1 a 5, 6 a 10, ...).
    * **Nível 2:** Ciclos de vida (Criança: 1-10, Adolescente: 11-17, Adulto Jovem: 18-35, Adulto: 36-59, Idoso: 60+).
    * **Nível 3:** Agregação total, englobando todas as idades (1-100). *(Nota: Registros com idades fora do intervalo 1-100 são desconsiderados durante o pré-processamento).*
* **Atributo `dataNascimento` (3 níveis de hierarquia):**
    * **Nível 0:** Dado original no formato `aaaa-mm-dd`.
    * **Nível 1:** Apenas ano e mês no formato `aaaa-mm`.
    * **Nível 2:** Apenas o ano no formato `aaaa`.

## Cálculo de Precisão
Para cada dataset gerado, o programa calcula a precisão dos dados, comparando o dataset anonimizado com o original. A fórmula matemática utilizada no código para o cálculo da precisão é:

$$Prec(D)=1-\frac{\sum_{i=1}^{N_{a}}\sum_{j=1}^{|D|}\frac{h}{|HG_{A_{i}}|}}{|D|\times|N_{a}|}$$

Onde:
* $N_a$ é o número de atributos semi-identificadores.
* $D$ é o conjunto de dados.
* $h$ é a altura da hierarquia de generalização do atributo após a generalização.
* $HG$ é a altura máxima da hierarquia do atributo.

## Estrutura do Projeto

* `main.py`: Script principal que inicializa o menu interativo, solicita os níveis de publicação ($n_i$ e $n_d$), percorre o dataset substituindo os valores pela hierarquia e salva os arquivos resultantes.
* `anonymize_age.py`: Módulo responsável pelas funções de mapeamento e generalização do atributo `idadeCaso`, além da geração dos JSONs de de/para e plotagem dos histogramas das idades.
* `anonymize_birthdate.py`: Módulo responsável pelas lógicas de formatação do atributo `dataNascimento`, criação dos dicionários JSON e geração dos gráficos de histograma por nível.
* `/Generated Datasets/`: Diretório onde os datasets de publicação (DT\_ni\_nd) gerados a cada iteração são armazenados, contendo o mesmo esquema e número de registros do dataset original.
* `/Generated JSONs/`: Diretório onde os arquivos `.json` de mapeamento das hierarquias ficam salvos.
* `/Generated Plots/`: Diretório de saída dos histogramas gerados para cada nível escolhido.
* `/Generated Precisions/`: Diretório contendo os resultados da métrica precision para os níveis escolhidos.

## Como Executar

1.  Certifique-se de possuir o **Python 3.x** instalado em sua máquina e as bibliotecas listadas como dependência (`pandas`, `numpy`, `matplotlib`).
2.  Coloque o dataset `Covid1.csv` na mesma pasta do script principal.
3.  Execute o arquivo principal pelo terminal:
    ```bash
    python main.py
    ```
4.  O menu interativo solicitará os valores de publicação. Insira o valor desejado para a idade (0 a 3) e para o nascimento (0 a 2).
5.  O programa gerará os arquivos correspondentes dinamicamente, os gravando em disco. 
6.  Para sair do loop, basta digitar `-1` para os dois valores de entrada, conforme a condição de saída.

***
