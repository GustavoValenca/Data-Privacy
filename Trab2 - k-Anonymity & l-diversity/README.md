# Trabalho 2: k-Anonimato + l-Diversidade - Privacidade de Dados 2026

Disciplina: Privacidade de Dados  
Professor: Javam Machado  
Equipe: Gustavo Valença, João Paulo Lima

## Objetivo
O trabalho consiste em implementar um algoritmo que anonimize um conjunto de dados contra ataques de ligaçãoo ao registro e ao atributo, atendendo ao mesmo tempo aos modelos k-anonimato e l-diversidade.

## Especificações Técnicas

Atributos Utilizados:

* Semi-identificadores: idadeCaso e dataNascimento.

* Atributo Sensível: racaCor.

Níveis de Generalização:

* Atributo `idadeCaso` (4 níveis de hierarquia):
    * **Nível 0:** Dado original.
    * **Nível 1:** Intervalos de 5 valores (ex: 1 a 5, 6 a 10, ...).
    * **Nível 2:** Ciclos de vida (Criança: 1-10, Adolescente: 11-17, Adulto Jovem: 18-35, Adulto: 36-59, Idoso: 60+).
    * **Nível 3:** Agregação total, englobando todas as idades (1-100). *(Nota: Registros com idades fora do intervalo 1-100 são desconsiderados durante o pré-processamento).*
* **Atributo `dataNascimento` (3 níveis de hierarquia):**
    * **Nível 0:** Dado original no formato `aaaa-mm-dd`.
    * **Nível 1:** Apenas ano e mês no formato `aaaa-mm`.
    * **Nível 2:** Apenas o ano no formato `aaaa`.

Configurações Suportadas
O programa aceita as seguintes combinações, respeitando a regra $l \le k$:
* k: {2, 4, 8, 16} 
* l: {2, 3, 4} 

## Cálculo de Precisão
Para cada dataset gerado, o programa calcula a precisão dos dados, comparando o dataset anonimizado com o original. A fórmula matemática utilizada no código para o cálculo da precisão é:

$$Prec(D)=1-\frac{\sum_{i=1}^{N_{a}}\sum_{j=1}^{|D|}\frac{h}{|HG_{A_{i}}|}}{|D|\times|N_{a}|}$$

Onde:
* $N_a$ é o número de atributos semi-identificadores.
* $D$ é o conjunto de dados.
* $h$ é a altura da hierarquia de generalização do atributo após a generalização.
* $HG$ é a altura máxima da hierarquia do atributo.

## Estrutura do Projeto


* `main.py`: Script principal que gerencia o fluxo de execução, interface de usuário (menu) e cálculo de precisão.

* `utils_tarefa2.py`: Contém a lógica de busca pela generalização mínima necessária para cada grupo.

* `anonymize_age.py` & `anonymize_birthdate.py`: Definem as hierarquias de generalização para os atributos "Idade" e "Data de Nascimento", respectivamente.

* `Covid_clean.csv`: Versão pré-processada do dataset original (removendo nulos).

## Como Executar

1.  Certifique-se de possuir o **Python 3.x** instalado em sua máquina e as bibliotecas listadas como dependência (`pandas`, `numpy`, `matplotlib`).

2. Certifique-se de que o arquivo Covid1.csv ou Covid_clean.csv esteja no diretório correto.

3. Execute o script principal

4. Insira os valores de k e l solicitados no menu interativo.

5. O programa gerará um arquivo de saída no formato covid_{k}_{l}.csv.

6.  Para sair do loop, basta digitar `-1` para os dois valores de entrada, conforme a condição de saída.

***
