# Trabalho III: Publicação Diferencialmente Privada de Histogramas
**Mecanismo de Laplace Aplicado a Dados Eleitorais**

Este projeto consiste na implementação e avaliação do mecanismo de Laplace de Privacidade Diferencial.   
O cenário experimental utiliza dados reais de votação por seção eleitoral das eleições municipais de 2024 (Fortaleza/CE), simulando um ambiente de dados sensíveis.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

O projeto foi desenvolvido em Python 3, utilizando as seguintes bibliotecas principais (conforme exigido na estrutura de entrega):

* **`pandas`**: Utilizada para a manipulação do dataset original (`votacao_prefeito_CE_2024.csv`).
* **`numpy`**: Utilizada para o processamento matemático das informações, operações estruturadas de vetores (cálculo de métricas como MAE e TAE) e, principalmente, para a amostragem de ruído através da função `numpy.random.laplace`.
* **`matplotlib`**: Utilizada para a geração e plotagem dos gráficos de análise (variação do erro absoluto médio, erro absoluto total e probabilidade empírica de preservação do vencedor em função de $\epsilon$).

---

## 📁 Estrutura de Arquivos do Código

O código-fonte está organizado de forma modular nos seguintes arquivos:

* `laplace.py`: Contém a implementação do mecanismo de Laplace, recebendo o histograma original, a sensibilidade global da consulta e o orçamento de privacidade $\epsilon$.
* `utils.py`: Concentra as funções auxiliares do projeto, incluindo as rotinas de pré-processamento dos dados, o cálculo das métricas de erro (MAE, TAE, erro no maior bin e probabilidade empírica) e a geração automatizada dos gráficos.
* `main.py`: O script principal que coordena o fluxo do experimento. Ele carrega os dados, executa as repetições necessárias (20 vezes) para cada valor de $\epsilon \in \{0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10\}$ e salva os resultados finais.

---

## 🚀 Instruções de Execução

Siga os passos abaixo para preparar o ambiente e rodar as experimentações locais.

### 1. Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado em sua máquina.

### 2. Instalação das Dependências
No terminal, navegue até a pasta raiz do projeto e instale as bibliotecas necessárias utilizando o `pip`:

```bash
pip install pandas numpy matplotlib
```

### 3. Execução do código
Para rodar a simulação completa, gerar as tabelas descritivas e exportar os gráficos exigidos no relatório, execute o seguinte comando:

```bash
python main.py
```
Após a execução, as informações do histograma real serão ilustradas no terminal, e os gráficos gerados serão salvos localmente na pasta do projeto.