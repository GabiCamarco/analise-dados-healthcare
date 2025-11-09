#  Projeto de Data Science: Otimização e Previsão de Risco na HealthCare Solutions

## Visão Geral do Projeto

Este projeto utiliza o ciclo completo de Data Science para transformar dados hospitalares simulados em inteligência acionável. O objetivo principal é desenvolver um **modelo preditivo** que avalie o **risco de readmissão hospitalar em 30 dias** e identifique os fatores operacionais e clínicos que mais contribuem para este risco.

O projeto foi estruturado em três etapas principais, conforme os scripts abaixo.

## Como Executar o Projeto

Para rodar a análise e o modelo de Machine Learning, siga a ordem de execução abaixo:

### 1. Pré-requisitos
Certifique-se de ter o Python (versão 3.8+) instalado. É altamente recomendável usar um ambiente virtual (`venv`).

### 2. Instalação das Dependências
Instale todas as bibliotecas necessárias via `pip`:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib faker
2. Como Executar o Projeto
​Os scripts devem ser executados em ordem numérica para garantir a correta dependência de dados:
​Script 1_gerar_dados.py (Criação do Dataset):
​Cria e anonimiza o arquivo dataset_healthcare_solutions_v2.csv.
​Comando: python 1_gerar_dados.py
​Script 2_eda_analise.py (Análise Exploratória de Dados - EDA):
​Gera gráficos exploratórios (Boxplots, Violin Plots) na pasta graficos_v2/.
​Comando: python 2_eda_analise.py
​Script 3_modelagem_preditiva.py (Modelagem Preditiva - ML):
​Treina o modelo Random Forest, exibe métricas no console e gera gráficos de Matriz de Confusão e Importância das Features na pasta graficos_v2/.
​Comando: python 3_modelagem_preditiva.py
​ 3. Estrutura do Repositório
​O repositório contém os seguintes arquivos essenciais:
​1_gerar_dados.py: Criação e anonimização dos dados.
​2_eda_analise.py: Análise Exploratória de Dados.
​3_modelagem_preditiva.py: Modelagem de Machine Learning e avaliação.
​dataset_healthcare_solutions_v2.csv: Base de dados simulada.
​graficos_v2/: Pasta que armazena todos os resultados visuais gerados.
​ 4. Principais Insights e Resultados
​O modelo de Machine Learning (Random Forest Classifier) alcançou uma Acurácia de 70% na previsão de readmissão.
​Fatores Críticos de Risco (Importância das Features):
​O modelo identificou as variáveis que mais impulsionam o risco de readmissão. A gestão deve priorizar a ação nestes pontos:
​Custo de Atendimento: É o fator número 1, indicando que pacientes de alto custo demandam atenção redobrada pós-alta.
​Tempo de Espera (min): É o segundo fator mais importante, o que exige a otimização urgente dos fluxos de triagem e atendimento inicial.
​Idade: O terceiro fator, que valida a necessidade de programas de acompanhamento focados em faixas etárias de maior risco.
​Conformidade Legal (LGPD):
​Todos os dados utilizados no projeto foram 100% simulados através da biblioteca Faker. Essa abordagem garante total aderência aos princípios da Lei Geral de Proteção de Dados (LGPD), assegurando a ética e a segurança de informações sensíveis.


Vídeo Pitch:** Assista à explicação detalhada e a demonstração dos resultados aqui: https://youtu.be/daRTDf8OSts
