# EDA plug-and-play com matplotlib e seaborn (Versão 2)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Adicionado para gráficos mais profissionais
import numpy as np # Necessário para np.polyfit e para .select_dtypes
import os

# Configurando o estilo do Seaborn para gráficos mais atraentes
sns.set_style("whitegrid")

# Criar pasta para gráficos, se não existir (MUDANÇA DE NOME DA PASTA)
os.makedirs("graficos_v2", exist_ok=True)

# Carregar o dataset (MUDANÇA DE NOME DO ARQUIVO)
df = pd.read_csv('dataset_healthcare_solutions_v2.csv')

# 1️⃣ Informações gerais
print("Informações do Dataset:")
print(df.info())
print("\n5 primeiras linhas:")
print(df.head())

# 2️⃣ Checar valores nulos
print("\nValores nulos por coluna:")
print(df.isnull().sum())

# 3️⃣ Estatísticas descritivas (incluindo categóricas)
print("\nEstatísticas descritivas:")
print(df.describe(include='all'))

# 4️⃣ Função para salvar gráfico e mostrar automaticamente
def mostrar_e_salvar(figura, nome_arquivo):
    
    # 1. SALVAR PRIMEIRO: Garante que a figura é salva antes de ser exibida/fechada.
    figura.savefig(f'graficos_v2/{nome_arquivo}.png')
    figura.savefig(f'graficos_v2/{nome_arquivo}.pdf')
    
    # 2. MOSTRAR: Exibe a figura em uma janela separada.
    plt.show(block=False) 
    
    # AUMENTE O TEMPO DE PAUSA PARA 3 SEGUNDOS (ou o que preferir)
    plt.pause(3)         

    # 3. FECHAR: Fecha a figura.
    plt.close(figura)

# 5️⃣ Gráficos (NOVAS VISUALIZAÇÕES E AJUSTES)

# 5.1 Distribuição de Idade (Original ajustado)
fig = plt.figure(figsize=(8,5))
plt.hist(df['idade'], bins=10, color='skyblue', edgecolor='black')
plt.axvline(df['idade'].mean(), color='red', linestyle='dashed', linewidth=1, label=f"Média: {df['idade'].mean():.1f}")
plt.axvline(df['idade'].median(), color='green', linestyle='dashed', linewidth=1, label=f"Mediana: {df['idade'].median():.1f}")
plt.title('Distribuição de Idade dos Pacientes')
plt.xlabel('Idade')
plt.ylabel('Número de Pacientes')
plt.legend()
mostrar_e_salvar(fig, 'hist_idade')

# 5.2 Boxplot do custo do atendimento (Original ajustado)
fig = plt.figure(figsize=(8,5))
plt.boxplot(df['custo_atendimento'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Boxplot do Custo do Atendimento')
mostrar_e_salvar(fig, 'box_custo')

# --- GRÁFICOS NOVOS (Seaborn) ---

# 5.3 Distribuição de Tempo de Espera por Urgência (NOVO GRÁFICO DE ORIGINALIDADE)
plt.figure(figsize=(10, 6))
sns.violinplot(x='nivel_urgencia', y='tempo_espera_min', data=df, palette='viridis', order=['Baixa', 'Média', 'Alta'])
plt.title('Distribuição do Tempo de Espera por Nível de Urgência', fontsize=14)
plt.xlabel('Nível de Urgência')
plt.ylabel('Tempo de Espera (minutos)')
fig = plt.gcf()
mostrar_e_salvar(fig, 'violin_tempo_espera_urgencia')

# 5.4 Relação entre Readmissão e Custo (NOVO GRÁFICO DE ORIGINALIDADE)
plt.figure(figsize=(8, 6))
sns.boxplot(x='readmissao_30dias', y='custo_atendimento', data=df, palette='Set2')
plt.title('Custo de Atendimento por Status de Readmissão', fontsize=14)
plt.xlabel('Houve Readmissão em 30 Dias?')
plt.ylabel('Custo do Atendimento')
fig = plt.gcf()
mostrar_e_salvar(fig, 'box_custo_readmissao')


# 5.5 Heatmap de correlação (Original ajustado com Seaborn)
# Seleciona apenas as colunas numéricas
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Mapa de Correlação das Variáveis Numéricas', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
fig = plt.gcf()
mostrar_e_salvar(fig, 'heatmap_correlacao_v2')


print("✅ EDA finalizado! Gráficos salvos em PNG e PDF na pasta 'graficos_v2'.")