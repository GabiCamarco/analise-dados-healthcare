# Módulos para Modelagem
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier # Ótima escolha para classificação e fácil interpretação
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurando o estilo do Seaborn
sns.set_style("whitegrid")
os.makedirs("graficos_v2", exist_ok=True) # Reutilizando a pasta de gráficos

# Carregar o dataset
df = pd.read_csv('dataset_healthcare_solutions_v2.csv')

print("--- 1. PRÉ-PROCESSAMENTO DE DADOS ---")
# 1.1 Remoção de Colunas Não Relevantes/Identificadoras
# 'nome' e 'cidade' são identificadores ou geográficos que não contribuem diretamente
# para a readmissão em um modelo inicial. 'id_paciente' é apenas um índice.
df_model = df.drop(columns=['nome', 'cidade', 'id_paciente'])

# 1.2 Codificação da Variável Alvo (Label Encoding)
# Transformar 'Sim' e 'Não' em 1 e 0 (Requisito para o Machine Learning)
le = LabelEncoder()
df_model['readmissao_30dias_num'] = le.fit_transform(df_model['readmissao_30dias'])
df_model = df_model.drop(columns=['readmissao_30dias']) # Remove a coluna original

print(f"Variável Alvo ('readmissao_30dias'): {le.classes_} -> {le.transform(le.classes_)}")

# 1.3 Codificação de Variáveis Categóricas (One-Hot Encoding)
# Transforma colunas categóricas (como 'sexo', 'diagnostico') em colunas numéricas
# binárias, necessárias para o modelo.
categorical_cols = df_model.select_dtypes(include='object').columns
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

print(f"Colunas após One-Hot Encoding: {df_model.columns.tolist()}")

# 1.4 Definição de Variáveis Preditivas (X) e Variável Alvo (y)
X = df_model.drop('readmissao_30dias_num', axis=1)
y = df_model['readmissao_30dias_num']

# 1.5 Divisão em Conjuntos de Treino e Teste
# 80% para treino e 20% para teste (boa prática)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
# stratify=y garante que a proporção de 'Sim'/'Não' seja mantida em ambos os conjuntos

# 1.6 Normalização/Escalonamento (Padronização)
# Embora o Random Forest seja menos sensível, é boa prática para garantir
# que as variáveis (como custo e tempo de espera) tenham pesos similares.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Pré-processamento concluído: Dados prontos para o treinamento.")

print("\n--- 2. TREINAMENTO DO MODELO (Random Forest) ---")
# 2.1 Treinamento do Random Forest Classifier
# O Random Forest é robusto e fornece a importância das features.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Treinamento do modelo concluído.")

print("\n--- 3. AVALIAÇÃO E INTERPRETAÇÃO ---")
# 3.1 Previsão no Conjunto de Teste
y_pred = model.predict(X_test_scaled)

# 3.2 Relatório de Classificação
print("\nRelatório de Classificação (Previsão de Readmissão):")
# Métricas: Precision, Recall, F1-Score e Support
print(classification_report(y_test, y_pred, target_names=['Não Readmitido (0)', 'Readmitido (1)']))

# 3.3 Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Readmitido (0)', 'Readmitido (1)'],
            yticklabels=['Não Readmitido (0)', 'Readmitido (1)'])
plt.title('Matriz de Confusão')
plt.ylabel('Valores Reais')
plt.xlabel('Previsões do Modelo')
plt.tight_layout()
plt.savefig('graficos_v2/matriz_confusao.png')
plt.close()
print("✅ Matriz de Confusão salva.")

# 3.4 Importância das Features (Variáveis)
# Informação crucial para os INSIGHTS do projeto
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values[:10], y=feature_importances.index[:10], palette='magma')
plt.title('Top 10 Variáveis Mais Importantes na Previsão de Readmissão')
plt.xlabel('Importância')
plt.ylabel('Variável (Feature)')
plt.tight_layout()
plt.savefig('graficos_v2/importancia_features.png')
plt.close()
print("✅ Gráfico de Importância das Features salvo.")

print("\n--- FIM DA MODELAGEM ---")
print(f"Acurácia Geral do Modelo: {accuracy_score(y_test, y_pred):.2f}")
print("Examine os gráficos de 'matriz_confusao.png' e 'importancia_features.png' para extrair INSIGHTS.")