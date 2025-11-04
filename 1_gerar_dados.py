import pandas as pd
import numpy as np
from faker import Faker
import random

# Inicializando gerador de dados
fake = Faker('pt_BR')
# MUDANÇA 1: Mudando a SEED para gerar uma sequência diferente de números
np.random.seed(123) 

# Quantidade de registros simulados
n = 150 # MUDANÇA 2: Aumentando o número de registros para 150

# Gerando dados simulados
dados = {
    'id_paciente': range(1, n + 1),
    # MUDANÇA 3: Adicionando Nome e Cidade (para análise de localização)
    'nome': [fake.name() for _ in range(n)], 
    'cidade': [fake.city() for _ in range(n)],
    
    'idade': np.random.randint(15, 95, size=n), # MUDANÇA 4: Mudando a faixa etária (15-95)
    'sexo': [random.choice(['Masculino', 'Feminino', 'Outro']) for _ in range(n)], # MUDANÇA 5: Adicionando 'Outro'
    
    # MUDANÇA 6: Usando distribuição Lognormal para Tempo de Espera 
    # (Simula que a maioria espera pouco, mas alguns esperam MUITO)
    'tempo_espera_min': np.round(np.random.lognormal(mean=3.5, sigma=0.8, size=n), 0).astype(int), 
    
    'satisfacao_paciente': np.random.randint(1, 6, size=n),
    'diagnostico': [random.choice(['Gripe', 'Covid-19', 'Fratura', 'Asma', 'Infecção Urinária']) for _ in range(n)], # MUDANÇA 7: Mudando diagnósticos
    
    'dispositivo_monitoramento': [random.choice(['Smartwatch', 'Oxímetro', 'Glicômetro', 'Pressão Arterial', 'Nenhum']) for _ in range(n)], # MUDANÇA 8: Mudando dispositivos
    
    # MUDANÇA 9: Nova variável para EDA (Nível de Urgência)
    'nivel_urgencia': [random.choice(['Baixa', 'Média', 'Alta']) for _ in range(n)],
    
    'readmissao_30dias': [random.choice(['Sim', 'Não']) for _ in range(n)],
    'tempo_internacao_dias': np.random.randint(1, 10, size=n), # MUDANÇA 10: Diminuindo o tempo máximo de internação
    'custo_atendimento': np.round(np.random.uniform(800, 20000, size=n), 2) # MUDANÇA 11: Aumentando faixa de custo
}

# Criando DataFrame
df = pd.DataFrame(dados)

# Ajuste extra para a nova distribuição de tempo de espera
df['tempo_espera_min'] = df['tempo_espera_min'].clip(lower=5, upper=240) 

# Exibindo as primeiras linhas
print(df.head())

# Salvando em CSV (JÁ ESTÁ COM O NOVO NOME!)
df.to_csv("dataset_healthcare_solutions_v2.csv", index=False, encoding='utf-8-sig')
print("\nArquivo 'dataset_healthcare_solutions_v2.csv' criado com sucesso!")