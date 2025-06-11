# Versão melhorada V_2.0 para 100 sorteios
#
# Autor: Sergio Tabarez
# Data: abril/maio 2025
# Descrição: Modelo de Machine Learning para previsão de dezenas de loteria usando LSTMs e TensorFlow/Keras.
# Obs: Treinamento com um pequeno subconjunto de dados devido a demanda computacional
#
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tabulate import tabulate
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# === 1. Configuração de sementes para reprodutibilidade ===
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# === 2. Carregamento e pré-processamento dos dados ===
file_path = 'Lotofacil-original100.xlsx'
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()

dezena_columns = [f'Bola{i}' for i in range(1, 16)]
data = df[dezena_columns].values

# Converter cada linha em vetor binário (25 posições)
def dezenas_para_binarios(data, n_dezenas=25):
    binarios = []
    for linha in data:
        binario = np.zeros(n_dezenas)
        for dezena in linha:
            binario[int(dezena)-1] = 1
        binarios.append(binario)
    return np.array(binarios)

binarios = dezenas_para_binarios(data)

# === 3. Construção de sequências (com tamanho reduzido) ===
sequence_length = 20  # Reduzido para aumentar o número de amostras
X, y = [], []
for i in range(len(binarios) - sequence_length):
    X.append(binarios[i:i+sequence_length])
    y.append(binarios[i+sequence_length])
X = np.array(X)
y = np.array(y)

# === 4. Validação cruzada temporal e treinamento dos modelos ===
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

models = []
media_acertos_folds = []
mse_folds = []
mae_folds = []

# Função para criar o modelo simplificado
def criar_modelo(sequence_length):
    model = Sequential([
        LSTM(64,
             return_sequences=True,
             input_shape=(sequence_length, 25),
             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        LayerNormalization(),
        Dropout(0.4),
        LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        LayerNormalization(),
        Dropout(0.4),
        Dense(25, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.0001)  # Taxa de aprendizado reduzida (melhor desempenho)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Função para selecionar as 15 dezenas com maior probabilidade
def select_dezenas_binarias(predicted_probs, top_k=15):
    top_indices = np.argsort(predicted_probs)[::-1][:top_k]
    return sorted(top_indices + 1)

# EarlyStopping com maior paciência
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,  # Paciência aumentada (melhor desempenho)
    restore_best_weights=True,
    min_delta=0.001,
    verbose=1
)

for fold, (train_index, val_index) in enumerate(tscv.split(X)):
    print(f"\nFold {fold+1}/{n_splits}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Criação do modelo
    model = criar_modelo(sequence_length)

    print(f"Treinando com {len(X_train)} amostras, validando com {len(X_val)} amostras")

    # Treinamento
    history = model.fit(
        X_train, y_train,
        epochs=150,  # Número aumentado de épocas (melhor desempenho)
        batch_size=16,  # Batch size reduzido (melhor deaempenho)
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # Armazenar o modelo treinado
    models.append(model)

    # Previsões
    y_pred_probs = model.predict(X_val)

    # Métricas
    mse = mean_squared_error(y_val, y_pred_probs)
    mae = mean_absolute_error(y_val, y_pred_probs)
    mse_folds.append(mse)
    mae_folds.append(mae)

    # Cálculo da média de acertos
    total_acertos_fold = 0
    for i in range(len(y_pred_probs)):
        pred_dezenas = set(select_dezenas_binarias(y_pred_probs[i]))
        true_dezenas = set(np.where(y_val[i] == 1)[0] + 1)
        total_acertos_fold += len(pred_dezenas & true_dezenas)

    media_acertos = total_acertos_fold / len(y_pred_probs)
    media_acertos_folds.append(media_acertos)
    print(f"→ Média de acertos: {media_acertos:.2f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Plot do histórico de perda para cada fold
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Treinamento')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Histórico de Perda - Fold {fold+1}')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True)
    plt.show()

# === 5. Simulação: múltiplas previsões probabilísticas ===
print("\n=== Simulação de 10 apostas (baseadas em probabilidade) ===")

# Entrada mais recente para prever o próximo concurso
entrada_recente = binarios[-sequence_length:].reshape((1, sequence_length, 25))
ensemble_output = np.mean([model.predict(entrada_recente)[0] for model in models], axis=0)
previsao_ensemble = select_dezenas_binarias(ensemble_output)

def simulacao_probabilistica(probs, n_apostas=10, dezenas_por_jogo=15):
    apostas = []
    for _ in range(n_apostas):
        # Amostragem ponderada pelas probabilidades
        indices = np.random.choice(range(25), size=dezenas_por_jogo, replace=False, p=probs/np.sum(probs))
        apostas.append(sorted(indices + 1))
    return apostas

simulacoes = simulacao_probabilistica(ensemble_output, n_apostas=10)
for i, jogo in enumerate(simulacoes, 1):
    print(f"Aposta {i:02d}: {jogo}")

# === 6. Teste retrospectivo: previsão de concursos reais passados ===
print("\n=== Teste Retroativo com Últimos 5 Concursos ===")
num_testes = 5
acertos_totais = []

for i in range(-num_testes - 1, -1):
    entrada = binarios[i - sequence_length:i].reshape((1, sequence_length, 25))
    verdadeira = set(np.where(binarios[i] == 1)[0] + 1)

    pred_media = np.mean([model.predict(entrada)[0] for model in models], axis=0)
    pred_dezenas = set(select_dezenas_binarias(pred_media))
    acertos = len(pred_dezenas & verdadeira)
    acertos_totais.append(acertos)

    print(f"Teste {abs(i)} - Acertos: {acertos:02d} / 15")

# === 7. Resultados ===
print("\n" + "="*80)
print("📊 RESULTADOS DO MODELO - PREVISÕES E ANÁLISES")
print("="*80)

# Dezenas previstas
print("\n✅ Dezenas Previstos (Ensemble Final):")
print("   " + ", ".join(str(int(d)) for d in previsao_ensemble))

# Simulação das apostas
print("\n🎲 Simulação de 10 Apostas com Probabilidade:")
for idx, aposta in enumerate(simulacoes, start=1):
    print(f"   Aposta {idx:02d}: " + ", ".join(str(int(d)) for d in sorted(aposta)))

# Teste retroativo
print("\n📅 Teste Retroativo - Últimos 5 Concursos:")
for i, acertos in enumerate(reversed(acertos_totais), start=1):
    print(f"   Teste {i:02d} - Acertos: {acertos:02d} / 15")

media_retro = np.mean(acertos_totais)
print(f"\n🔍 Média de acertos nos últimos 5 concursos: {media_retro:.2f}")

# Resultados dos folds em tabela
print("\n📈 Desempenho por Fold (Validação Cruzada):")
tabela_folds = []
for i in range(n_splits):
    tabela_folds.append([f"Fold {i+1}", f"{media_acertos_folds[i]:.2f}", f"{mse_folds[i]:.4f}", f"{mae_folds[i]:.4f}"])

print(tabulate(tabela_folds,
               headers=["Fold", "Média Acertos", "MSE", "MAE"],
               tablefmt="fancy_grid"))

# Resumo geral
media_folds = np.mean(media_acertos_folds)
mse_medio = np.mean(mse_folds)
mae_medio = np.mean(mae_folds)

print("\n📋 Resumo Geral:")
print(f"   📊 Média geral de acertos por fold: {media_folds:.2f}")
print(f"   📉 MSE médio: {mse_medio:.4f}")
print(f"   📉 MAE médio: {mae_medio:.4f}")
print("="*80)

# === 8. Gráfico de perda do último fold ===
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss (Treinamento)')
plt.plot(history.history['val_loss'], label='Loss (Validação)')
plt.title('Histórico de Perda - Último Fold')
plt.xlabel('Épocas')
plt.ylabel('Binary Crossentropy')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png', dpi=300)
plt.show()
