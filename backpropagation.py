import numpy as np

# 1. Função de Ativação e sua Derivada
# A sigmoide comprime os valores entre 0 e 1.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# A derivada da sigmoide é fundamental para o backpropagation (Regra da Cadeia)
def sigmoid_derivative(x):
    return x * (1 - x) # Nota: aqui 'x' já é o valor que passou pela sigmoide no forward pass

# 2. Configurando o Dataset (Problema XOR)
# Entradas (4 exemplos, 2 variáveis)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Saídas esperadas (Target)
y = np.array([[0],
              [1],
              [1],
              [0]])

# 3. Inicialização dos Pesos e Biases da Rede
np.random.seed(42) # Semente fixa para podermos reproduzir o mesmo resultado

# Arquitetura: 2 neurônios na entrada, 2 na camada oculta, 1 na saída
tamanho_entrada = 2
tamanho_oculta = 2
tamanho_saida = 1
taxa_aprendizado = 0.1 # O quão rápido a rede aprende (Learning Rate)

# Pesos da Entrada para a Camada Oculta (Matriz 2x2)
W_oculta = np.random.uniform(size=(tamanho_entrada, tamanho_oculta))
bias_oculta = np.random.uniform(size=(1, tamanho_oculta))

# Pesos da Camada Oculta para a Saída (Matriz 2x1)
W_saida = np.random.uniform(size=(tamanho_oculta, tamanho_saida))
bias_saida = np.random.uniform(size=(1, tamanho_saida))

# 4. O Loop de Treinamento (Épocas)
epocas = 10000

for epoca in range(epocas):
    
    # --- PASSO 1: FORWARD PASS ---
    # Multiplica entrada pelos pesos e soma o bias
    entrada_oculta = np.dot(X, W_oculta) + bias_oculta
    # Passa pela função de ativação
    saida_oculta = sigmoid(entrada_oculta)
    
    # Repete o processo para a camada de saída
    entrada_saida = np.dot(saida_oculta, W_saida) + bias_saida
    previsao_final = sigmoid(entrada_saida)
    
    # --- PASSO 2: CÁLCULO DO ERRO (LOSS) ---
    erro = y - previsao_final
    
    # --- PASSO 3: BACKWARD PASS (RETROPROPAGAÇÃO) ---
    # Calculando os gradientes da camada de saída (Erro * Derivada da Ativação)
    d_saida = erro * sigmoid_derivative(previsao_final)
    
    # Calculando a contribuição da camada oculta para o erro (Retropropagando o erro)
    erro_oculta = d_saida.dot(W_saida.T) # .T é a matriz transposta
    
    # Calculando os gradientes da camada oculta
    d_oculta = erro_oculta * sigmoid_derivative(saida_oculta)
    
    # --- PASSO 4: ATUALIZAÇÃO DOS PESOS (Gradient Descent) ---
    W_saida += saida_oculta.T.dot(d_saida) * taxa_aprendizado
    bias_saida += np.sum(d_saida, axis=0, keepdims=True) * taxa_aprendizado
    
    W_oculta += X.T.dot(d_oculta) * taxa_aprendizado
    bias_oculta += np.sum(d_oculta, axis=0, keepdims=True) * taxa_aprendizado

    # Apenas para acompanharmos o progresso a cada 2000 épocas
    if epoca % 2000 == 0:
        perda_media = np.mean(np.abs(erro))
        print(f"Época {epoca} | Erro Médio: {perda_media:.4f}")

# Testando a rede após o treinamento
print("\nPrevisões finais após o treinamento:")
print(previsao_final)
