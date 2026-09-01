import numpy as np
import time

# Captura o tempo atual em segundos desde a época e multiplica por 1000
# O int() remove as casas decimais, pois a semente exige um número inteiro
semente_ms = int(time.time() * 1000)
# Injeta os milissegundos do relógio como semente do gerador do NumPy
np.random.seed(semente_ms)


# ==========================================
# 1. FUNÇÕES AUXILIARES
# ==========================================

# Função da Fase 1: Transforma o gabarito no formato de probabilidades
def one_hot(y, num_classes=10):
    matriz = np.zeros((y.size, num_classes))
    matriz[np.arange(y.size), y] = 1.0
    return matriz

# Função da Fase 3: Transforma a saída linear em porcentagens de 0 a 1
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# ==========================================
# 2. FUNÇÃO PRINCIPAL DE TREINAMENTO (FASE 3 e 4)
# ==========================================
def treinar_rede(X, y, W, b, taxa_aprendizado=0.5, epocas=100):
    # A variável matriz_one_hot nasce aqui dentro e não se perde, pois é usada no mesmo escopo
    matriz_one_hot = one_hot(y)
    m = X.shape[0] # Número de imagens
    
    for epoca in range(epocas):
        # Forward Pass: Y = f(X*W + B)
        Z = np.dot(X, W) + b
        A = softmax(Z)
        
        # Backpropagation: Derivadas parciais
        dZ = A - matriz_one_hot
        dW = (1 / m) * np.dot(X.T, dZ)
        db = (1 / m) * np.sum(dZ, axis=0)
        
        # Descida do Gradiente: Atualização
        W = W - (taxa_aprendizado * dW)
        b = b - (taxa_aprendizado * db)
        
        # Log de acompanhamento
        if (epoca + 1) % 10 == 0:
            predicoes_corretas = np.argmax(A, axis=1) == y
            acuracia = np.mean(predicoes_corretas) * 100
            print(f"Época {epoca + 1}/{epocas} | Acurácia no Treino: {acuracia:.2f}%")
            
    # CRÍTICO: Devolve as matrizes com os pesos atualizados (aprendidos) para o escopo global
    return W, b

# ==========================================
# 3. EXECUÇÃO DO PROGRAMA
# ==========================================
print("\n--- Iniciando o Treinamento Modular ---")
from keras.datasets import mnist

# Carregando os dados no escopo global deste script
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()
X_train = X_train_raw.reshape(-1, 784) / 255.0


# (Assumindo que X_train e y já existem da sua Fase 1)
# Fase 2: Criamos as matrizes aleatórias iniciais
W_inicial = np.random.randn(784, 10) * 0.01
b_inicial = np.zeros(10)

# Passamos as matrizes brutas para a função e recebemos o "cérebro" treinado de volta
W_treinado, b_treinado = treinar_rede(X_train, y_train_raw, W_inicial, b_inicial)

print("\nTreinamento concluído! Os pesos e biases ajustados estão salvos na memória global.")