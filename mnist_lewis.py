#################### 1ª parte #######################
import numpy as np
from keras.datasets import mnist # usarei apenas para baixar imagens do mnist

# recebendo os dados do mnist
# separando os dados para treino e teste
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

# achatando (vetorizando os pixels) as imagens de 28x28 para 784 e normalizando (/255.0)
# O -1 diz ao Python "mantenha todas as $60.000$ imagens"
# Essas 60.000 imagens são amostras de dígitos numéricos manuscritos (de 0 a 9). 
# Cada uma dessas imagens é formada por uma matriz de 30 por 28 pixels em escala de cinza.
X_train = X_train_raw.reshape(-1, 784) / 255.0
X_test = X_test_raw.reshape(-1, 784) / 255.0

# Transformando os rótulos (y) em One-Hot Encoding
def one_hot(y, num_classes=10):
    matriz_one_hot = np.zeros((y.size, num_classes))
    matriz_one_hot[np.arange(y.size), y] = 1.0
    return matriz_one_hot

y_train = one_hot(y_train_raw)
y_test = one_hot(y_test_raw)

# Checagem até aqui:
print(f"Formato das Imagens de Treino prontas: {X_train.shape}")
print(f"Gabarito da primeira imagem (era {y_train_raw[0]}): {y_train[0]}")

####################### 2ª parte #######################
# ==========================================
# INICIALIZAÇÃO DA ARQUITETURA - arquitetura shallow - sem camada oculta
# ==========================================
print("\n--- Iniciando a Fase 2: Pesos e Biases ---")

# 1. Matriz de Pesos (W): 784 entradas (pixels) x 10 neurônios de saída.
# Inicializados com números aleatórios de uma distribuição Normal (Gaussiana) pequenos.
W = np.random.randn(784, 10) * 0.01

# 2. Vetor de Biases (B): 10 posições (uma para cada neurônio).
# Inicializados com zeros, pois deslocarão a função apenas quando o aprendizado começar.
b = np.zeros(10)

# Verificando se estão no tamanho certo
print(f"Dimensão da Matriz de Pesos (W): {W.shape}")
print(f"Dimensão do Vetor de Bias (b): {b.shape}")

# ==========================================
# FASE 3 E 4: O CICLO DE APRENDIZADO
# ==========================================
print("\n--- Iniciando o Treinamento (100 Épocas) ---")

taxa_aprendizado = 0.5
epocas = 100
m = X_train.shape[0] # Número total de imagens (60.000)

for epoca in range(epocas):
    
    # -----------------------------------------
    # FASE 3: FORWARD PASS (A Predição)
    # -----------------------------------------
    Z = np.dot(X_train, W) + b
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    # -----------------------------------------
    # FASE 4: BACKPROPAGATION E DESCIDA DO GRADIENTE
    # -----------------------------------------
    # 1. Calculando a margem de erro (Predição A menos o Gabarito real)
    dZ = A - y_train
    
    # 2. Backpropagation: Calculando as derivadas parciais
    dW = (1 / m) * np.dot(X_train.T, dZ)
    db = (1 / m) * np.sum(dZ, axis=0)
    
    # 3. Descida do Gradiente: Atualizando Pesos e Biases
    W = W - (taxa_aprendizado * dW)
    b = b - (taxa_aprendizado * db)
    
    # Imprimindo o progresso a cada 10 épocas
    if (epoca + 1) % 10 == 0:
        # Pega o índice da maior probabilidade como a resposta final da rede
        predicoes_corretas = np.argmax(A, axis=1) == y_train_raw
        acuracia = np.mean(predicoes_corretas) * 100
        print(f"Época {epoca + 1}/100 | Acurácia no Treino: {acuracia:.2f}%")