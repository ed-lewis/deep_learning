#################### 1ª parte #######################
import numpy as np
from keras.datasets import mnist # usarei apenas para baixar imagens do mnist

# recebendo os dados do mnist
# separando os dados para treino e teste
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

# achatando (vetorizando os pixels) as imagens de 28x28 para 784 e normalizando (/255.0)
# O -1 diz ao Python "mantenha todas as $60.000$ imagens"
# Essas 60.000 imagens são amostras de dígitos numéricos manuscritos (de 0 a 9). 
# Cada uma dessas imagens é formada por uma matriz de 28 por 28 pixels em escala de cinza.
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
# INICIALIZAÇÃO DA ARQUITETURA
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