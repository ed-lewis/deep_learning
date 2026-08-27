import numpy as np
from keras.datasets import mnist # Usado apenas para baixar as imagens do MNIST

# ==========================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO

print("Baixando e carregando o dataset MNIST...")
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

# A imagem original tem tamanho 28x28. 
# Precisamos vetorizar (achatar) a matriz para um vetor de 784 posições.
# Também normalizamos os pixels dividindo por 255.0 para que fiquem entre 0 e 1.
X_train = X_train_raw.reshape(-1, 784) / 255.0
X_test = X_test_raw.reshape(-1, 784) / 255.0

# O gabarito (y) vem como um número (ex: 5). 
# Precisamos transformá-lo em One-Hot Encoding (ex: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
def one_hot(y, num_classes=10):
    matriz_one_hot = np.zeros((y.size, num_classes))
    matriz_one_hot[np.arange(y.size), y] = 1.0
    return matriz_one_hot

y_train = one_hot(y_train_raw)
y_test = one_hot(y_test_raw)

# ==========================================
# 2. FUNÇÕES DE ATIVAÇÃO
# ==========================================
def relu(Z):
    """
    Função ReLU: Retorna Z se Z for positivo, senão retorna 0.
    Evita o problema do gradiente nulo (Vanishing Gradient).
    """
    return np.maximum(0, Z)

def deriv_relu(Z):
    """
    Derivada da ReLU: Retorna 1 se Z > 0, senão 0.
    Essencial para a Regra da Cadeia no Backpropagation.
    """
    return Z > 0

def softmax(Z):
    """
    Função Softmax: Transforma as saídas brutas em probabilidades (soma = 1).
    A subtração de np.max(Z) evita estouro de memória (estabilidade numérica).
    """
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# ==========================================
# 3. INICIALIZAÇÃO DA ARQUITETURA DA REDE
# ==========================================
tamanho_entrada = 784  # 28x28 pixels vetorizados
tamanho_oculta = 64    # Número de neurônios na camada intermediária
tamanho_saida = 10     # 10 possibilidades (dígitos de 0 a 9)
taxa_aprendizado = 0.1 # O passo de aprendizagem da descida do gradiente (eta)

# Inicializando Pesos (W) aleatoriamente e Biases (b) com zeros
np.random.seed(42)
W1 = np.random.randn(tamanho_entrada, tamanho_oculta) * 0.1
b1 = np.zeros((1, tamanho_oculta))

W2 = np.random.randn(tamanho_oculta, tamanho_saida) * 0.1
b2 = np.zeros((1, tamanho_saida))

# ==========================================
# 4. O LOOP DE TREINAMENTO (ÉPOCAS E BATCHES)
# ==========================================
epocas = 15
tamanho_batch = 64 # Processaremos 64 imagens por vez
num_amostras = X_train.shape[0]

print("\nIniciando o treinamento da rede neural na mão...")

for epoca in range(epocas):
    
    # Embaralhar as imagens a cada época melhora o aprendizado
    indices = np.random.permutation(num_amostras)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Treinamento em Mini-Batches (lotes)
    for i in range(0, num_amostras, tamanho_batch):
        X_batch = X_shuffled[i:i + tamanho_batch]
        y_batch = y_shuffled[i:i + tamanho_batch]
        m = X_batch.shape[0] # Quantidade real de imagens no batch (geralmente 64)
        
        # ----------------------------------------
        # PASSO A: FORWARD PROPAGATION (A IDA)
        # ----------------------------------------
        # 1. Estímulo da Camada Oculta e Ativação ReLU
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)
        
        # 2. Estímulo da Camada de Saída e Ativação Softmax
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2) # Esta é a predição da rede (probabilidades)
        
        # ----------------------------------------
        # PASSO B: BACKPROPAGATION (A VOLTA)
        # ----------------------------------------
        # O erro na camada de saída (Derivada combinada de Cross-Entropy com Softmax)
        dZ2 = A2 - y_batch
        
        # Gradientes da Camada de Saída (Peso 2 e Bias 2)
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Propagando o erro para a Camada Oculta (multiplicando pela Transposta de W2)
        dA1 = np.dot(dZ2, W2.T)
        
        # Gradiente passando pela função de ativação ReLU (só passa se Z1 > 0)
        dZ1 = dA1 * deriv_relu(Z1)
        
        # Gradientes da Camada Oculta (Peso 1 e Bias 1)
        dW1 = (1 / m) * np.dot(X_batch.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # ----------------------------------------
        # PASSO C: ATUALIZAÇÃO DOS PARÂMETROS
        # ----------------------------------------
        # Aplicando a descida do gradiente: Peso_novo = Peso_antigo - (taxa * gradiente)
        W2 -= taxa_aprendizado * dW2
        b2 -= taxa_aprendizado * db2
        W1 -= taxa_aprendizado * dW1
        b1 -= taxa_aprendizado * db1
        
    # --- AVALIAÇÃO DA ÉPOCA ---
    # Ao final de cada época, testamos a rede no conjunto de Teste (10.000 imagens que ela nunca viu)
    Z1_teste = np.dot(X_test, W1) + b1
    A1_teste = relu(Z1_teste)
    Z2_teste = np.dot(A1_teste, W2) + b2
    A2_teste = softmax(Z2_teste)
    
    # A resposta final da rede é o neurônio com a maior probabilidade (np.argmax)
    predicoes = np.argmax(A2_teste, axis=1)
    acertos = np.sum(predicoes == y_test_raw)
    acuracia = acertos / len(y_test_raw)
    
    print(f"Época {epoca+1:02d}/{epocas} | Acurácia no Teste: {acuracia * 100:.2f}%")

print("\nTreinamento concluído com sucesso!")