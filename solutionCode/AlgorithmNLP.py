from sklearn.linear_model import LinearRegression
import numpy as np

# Dados de entrada e saída (exemplo)
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([5, 7, 9])

# Inicialização dos pesos
weights = np.array([0.5, 0.5])  # Exemplo de inicialização

# Implementação da média ponderada
def weighted_average(X, weights):
    return np.dot(X, weights)

# Regressão Linear
model = LinearRegression()
model.fit(X, y)

# Obtendo os pesos otimizados
optimized_weights = model.coef_

# Avaliação do modelo
predictions = weighted_average(X, optimized_weights)
mse = np.mean((predictions - y) ** 2)# erro quadratico medio
mae = np.mean(np.abs(predictions - y))# erro absoluto medio

print("Erro quadrático médio:", mse)
print("Erro médio absoluto:", mae)
print("Pesos otimizados:", optimized_weights)
