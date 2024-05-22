from sklearn.linear_model import LinearRegression
from Models import Question, Answer, RefResponse, Keywords, AnswerParams
import numpy as np
import DataClassFiller as dcf
#import solutionCode.DataHandlersPtbr.DataClassFiller as dcf

#questions = dcf.loadQuestions()

#for question in questions:
#    print(str(question.number_question) + " " + question.question_text)

loadedAnswersParams = dcf.loadAnswersParams()
#quest = dcf.loadQuestions()

# Dados de entrada e saída (exemplo)
# X = np.array([[1, 2], [2, 3], [3, 4]])
# y = np.array([5, 7, 9])

input_data = []
output_data = []

for answer in loadedAnswersParams:
    input_data.append([answer.frequence_similarity, answer.liv_distance])
    output_data.append(answer.answer_values.grade)

# Inicialização dos pesos
weights = np.array([0.5, 0.5])  # Exemplo de inicialização

# Implementação da média ponderada
# def weighted_average(X, weights):
#     return np.dot(X, weights)

def weighted_average(input_data, weights):
    return np.dot(input_data, weights)

# Regressão Linear
model = LinearRegression()
# model.fit(X, y)
model.fit(input_data, output_data)

# Obtendo os pesos otimizados
optimized_weights = model.coef_

# Avaliação do modelo
# predictions = weighted_average(X, optimized_weights)
# mse = np.mean((predictions - y) ** 2)# erro quadratico medio
# mae = np.mean(np.abs(predictions - y))# erro absoluto medio

predictions = weighted_average(input_data, optimized_weights)
mse = np.mean((predictions - output_data) ** 2)# erro quadratico medio
mae = np.mean(np.abs(predictions - output_data))# erro absoluto medio

print("Erro quadrático médio:", mse)
print("Erro médio absoluto:", mae)
print("Pesos otimizados:", optimized_weights)