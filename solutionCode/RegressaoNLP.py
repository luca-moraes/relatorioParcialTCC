from sklearn.linear_model import LinearRegression
from Models import Question, Answer, RefResponse, Keywords, AnswerParams
import numpy as np
import DataClassFiller as dcf

loadedAnswersParams = dcf.loadAnswersParams()

input_data = []
output_data = []

for answer in loadedAnswersParams:
    input_data.append([answer.frequence_similarity, answer.liv_distance, answer.bert_score])
    output_data.append(answer.answer_values.grade)

weights = np.array([1/3, 1/3, 1/3])

def weighted_average(input_data, weights):
    return np.dot(input_data, weights)

model = LinearRegression()
model.fit(input_data, output_data)

optimized_weights = model.coef_

predictions = weighted_average(input_data, optimized_weights)
mse = np.mean((predictions - output_data) ** 2)
mae = np.mean(np.abs(predictions - output_data))

print("Erro quadrático médio:", mse)
print("Erro médio absoluto:", mae)
print("Pesos otimizados:", optimized_weights)