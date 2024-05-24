from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import DataClassFiller as dcf

import PercentualTests as ptest

def weighted_average(input_data, weights):
    return np.dot(input_data, weights)

def normalize_predictions(predictions, min_val=0, max_val=3):
    return np.clip(predictions, min_val, max_val)

#loadedAnswersParamsAll = dcf.loadAnswersParams()
loadedAnswersParamsAll = dcf.loadEnAnswersParams()

trainPercent = 0.7
testPercent = 0.7

loadedAnswersParams = loadedAnswersParamsAll[:int(trainPercent * len(loadedAnswersParamsAll))]

loadedAnswersTests = loadedAnswersParamsAll[int(testPercent * len(loadedAnswersParamsAll)):]

input_data = []
output_data = []

for answer in loadedAnswersParams:
    input_data.append([answer.frequence_similarity, answer.liv_distance, answer.bert_score])
    output_data.append(answer.answer_values.grade)

input_data = np.array(input_data)
output_data = np.array(output_data)

scaler_input = MinMaxScaler(feature_range=(0, 1))
input_data_normalized = scaler_input.fit_transform(input_data)

scaler_output = MinMaxScaler(feature_range=(0, 1))
output_data_normalized = scaler_output.fit_transform(output_data.reshape(-1, 1)).flatten()

model = LinearRegression()
model.fit(input_data_normalized, output_data_normalized)

optimized_weights = model.coef_

raw_predictions_normalized = model.predict(input_data_normalized)
raw_predictions = scaler_output.inverse_transform(raw_predictions_normalized.reshape(-1, 1)).flatten()

#normalized_predictions = normalize_predictions(raw_predictions, 0, 3)
normalized_predictions = normalize_predictions(raw_predictions, 0, 5)

mse_before_clipping = np.mean((raw_predictions - output_data) ** 2)
mae_before_clipping = np.mean(np.abs(raw_predictions - output_data))

mse_after_clipping = np.mean((normalized_predictions - output_data) ** 2)
mae_after_clipping = np.mean(np.abs(normalized_predictions - output_data))

print("Erro quadrático médio (antes da clippagem):", mse_before_clipping)
print("Erro médio absoluto (antes da clippagem):", mae_before_clipping)
print("Pesos otimizados:", optimized_weights)

print("Erro quadrático médio (após clippagem):", mse_after_clipping)
print("Erro médio absoluto (após clippagem):", mae_after_clipping)

ptest.test_all(optimized_weights[0], optimized_weights[1], optimized_weights[2], loadedAnswersTests)