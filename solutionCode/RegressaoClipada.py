from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import DataClassFiller as dcf

loadedAnswersParams = dcf.loadAnswersParams()

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

predictions_normalized = model.predict(input_data_normalized)
predictions = scaler_output.inverse_transform(predictions_normalized.reshape(-1, 1)).flatten()

mse = np.mean((predictions - output_data) ** 2)
mae = np.mean(np.abs(predictions - output_data))

print("Erro quadrático médio:", mse)
print("Erro médio absoluto:", mae)
print("Pesos otimizados:", model.coef_)

predictions_clipped = np.clip(predictions, 0, 3)

mse_clipped = np.mean((predictions_clipped - output_data) ** 2)
mae_clipped = np.mean(np.abs(predictions_clipped - output_data))

print("Erro quadrático médio (após clippagem):", mse_clipped)
print("Erro médio absoluto (após clippagem):", mae_clipped)