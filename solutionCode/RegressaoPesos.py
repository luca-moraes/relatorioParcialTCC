from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import DataClassFiller as dcf
import json
import PercentualTests as ptest

def weighted_average(input_data, weights):
    return np.dot(input_data, weights)

def normalize_predictions(predictions, min_val, max_val):
    return np.clip(predictions, min_val, max_val)

def write_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def write_to_txt(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(data)
    except Exception as e:
        print(f"An error occurred while writing to {filename}: {e}")

def regressionAndTest(loadedAnswersParamsAll, trainPercent, testPercent, min_val, max_val, headerText, basePath, docName):
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
    normalized_predictions = normalize_predictions(raw_predictions, min_val, max_val)

    mse_before_clipping = np.mean((raw_predictions - output_data) ** 2)
    mae_before_clipping = np.mean(np.abs(raw_predictions - output_data))

    mse_after_clipping = np.mean((normalized_predictions - output_data) ** 2)
    mae_after_clipping = np.mean(np.abs(normalized_predictions - output_data))
    
    testResultsText = ""
    testResultsText += headerText
    testResultsText += "Total de dados: {}\n".format(len(loadedAnswersParamsAll))
    testResultsText += "Quantidade de dados de treino: {}\n".format(len(loadedAnswersParams))
    testResultsText += "Quantidade de dados de teste: {}\n".format(len(loadedAnswersTests))
    
    testResultsText += "Erro quadrático médio (antes da clippagem): {}\n".format(mse_before_clipping)
    testResultsText += "Erro médio absoluto (antes da clippagem): {}\n".format(mae_before_clipping)
    testResultsText += "Pesos otimizados: {}\n".format(optimized_weights)
    testResultsText += "Erro quadrático médio (após clippagem): {}\n".format(mse_after_clipping)
    testResultsText += "Erro médio absoluto (após clippagem): {}\n".format(mae_after_clipping)

    # print("Erro quadrático médio (antes da clippagem):", mse_before_clipping)
    # print("Erro médio absoluto (antes da clippagem):", mae_before_clipping)
    # print("Pesos otimizados:", optimized_weights)
    # print("Erro quadrático médio (após clippagem):", mse_after_clipping)
    # print("Erro médio absoluto (após clippagem):", mae_after_clipping)
     
    ptext, testDic = ptest.test_all(optimized_weights[0], optimized_weights[1], optimized_weights[2], loadedAnswersTests, max_val)
    
    testResultsText += ptext
    
    #write_to_json(testDic, '../normalizedData/esDataset/testsResults.json')
    write_to_txt(testResultsText, basePath + "sintese" + docName)
    write_to_json(testDic, basePath + docName)
    
    
def main():
    
    # ------------- Usando os dados dOS Modelos Base -------------
    
    loadedAnswersParamsBr = dcf.loadAnswersParams()
    loadedAnswersParamsEn = dcf.loadEnAnswersParams()
    loadedAnswersParamsEs = dcf.loadEsAnswersParams()
    
    brPercentsTrain = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    brPercentsTest = [0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
    
    for i in range(len(brPercentsTrain)):
        regressionAndTest(loadedAnswersParamsBr, brPercentsTrain[i], brPercentsTest[i], 0, 3, "Português Base Percentual Treino com {}\n".format(brPercentsTrain[i]), '../normalizedData/ptbrDataset/', 'testsResults{}.json'.format(i))
        
    enPercentsTrain = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    enPercentsTest = [0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
    
    for i in range(len(enPercentsTrain)):
        regressionAndTest(loadedAnswersParamsEn, enPercentsTrain[i], enPercentsTest[i], 0, 5, "Inglês Base Percentual Treino com {}\n".format(enPercentsTrain[i]), '../normalizedData/enDataset/', 'testsResults{}.json'.format(i))
        
    esPercentsTrain = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    esPercentsTest = [0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
    
    for i in range(len(esPercentsTrain)):
        regressionAndTest(loadedAnswersParamsEs, esPercentsTrain[i], esPercentsTest[i], 0, 5, "Espanhol Base Percentual Treino com {}\n".format(enPercentsTrain[i]), '../normalizedData/esDataset/', 'testsResults{}.json'.format(i))
        
    # ------------- Usando os dados dos Modelos Large -------------
        
    loadedAnswersParamsLargeBr = dcf.loadAnswersParamsLarge()
    loadedAnswersParamsLargeEn = dcf.loadEnAnswersParamsLarge()
    loadedAnswersParamsLargeEs = dcf.loadEsAnswersParamsLarge()
    
    brLargePercentsTrain = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    brLargePercentsTest = [0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
    
    for i in range(len(brLargePercentsTrain)):
        regressionAndTest(loadedAnswersParamsLargeBr, brLargePercentsTrain[i], brLargePercentsTest[i], 0, 3, "Português Large Percentual Treino com {}\n".format(brLargePercentsTrain[i]), '../normalizedData/ptbrDataset/', 'testsResultsLarge{}.json'.format(i))
        
    enLargePercentsTrain = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    enLargePercentsTest = [0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
    
    for i in range(len(enLargePercentsTrain)):
        regressionAndTest(loadedAnswersParamsLargeEn, enLargePercentsTrain[i], enLargePercentsTest[i], 0, 5, "Inglês Large Percentual Treino com {}\n".format(enLargePercentsTrain[i]), '../normalizedData/enDataset/', 'testsResultsLarge{}.json'.format(i))
        
    esLargePercentsTrain = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    esLargePercentsTest = [0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
    
    for i in range(len(esLargePercentsTrain)):
        regressionAndTest(loadedAnswersParamsLargeEs, esLargePercentsTrain[i], esLargePercentsTest[i], 0, 5, "Espanhol Large Percentual Treino com {}\n".format(esLargePercentsTrain[i]), '../normalizedData/esDataset/', 'testsResultsLarge{}.json'.format(i))
        
    # ------------- Usando os dados do Modelo Roberta -------------
        
    loadedAnswersParamsEsRoberta = dcf.loadEsAnswersParamsBerta()
    
    esRobertaPercentsTrain = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    esRobertaPercentsTest = [0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
    
    for i in range(len(esRobertaPercentsTrain)):
        regressionAndTest(loadedAnswersParamsEsRoberta, esRobertaPercentsTrain[i], esRobertaPercentsTest[i], 0, 5, "Espanhol Roberta Percentual Treino com {}\n".format(esRobertaPercentsTrain[i]), '../normalizedData/esDataset/', 'testsResultsRoberta{}.json'.format(i))
    
    
if __name__ == "__main__":
    main()