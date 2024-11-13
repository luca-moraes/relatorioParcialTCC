import numpy as np
import json
from dataclasses import asdict
from Models import Question, Answer, RefResponse, Keywords, AnswerParams, AnswerTestParams

def write_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def calculate_general_error(answer_tests):
    total_error = 0
    num_answers = len(answer_tests)
    
    quantidade_erros_por_faixa = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for answer_test in answer_tests:
        total_error += answer_test.percentual_error
        
        if answer_test.percentual_error <= 0.1:
            quantidade_erros_por_faixa[0] += 1
        elif 0.1 < answer_test.percentual_error <= 0.2:
            quantidade_erros_por_faixa[1] += 1
        elif 0.2 < answer_test.percentual_error <= 0.3:
            quantidade_erros_por_faixa[2] += 1
        elif 0.3 < answer_test.percentual_error <= 0.4:
            quantidade_erros_por_faixa[3] += 1
        elif 0.4 < answer_test.percentual_error <= 0.5:
            quantidade_erros_por_faixa[4] += 1
        elif 0.5 < answer_test.percentual_error <= 0.6:
            quantidade_erros_por_faixa[5] += 1
        elif 0.6 < answer_test.percentual_error <= 0.7:
            quantidade_erros_por_faixa[6] += 1
        elif 0.7 < answer_test.percentual_error <= 0.8:
            quantidade_erros_por_faixa[7] += 1
        elif 0.8 < answer_test.percentual_error <= 0.9:
            quantidade_erros_por_faixa[8] += 1
        elif 0.9 < answer_test.percentual_error <= 1.0:
            quantidade_erros_por_faixa[9] += 1

    general_error = total_error / num_answers
    
    values = ""
    
    values += "Numero de Testes: {}\n".format(num_answers)
    values += "Erro Geral do Algoritmo: {:.2%}\n".format(general_error)
    # values += "Numero de Erros Graves: {}\n".format(highest_error)
    # values += "Porcentagem de Erros Graves: {:.2%}\n".format(highest_error / num_answers)
    values += "Erro por faixa de porcentagem:\n"
    values += "0-10% Quantidade: {} - Percentual: {}\n".format(quantidade_erros_por_faixa[0], quantidade_erros_por_faixa[0] / num_answers)
    values += "10-20% Quantidade: {} - Percentual: {}\n".format(quantidade_erros_por_faixa[1], quantidade_erros_por_faixa[1] / num_answers)
    values += "20-30% Quantidade: {} - Percentual: {}\n".format(quantidade_erros_por_faixa[2], quantidade_erros_por_faixa[2] / num_answers)
    values += "30-40% Quantidade: {} - Percentual: {}\n".format(quantidade_erros_por_faixa[3], quantidade_erros_por_faixa[3] / num_answers)
    values += "40-50% Quantidade: {} - Percentual: {}\n".format(quantidade_erros_por_faixa[4], quantidade_erros_por_faixa[4] / num_answers)
    values += "50-60% Quantidade: {} - Percentual: {}\n".format(quantidade_erros_por_faixa[5], quantidade_erros_por_faixa[5] / num_answers)
    values += "60-70% Quantidade: {} - Percentual: {}\n".format(quantidade_erros_por_faixa[6], quantidade_erros_por_faixa[6] / num_answers)
    values += "70-80% Quantidade: {} - Percentual: {}\n".format(quantidade_erros_por_faixa[7], quantidade_erros_por_faixa[7] / num_answers)
    values += "80-90% Quantidade: {} - Percentual: {}\n".format(quantidade_erros_por_faixa[8], quantidade_erros_por_faixa[8] / num_answers)
    values += "90-100% Quantidade: {} - Percentual: {}\n".format(quantidade_erros_por_faixa[9], quantidade_erros_por_faixa[9] / num_answers)
    
    # print("Numero de Testes: {}".format(num_answers))
    # print("Erro Geral do Algoritmo: {:.2%}".format(general_error))
    # print("Numero de Erros Graves: {}".format(highest_error))
    # print("Porcentagem de Erros Graves: {:.2%}".format(highest_error / num_answers))   
    
    return values 

def test_all(peso1, peso2, peso3, answers, max_val):
    answerTestsList = []

    for answer in answers:
        fator1 = peso1 * answer.frequence_similarity
        fator2 = peso2 * answer.liv_distance
        fator3 = peso3 * answer.bert_score

        soma = fator1 + fator2 + fator3

        peso = peso1 + peso2 + peso3

        nota = soma / peso

        error = 0
        if(answer.answer_values.grade == 0):
            #error = nota / 3
            error = nota / max_val
        else:
            # percentual = nota / answer.answer_values.grade
            percentual = (min(nota, answer.answer_values.grade) / max(nota, answer.answer_values.grade))
            error = 1 - percentual

        answerTestsList.append(AnswerTestParams(answer.answer_number,
            answer.answer_values,
            answer.frequence_similarity,
            answer.liv_distance,
            answer.bert_score,
            nota,
            error
        ))
    
    text = calculate_general_error(answerTestsList)
    
    answerTestsDict = [asdict(answerTests) for answerTests in answerTestsList]
    #write_to_json(answerTestsDict, '../normalizedData/esDataset/testsResults.json')
    
    return text, answerTestsDict
                
    # return nota
    #return normalize(nota, 0, 3, 0, 10)