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

    for answer_test in answer_tests:
        total_error += answer_test.percentual_error

    general_error = total_error / num_answers
    
    print("Numero de Testes: {}".format(num_answers))
    print("Erro Geral do Algoritmo: {:.2%}".format(general_error))


def test_all(peso1, peso2, peso3, answers):
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
            error = nota / 3
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
    
    calculate_general_error(answerTestsList)
    
    answerTestsDict = [asdict(answerTests) for answerTests in answerTestsList]
    write_to_json(answerTestsDict, '../normalizedData/ptbrDataset/testsResults.json')
                
    # return nota
    #return normalize(nota, 0, 3, 0, 10)