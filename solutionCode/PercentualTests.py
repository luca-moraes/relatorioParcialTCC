import numpy as np
import json
from dataclasses import asdict
from Models import Question, Answer, RefResponse, Keywords, AnswerParams, AnswerTestParams

def write_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def test_all(peso1, peso2, peso3, answers):
    answerTestsList = []

    for answer in answers:
        fator1 = peso1 * answer.frequence_similarity
        fator2 = peso2 * answer.liv_distance
        fator3 = peso3 * answer.bert_score

        soma = fator1 + fator2 + fator3

        peso = peso1 + peso2 + peso3

        nota = soma / peso

        percentual = 0
        if(answer.answer_values.grade == 0):
            percentual = 
        else:
            percentual = nota / answer.answer_values.grade

        error = 1 - percentual

        answerTestsList.append(AnswerTestParams(answer.answer_number,
                answer.answer_values,
                answer.frequence_similarity,
                answer.liv_distance,
                answer.bert_score,
                nota,
                error
                ))
    
    answerTestsDict = [asdict(answerTests) for answerTests in answerTestsList]
    write_to_json(answerTestsDict, '../normalizedData/ptbrDataset/testsResults.json')
                
    # return nota
    #return normalize(nota, 0, 3, 0, 10)