import json
import numpy as np
import DataClassFiller as dcf
from dataclasses import asdict
from Models import Question, Answer, RefResponse, Keywords, AnswerParams
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from Levenshtein import distance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def write_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def calculate_cosine_similarity(reference_response, student_responses):
    vectorizer = CountVectorizer().fit([reference_response] + student_responses)
    vectors = vectorizer.transform([reference_response] + student_responses)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0][1:]

def main():
    questionList = dcf.loadQuestions()
    
    answersNum = 0
    answerParamsList = []
    for question in questionList:
        for answer in question.responses_students:
            similarities = []
            livDistances = []
            
            for reference in question.reference_responses:
                similarity = calculate_cosine_similarity(reference.reference_response, [answer.answer_question])
                livDistance = distance(answer.answer_question, reference.reference_response)
                
                similarities.append(similarity[0])
                livDistances.append(livDistance)
            
            answerParamsList.append(AnswerParams(answersNum,
                answer,
                max(similarities),
                min(livDistances)))
            answersNum += 1
    
    answerParamsDict = [asdict(answerParams) for answerParams in answerParamsList]
    write_to_json(answerParamsDict, './normalizedData/ptbrDataset/answersParams.json')
                
if __name__ == "__main__":
    main()