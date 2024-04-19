from dataclasses import dataclass
from typing import List
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from Levenshtein import distance

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Adicione essas importações no início do seu script
from dataclasses import dataclass
from typing import List
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from Levenshtein import distance

@dataclass
class Answer:
    #identification_student: int
    number_question: int
    answer_question: str
    grade: int

@dataclass
class RefResponse:
    number_question: int
    reference_response: str

@dataclass
class Question:
    number_question: int
    question_text: str
    reference_responses: List[RefResponse]
    responses_students: List[Answer]

def read_csv(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append(row)
    return data

# def preprocess_data(data):
#     questions = {}
#     for row in data:
#         question_id = int(row[0])
#         question_text = row[1]
#         if question_id not in questions:
#             questions[question_id] = Question(number_question=question_id, reference_response="", responses_students=[])
#         if len(row) == 2:
#             if question_id not in questions:
#                 questions[question_id].reference_response = question_text
#         else:
#             answer = Answer(identification_student=len(questions[question_id].responses_students) + 1,
#                             number_question=question_id,
#                             answer_question=question_text,
#                             grade=int(row[2]))
#             questions[question_id].responses_students.append(answer)
#     return list(questions.values())

def calculate_cosine_similarity(reference_response, student_responses):
    vectorizer = CountVectorizer().fit([reference_response] + student_responses)
    vectors = vectorizer.transform([reference_response] + student_responses)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0][1:]

def main():
    # distance("pitom", "python")

    questions_data = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/questions.csv')
    reference_responses_data = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/reference_answers.csv')
    students_responses_data = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/student_answers_and_grades_v1.csv')

    # questions = preprocess_data(questions_data)
    # reference_responses = {}
    # for row in reference_responses_data:
    #     question_id = int(row[0])
    #     if question_id not in reference_responses:
    #         reference_responses[question_id] = row[1]

    questionList = []
    for row in questions_data:
        question_id = int(row[0])
        question_text = row[1]
        questionList.append(Question(number_question=question_id, question_text=question_text, reference_responses=[], responses_students=[]))

    referenceResponsesList = []
    for row in reference_responses_data:
        question_id = int(row[0])
        reference_response_text = row[1]
        referenceResponsesList.append(RefResponse(number_question=question_id, reference_response=reference_response_text))

    studentsResponseList = []
    for row in students_responses_data:
        question_id = int(row[0])
        studentsResponseList.append(Answer(number_question=question_id,
                        answer_question=row[1],
                        grade=int(row[2])))
    
    # for row in students_responses_data:
    #     question_id = int(row[0])
    #     if question_id in questions:
    #         question = questions[question_id]
    #         answer = Answer(identification_student=len(question.responses_students) + 1,
    #                         number_question=question_id,
    #                         answer_question=row[1],
    #                         grade=int(row[2]))
    #         question.responses_students.append(answer)
    #     else:
    #         print(f"Questão {question_id} não listada.")
        
    for question in questionList:
        for reference in referenceResponsesList:
            if question.number_question == reference.number_question:
                question.reference_responses.append(reference)

    for question in questionList:
        for student in studentsResponseList:
            if question.number_question == student.number_question:
                question.responses_students.append(student)

    # for question in questions:
    #     if question.number_question in reference_responses:
    #         reference_response = reference_responses[question.number_question]
    #         student_responses = [student_response.answer_question for student_response in question.responses_students]
    #         similarities = calculate_cosine_similarity(reference_response, student_responses)
    #         print(f"Questao {question.number_question}:")
    #         for i, similarity in enumerate(similarities):
    #             print(f"  Aluno {question.responses_students[i].identification_student} - Nota: {question.responses_students[i].grade}, Sim: {similarity:.4f}")
    #     else:
    #         print(f"Resposta modelo da {question.number_question} nao listada.")

    # for question in questionList:
    #     for answer in question.responses_students:
    #         maxSimilarity = -999
    #         minLivDistance = 999
    #         mediaGrade = 0
    #         finalRef = ""
    #         for reference in question.reference_responses:
    #             similarity = calculate_cosine_similarity(reference.reference_response, [answer.answer_question])
    #             livDistance = distance(answer.answer_question, reference.reference_response)

    #             normalizedSimilarity = 0 + ((similarity - 1) * (3 - 0)) / (1 - 0)

    #             # normalizedDistance = MaxAbsScaler().fit_transform(np.array([livDistance]).reshape(-1, 1))
    #             normalizedDistance = len(reference.reference_response) - livDistance

    #             peso1 = 0.5
    #             peso2 = 0.5

    #             maxMediaGrade = (normalizedDistance*peso1 + normalizedSimilarity[0]*peso2)/(peso1+peso2)

    #             if maxMediaGrade > mediaGrade:
    #                 mediaGrade = maxMediaGrade
    #                 maxSimilarity = similarity
    #                 minLivDistance = livDistance
    #                 finalRef = reference.reference_response
                
    #         print(f"Questao {question.number_question}:")
    #         print(f"  Aluno x - Nota: {answer.grade}, Sim: {maxSimilarity[0]}, Dist: {minLivDistance}, Grade: {mediaGrade}")
    #         print("Resposta do aluno: ", answer.answer_question + "\n Resposta do professor: " + finalRef + "\n")
                
    X = []  # Entradas: [maxSimilarity, minLivDistance]
    y = []  # Saídas: notas reais (grades)
    
    for question in questionList:
        for answer in question.responses_students:
            features_per_answer = []
            grades_per_answer = []
            for reference in question.reference_responses:
                similarity = calculate_cosine_similarity(reference.reference_response, [answer.answer_question])
                livDistance = distance(answer.answer_question, reference.reference_response)

                #normalizedSimilarity = similarity[0]
                normalizedSimilarity = 0 + ((similarity - 1) * (3 - 0)) / (1 - 0)
                #normalizedDistance = livDistance
                normalizedDistance = len(reference.reference_response) - livDistance

                features_per_answer.append([normalizedSimilarity, normalizedDistance])
                grades_per_answer.append(answer.grade)
            
            if features_per_answer:
                X.append(features_per_answer[0])
                y.append(grades_per_answer[0])
    
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Erro quad mediio (MSE) no conjunto de teste: {mse}")
    
    # Aplicar os pesos otimizados (model.coef_) para calcular a mediaGrade otimizada
    # Os coeficientes são os pesos otimizados para cada feature
    print("Pesos otimizados final:", model.coef_)
    
    # Opcionalmente, você pode aplicar os pesos otimizados para recalcular mediaGrade para cada entrada
    # e comparar com as notas reais.
                
if __name__ == "__main__":
    main()