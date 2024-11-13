import numpy as np
import DataClassFiller as dcf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from Levenshtein import distance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def calculate_cosine_similarity(reference_response, student_responses):
    vectorizer = CountVectorizer().fit([reference_response] + student_responses)
    vectors = vectorizer.transform([reference_response] + student_responses)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0][1:]

def main():
    # distance("pitom", "python")

    questionList = dcf.loadQuestions()

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
                
    X = []  # Entradas: [maxSimilarity, minLivDistance. bertSim]
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