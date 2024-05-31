from dataclasses import asdict
from Models import Answer, RefResponse, Keywords, Question, QuestionsOnly
import csv
import json

def read_en_txt(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(line.strip().split('\t'))
    return data

def read_csv(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append(row)
    return data

def write_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def parseEsDataset():
    # Informações do dataset
    data = read_csv('./dataset/spanish-asag/dataset_q_ma_sa.csv')
    
    # Parser dos dados
    question_dict = {}
    for row in data:
        question_text = row[5]  # Ajuste conforme necessário para corresponder à coluna correta
        reference_response_text = row[6]
        student_response_text = row[1]
        student_grade = float(row[3])
        
        if question_text not in question_dict:
            question_dict[question_text] = {
                "question_text": question_text,
                "reference_response": RefResponse(number_question=len(question_dict)+1, reference_response=reference_response_text),
                "responses_students": []
            }
        
        question_dict[question_text]["responses_students"].append(
            Answer(number_question=len(question_dict), answer_question=student_response_text, grade=student_grade)
        )
    
    questionList = []
    for idx, (question_text, question_data) in enumerate(question_dict.items(), start=1):
        question = Question(
            number_question=idx,
            question_text=question_text,
            keywords=[],
            reference_responses=[question_data["reference_response"]],
            responses_students=question_data["responses_students"]
        )
        questionList.append(question)
    
    questionsOnlyList = [QuestionsOnly(
        number_question=question.number_question,
        question_text=question.question_text,
        keywords=question.keywords,
        reference_responses=question.reference_responses
    ) for question in questionList]
    
    # Criação do JSON
    question_dicts = [asdict(question) for question in questionList]
    write_to_json(question_dicts, './normalizedData/esDataset/esData.json')
    
    # JSON de perguntas
    questionOnly_dicts = [asdict(question) for question in questionsOnlyList]
    write_to_json(questionOnly_dicts, './normalizedData/esDataset/esQuestionsOnly.json')
    
    print("Done!")

def parseEnDataset():
    # Informações do dataset
    data = read_en_txt('./dataset/texas-asag/ASAG-Method/dataset/NorthTexasDataset/expand.txt')
    
    # Parser dos dados
    question_dict = {}
    for row in data:
        question_text = row[0]
        reference_response_text = row[1]
        student_response_text = row[2]
        student_grade = float(row[3])
        
        if question_text not in question_dict:
            question_dict[question_text] = {
                "question_text": question_text,
                "reference_response": RefResponse(number_question=len(question_dict)+1, reference_response=reference_response_text),
                "responses_students": []
            }
        
        question_dict[question_text]["responses_students"].append(
            Answer(number_question=len(question_dict), answer_question=student_response_text, grade=student_grade)
        )
    
    questionList = []
    for idx, (question_text, question_data) in enumerate(question_dict.items(), start=1):
        question = Question(
            number_question=idx,
            question_text=question_data["question_text"],
            keywords=[],
            reference_responses=[question_data["reference_response"]],
            responses_students=question_data["responses_students"]
        )
        questionList.append(question)
    
    questionsOnlyList = [QuestionsOnly(
        number_question=question.number_question,
        question_text=question.question_text,
        keywords=question.keywords,
        reference_responses=question.reference_responses
    ) for question in questionList]
    
    # Criação do JSON
    question_dicts = [asdict(question) for question in questionList]
    write_to_json(question_dicts, './normalizedData/enDataset/enData.json')
    
    # JSON de perguntas
    questionOnly_dicts = [asdict(question) for question in questionsOnlyList]
    write_to_json(questionOnly_dicts, './normalizedData/enDataset/enQuestionsOnly.json')
    
    print("Done!")

def parsePtBrDataset():
    #infos dos datasets 
    questions_data = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/questions.csv')
    keywords_data = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/concepts.csv')
    
    reference_responses_data = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/reference_answers.csv')
    reference_responses_data2 = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/reference_answers_extended.csv')
    
    students_responses_data = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/student_answers_and_grades_v1.csv')
    students_responses_data2 = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/student_answers_and_grades_v2.csv')
    students_responses_data3 = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/student_answers_and_grades_v2_other_graders.csv')
    
    #parser dos dados
    questionList = []
    for row in questions_data:
        question_id = int(row[0])
        question_text = row[1]
        questionList.append(Question(number_question=question_id, question_text=question_text, keywords=[], reference_responses=[], responses_students=[]))
        
    keywordsList = []
    for row in keywords_data:
        question_id = int(row[0])
        keyword_text = row[1]
        keywordsList.append(Keywords(number_question=question_id, word=keyword_text))

    referenceResponsesList = []
    for row in reference_responses_data:
        question_id = int(row[0])
        reference_response_text = row[1]
        referenceResponsesList.append(RefResponse(number_question=question_id, reference_response=reference_response_text))
    for row in reference_responses_data2:
        question_id = int(row[0])
        reference_response_text = row[1]
        referenceResponsesList.append(RefResponse(number_question=question_id, reference_response=reference_response_text))

    studentsResponseList = []
    for row in students_responses_data:
        question_id = int(row[0])
        answer_text = row[1]
        answer_grade = float(row[2])
        studentsResponseList.append(Answer(number_question=question_id, answer_question=answer_text, grade=answer_grade))
    for row in students_responses_data2:
        question_id = int(row[0])
        answer_text = row[1]
        answer_grade = float(row[2])
        studentsResponseList.append(Answer(number_question=question_id, answer_question=answer_text, grade=answer_grade))
    for row in students_responses_data3:
        question_id = int(row[0])
        answer_text = row[1]
        answer_grade = float(row[2])
        studentsResponseList.append(Answer(number_question=question_id, answer_question=answer_text, grade=answer_grade))
        
    #juncao das infos
    for question in questionList:
        for keyword in keywordsList:
            if question.number_question == keyword.number_question:
                question.keywords.append(keyword)
        
    for question in questionList:
        for reference in referenceResponsesList:
            if question.number_question == reference.number_question:
                question.reference_responses.append(reference)

    for question in questionList:
        for student in studentsResponseList:
            if question.number_question == student.number_question:
                question.responses_students.append(student)
              
    questionsOnlyList = []
    for question in questionList:
        questionsOnlyList.append(QuestionsOnly(number_question=question.number_question,
            question_text=question.question_text,
            keywords=question.keywords,
            reference_responses=question.reference_responses))    
        
    #criacao do json  
    question_dicts = [asdict(question) for question in questionList]
    write_to_json(question_dicts, './normalizedData/ptbrDataset/ptbrData.json')
    
    #json de perguntas
    questionOnly_dicts = [asdict(question) for question in questionsOnlyList]
    write_to_json(questionOnly_dicts, './normalizedData/ptbrDataset/ptbrQuestionsOnly.json')
    
    print("Done!")
            
def main():
    # parsePtBrDataset()
    # parseEnDataset()
    parseEsDataset()
        
if __name__ == "__main__":
    main()