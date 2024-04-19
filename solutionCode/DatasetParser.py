from dataclasses import dataclass, field, asdict
from Models import Answer, RefResponse, Keywords, Question
from typing import List
import csv
import json

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

def main():
    questions_data = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/questions.csv')
    keywords_data = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/concepts.csv')
    reference_responses_data = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/reference_answers.csv')
    students_responses_data = read_csv('./dataset/ptbr-asag/PT_ASAG_2018_v2.0/PT_ASAG_2018_v2.0/student_answers_and_grades_v1.csv')

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

    studentsResponseList = []
    for row in students_responses_data:
        question_id = int(row[0])
        answer_text = row[1]
        answer_grade = float(row[2])
        studentsResponseList.append(Answer(number_question=question_id, answer_question=answer_text, grade=answer_grade))
        
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
                
    question_dicts = [asdict(question) for question in questionList]
    write_to_json(question_dicts, './normalizedData/ptbrData.json')
    
    print("Done!")
                
if __name__ == "__main__":
    main()