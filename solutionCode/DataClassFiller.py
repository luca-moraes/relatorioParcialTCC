import json
from dataclasses import dataclass
from typing import List

@dataclass
class Answer:
    number_question: int
    answer_question: str
    grade: float

@dataclass
class RefResponse:
    number_question: int
    reference_response: str

@dataclass
class Keywords:
    number_question: int
    word: str

@dataclass
class Question:
    number_question: int
    question_text: str
    keywords: List[Keywords]
    reference_responses: List[RefResponse]
    responses_students: List[Answer]

def load_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    question_list = []
    for item in data:
        number_question = item['number_question']
        question_text = item['question_text']
        keywords = [Keywords(**kw) for kw in item['keywords']]
        reference_responses = [RefResponse(**rr) for rr in item['reference_responses']]
        responses_students = [Answer(**ans) for ans in item['responses_students']]

        question = Question(number_question=number_question,
                            question_text=question_text,
                            keywords=keywords,
                            reference_responses=reference_responses,
                            responses_students=responses_students)
        
        question_list.append(question)

    return question_list

def main():
    filename = './normalizedData/ptbrData.json'
    loaded_questions = load_from_json(filename)

    for question in loaded_questions:
        print(f"Question {question.number_question}: {question.question_text}")
        print(f"Keywords: {', '.join([kw.word for kw in question.keywords])}")
        print(f"Reference Responses: {[rr.reference_response for rr in question.reference_responses]}")
        print(f"Number of Student Responses: {len(question.responses_students)}")

if __name__ == "__main__":
    main()
