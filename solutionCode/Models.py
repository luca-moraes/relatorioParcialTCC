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
    
@dataclass
class QuestionsOnly:
    number_question: int
    question_text: str
    keywords: List[Keywords]
    reference_responses: List[RefResponse]
    
@dataclass
class AnswerParams:
    answer_number: int
    answer_values: Answer
    frequence_similarity: float
    liv_distance: float
    bert_score: float

@dataclass
class AnswerTestParams:
    answer_number: int
    answer_values: Answer
    frequence_similarity: float
    liv_distance: float
    bert_score: float
    nota_atribuida: float
    percentual_error: float