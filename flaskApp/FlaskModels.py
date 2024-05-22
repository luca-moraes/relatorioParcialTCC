from dataclasses import dataclass
from typing import List

@dataclass
class RefResponse:
    number_question: int
    reference_response: str
    
@dataclass
class Keywords:
    number_question: int
    word: str
    
@dataclass
class QuestionsOnly:
    number_question: int
    question_text: str
    keywords: List[Keywords]
    reference_responses: List[RefResponse]
    
@dataclass
class AnswerParams:
    frequencia_termos: float
    levenshtein_distancia: float
    bert_semantica: float
    nota_final: float