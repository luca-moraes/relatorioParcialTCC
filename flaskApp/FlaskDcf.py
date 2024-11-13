import json
from FlaskModels import RefResponse, Keywords, QuestionsOnly
    
def load_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    question_list = []
    for item in data:
        number_question = item['number_question']
        question_text = item['question_text']
        keywords = [Keywords(**kw) for kw in item['keywords']]
        reference_responses = [RefResponse(**rr) for rr in item['reference_responses']]
        
        question = QuestionsOnly(number_question=number_question,
                            question_text=question_text,
                            keywords=keywords,
                            reference_responses=reference_responses)
        
        question_list.append(question)

    return question_list

def loadQuestionsOnly():
    filename = './questions.json'
    loaded_questions = load_from_json(filename)

    return loaded_questions

def loadEnQuestionsOnly():
    filename = './questionsEn.json'
    loaded_questions = load_from_json(filename)

    return loaded_questions

def loadEsQuestionsOnly():
    filename = './questionsEs.json'
    loaded_questions = load_from_json(filename)

    return loaded_questions