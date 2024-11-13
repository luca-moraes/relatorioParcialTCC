import json
from Models import Answer, RefResponse, Keywords, Question, AnswerParams

baseString = '../'

def loadAnswersParamsJson(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    answer_list = []
    for item in data:
        answer = Answer(**item['answer_values'])
        cosine_similarity = item['frequence_similarity']
        liv_distance = item['liv_distance']
        bert_score = item['bert_score']

        answer_params = AnswerParams(answer_number=item['answer_number'],
                                    answer_values=answer,
                                    frequence_similarity=cosine_similarity,
                                    liv_distance=liv_distance,
                                    bert_score=bert_score)
        
        answer_list.append(answer_params)

    return answer_list
    
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

def loadQuestions():
    filename = baseString + 'normalizedData/ptbrDataset/ptbrData.json'
    loaded_questions = load_from_json(filename)

    # for question in loaded_questions:
    #     print(f"Question {question.number_question}: {question.question_text}")
    #     print(f"Keywords: {', '.join([kw.word for kw in question.keywords])}")
    #     print(f"Reference Responses: {[rr.reference_response for rr in question.reference_responses]}")
    #     print(f"Number of Student Responses: {len(question.responses_students)}")

    return loaded_questions

def loadEnQuestions():
    filename = baseString + 'normalizedData/enDataset/enData.json'
    loaded_questions = load_from_json(filename)
    
    return loaded_questions

def loadEsQuestions():
    filename = baseString + 'normalizedData/esDataset/esData.json'
    loaded_questions = load_from_json(filename)
    
    return loaded_questions

def loadAnswersParams():
    filename = baseString + 'normalizedData/ptbrDataset/answersParams.json'
    loaded_answers_params = loadAnswersParamsJson(filename)
    return loaded_answers_params

def loadAnswersParamsLarge():
    filename = baseString + 'normalizedData/ptbrDataset/answersParamsLarge.json'
    loaded_answers_params = loadAnswersParamsJson(filename)
    return loaded_answers_params

def loadEnAnswersParams():
    filename = baseString + 'normalizedData/enDataset/answersParams.json'
    loaded_answers_params = loadAnswersParamsJson(filename)
    return loaded_answers_params

def loadEnAnswersParamsLarge():
    filename = baseString + 'normalizedData/enDataset/answersParamsLarge.json'
    loaded_answers_params = loadAnswersParamsJson(filename)
    return loaded_answers_params

def loadEsAnswersParams():
    filename = baseString + 'normalizedData/esDataset/answersParams.json'
    loaded_answers_params = loadAnswersParamsJson(filename)
    return loaded_answers_params

def loadEsAnswersParamsBerta():
    filename = baseString + 'normalizedData/esDataset/answersParamsRoberta.json'
    loaded_answers_params = loadAnswersParamsJson(filename)
    return loaded_answers_params

def loadEsAnswersParamsLarge():
    filename = baseString + 'normalizedData/esDataset/answersParamsLarge.json'
    loaded_answers_params = loadAnswersParamsJson(filename)
    return loaded_answers_params
    
# if __name__ == "__main__":
#     main()