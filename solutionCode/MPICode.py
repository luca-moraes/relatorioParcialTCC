import json
import numpy as np
import DataClassFiller as dcf
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
import torch
#from torch.nn.functional import cosine_similarity
import torch.nn.functional as tnf

from dataclasses import asdict
from Models import Question, Answer, RefResponse, Keywords, AnswerParams
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from Levenshtein import distance
from concurrent.futures import ThreadPoolExecutor

from mpi4py import MPI

# model_name = "neuralmind/bert-large-portuguese-cased"
# model_name = "google-bert/bert-large-cased"
# model_name = "dccuchile/bert-base-spanish-wwm-cased"

# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)

#model_name = "flax-community/bertin-roberta-large-spanish"
model_name = "bertin-project/bertin-roberta-base-spanish"

tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_embedding_device(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()

def write_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def calculate_cosine_similarity(reference_response, student_responses):
    vectorizer = CountVectorizer().fit([reference_response] + student_responses)
    vectors = vectorizer.transform([reference_response] + student_responses)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0][1:]

def calculate_freq_and_liv(reference, answer_question):
    freqSimilarity = calculate_cosine_similarity(reference.reference_response, [answer_question])
    normalizedFreq = normalize(freqSimilarity[0], -1, 1, 0, 3)
    
    livDistance = distance(answer_question, reference.reference_response)
    normalizedDistance = 0
    if livDistance < len(reference.reference_response):
        normalizedDistance = 3 * (len(reference.reference_response) - livDistance) / len(reference.reference_response)
    
    return normalizedFreq, normalizedDistance

def calculate_freq_and_liv_en(reference, answer_question):
    freqSimilarity = calculate_cosine_similarity(reference.reference_response, [answer_question])
    normalizedFreq = normalize(freqSimilarity[0], -1, 1, 0, 5)
    
    livDistance = distance(answer_question, reference.reference_response)
    normalizedDistance = 0
    if livDistance < len(reference.reference_response):
        normalizedDistance = 5 * (len(reference.reference_response) - livDistance) / len(reference.reference_response)
    
    return normalizedFreq, normalizedDistance

def calculate_freq_and_liv_es(reference, answer_question):
    freqSimilarity = calculate_cosine_similarity(reference.reference_response, [answer_question])
    normalizedFreq = normalize(freqSimilarity[0], -1, 1, 0, 5)
    
    livDistance = distance(answer_question, reference.reference_response)
    normalizedDistance = 0
    if livDistance < len(reference.reference_response):
        normalizedDistance = 5 * (len(reference.reference_response) - livDistance) / len(reference.reference_response)
    
    return normalizedFreq, normalizedDistance

def normalize(value, left_min, left_max, right_min, right_max):
    result = right_min + ((value - left_min) / (left_max - left_min)) * (right_max - right_min)
    return right_max if result >= right_max else result

def process_questions(questionList):
    answerParamsList = []
    for question in questionList:
        reference_embeddings = get_embedding_device([ref.reference_response for ref in question.reference_responses])

        answersList = question.responses_students
        for answer in answersList:
            student_embedding = get_embedding_device([answer.answer_question]).squeeze(0)
            bert_scores = [tnf.cosine_similarity(student_embedding, ref_emb.unsqueeze(0)).item() for ref_emb in reference_embeddings]
            normalized_bert_scores = [normalize(score, -1, 1, 0, 3) for score in bert_scores]

            freqSimilarity = []
            livDistances = []
            
            # for reference in question.reference_responses:
            #     freqSimilarity = calculate_cosine_similarity(reference.reference_response, [answer.answer_question])
            #     normalizedFreq = normalize(freqSimilarity[0], -1, 1, 0, 3)
                
            #     livDistance = distance(answer.answer_question, reference.reference_response)
            #     normalizedDistance = 0
            #     if livDistance < len(reference.reference_response):
            #         normalizedDistance = 3 * (len(reference.reference_response) - livDistance) / len(reference.reference_response)
                
            #     freqSimilarity.append(normalizedFreq)
            #     livDistances.append(normalizedDistance)
                
            with ThreadPoolExecutor() as executor:
                future_to_ref = {executor.submit(calculate_freq_and_liv, reference, answer.answer_question): reference for reference in question.reference_responses}
                for future in future_to_ref:
                    normalizedFreq, normalizedDistance = future.result()
                    freqSimilarity.append(normalizedFreq)
                    livDistances.append(normalizedDistance)
            
            answerParamsList.append(AnswerParams(0,
                answer,
                max(freqSimilarity),
                max(livDistances),
                max(normalized_bert_scores)
                ))
    
    return answerParamsList
       
def process_en_questions(questionList):
    answerParamsList = []
    for question in questionList:
        reference_embeddings = get_embedding_device([ref.reference_response for ref in question.reference_responses])

        answersList = question.responses_students
        for answer in answersList:
            student_embedding = get_embedding_device([answer.answer_question]).squeeze(0)
            bert_scores = [tnf.cosine_similarity(student_embedding, ref_emb.unsqueeze(0)).item() for ref_emb in reference_embeddings]
            normalized_bert_scores = [normalize(score, -1, 1, 0, 5) for score in bert_scores]

            freqSimilarity = []
            livDistances = []

            with ThreadPoolExecutor() as executor:
                future_to_ref = {executor.submit(calculate_freq_and_liv_en, reference, answer.answer_question): reference for reference in question.reference_responses}
                for future in future_to_ref:
                    normalizedFreq, normalizedDistance = future.result()
                    freqSimilarity.append(normalizedFreq)
                    livDistances.append(normalizedDistance)
            
            answerParamsList.append(AnswerParams(0,
                answer,
                max(freqSimilarity),
                max(livDistances),
                max(normalized_bert_scores)
                ))
    
    return answerParamsList
           
def process_es_questions(questionList):
    answerParamsList = []
    for question in questionList:
        reference_embeddings = get_embedding_device([ref.reference_response for ref in question.reference_responses])

        answersList = question.responses_students
        for answer in answersList:
            student_embedding = get_embedding_device([answer.answer_question]).squeeze(0)
            bert_scores = [tnf.cosine_similarity(student_embedding, ref_emb.unsqueeze(0)).item() for ref_emb in reference_embeddings]
            normalized_bert_scores = [normalize(score, -1, 1, 0, 5) for score in bert_scores]

            freqSimilarity = []
            livDistances = []

            with ThreadPoolExecutor() as executor:
                future_to_ref = {executor.submit(calculate_freq_and_liv_es, reference, answer.answer_question): reference for reference in question.reference_responses}
                for future in future_to_ref:
                    normalizedFreq, normalizedDistance = future.result()
                    freqSimilarity.append(normalizedFreq)
                    livDistances.append(normalizedDistance)
            
            answerParamsList.append(AnswerParams(0,
                answer,
                max(freqSimilarity),
                max(livDistances),
                max(normalized_bert_scores)
                ))
    
    return answerParamsList
                
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # questionList = dcf.loadQuestions()
        # questionList = dcf.loadEnQuestions()
        questionList = dcf.loadEsQuestions()
        chunks = np.array_split(questionList, size)
    else:
        chunks = None
    
    local_chunk = comm.scatter(chunks, root=0)

    local_results = process_en_questions(local_chunk)

    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        answers_full_list = [item for sublist in all_results for item in sublist]
        
        for idx, answerParam in enumerate(answers_full_list):
            answerParam.answer_number = idx
        
        answerParamsDict = [asdict(answerParams) for answerParams in answers_full_list]
        # write_to_json(answerParamsDict, '../normalizedData/ptbrDataset/answersParamsLarge.json')
        # write_to_json(answerParamsDict, '../normalizedData/enDataset/answersParamsLarge.json')
        write_to_json(answerParamsDict, '../normalizedData/esDataset/answersParamsRoberta.json')
                
if __name__ == "__main__":
    main()