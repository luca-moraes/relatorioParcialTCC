import json
import numpy as np
import DataClassFiller as dcf
# import nlu
import os

from transformers import AutoTokenizer, AutoModel

from transformers import BertModel, BertTokenizer
import torch

from torch.nn.functional import cosine_similarity
import torch.nn.functional as tnf

#from transformers import BertTokenizer, BertModel
import torch
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from dataclasses import asdict
from Models import Question, Answer, RefResponse, Keywords, AnswerParams
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from Levenshtein import distance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mpi4py import MPI

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# pipe = nlu.load('pt.embed_sentence')

model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()

def get_embedding_device(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()

def get_all_embedding(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def write_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def calculate_cosine_similarity(reference_response, student_responses):
    vectorizer = CountVectorizer().fit([reference_response] + student_responses)
    vectors = vectorizer.transform([reference_response] + student_responses)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0][1:]

# def get_sentence_embedding(text, tokenizer, model):
#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Use a média dos tokens como o embedding da sentença
#     embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings.numpy()

# def get_embeddings(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     outputs = model(**inputs)
#     embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings.detach().numpy()

# def calculate_bert_similarity(text1, text2):
#     embeddings1 = get_embeddings(text1)
#     embeddings2 = get_embeddings(text2)
#     similarity = cosine_similarity(embeddings1, embeddings2)
#     return similarity[0][0]

# def get_embeddings_nlu(text):
#     # Criar um DataFrame com o texto
#     df = pipe.predict([text], output_level='document')
#     # Extrair os embeddings
#     embeddings = np.array(df['pt_embed_sentence_embeddings'].tolist())
#     return embeddings

# def calculate_similarity_nlu(text1, text2):
#     embeddings1 = get_embeddings_nlu(text1)
#     embeddings2 = get_embeddings_nlu(text2)
#     similarity = cosine_similarity(embeddings1, embeddings2)
#     return similarity[0][0]

# def calculate_similarity_ebd(predictions, e_col):
#     # Extrai os embeddings
#     embed_mat = np.array([x for x in predictions[e_col]])
#     # Calcula a similaridade de cosseno
#     sim_mat = cosine_similarity(embed_mat, embed_mat)
#     return sim_mat

def normalize(value, left_min, left_max, right_min, right_max):
    # Mapeia value do intervalo [left_min, left_max] para [right_min, right_max]
    return right_min + ((value - left_min) / (left_max - left_min)) * (right_max - right_min)

def main():
    questionList = dcf.loadQuestions()
    #pipe = nlu.load('xx.embed_sentence')
    
    answersNum = 0
    answerParamsList = []
    for question in questionList:
        reference_embeddings = get_embedding_device([ref.reference_response for ref in question.reference_responses])

        answersList = question.responses_students
        for answer in answersList:
            student_embedding = get_embedding_device([answer.answer_question]).squeeze(0)
            bert_scores = [tnf.cosine_similarity(student_embedding, ref_emb.unsqueeze(0)).item() for ref_emb in reference_embeddings]
            normalized_bert_scores = [normalize(score, -1, 1, 0, 3) for score in bert_scores]

            similarities = []
            livDistances = []
            # bertSim = []
            
            for reference in question.reference_responses:
            #     print("Bert")
                freqSimilarity = calculate_cosine_similarity(reference.reference_response, [answer.answer_question])
                livDistance = distance(answer.answer_question, reference.reference_response)

                #normalizedSimilarity = similarity[0]
                #normalizedSimilarity = 0 + ((freqSimilarity[0] - 1) * (3 - 0)) / (1 - 0)
                #normalizedSimilarity = 0 + ((freqSimilarity[0] + 1) * (3 - 0)) / (1 - (-1))
                normalizedFreq = normalize(freqSimilarity[0], -1, 1, 0, 3)

                #normalizedDistance = livDistance
                #normalizedDistance = len(reference.reference_response) - livDistance
                normalizedDistance = 0
                if livDistance < len(reference.reference_response):
                    normalizedDistance = 3 * (len(reference.reference_response) - livDistance) / len(reference.reference_response)
                                
                #bert semantica:
                #similarity_score = calculate_bert_similarity(reference.reference_response, answer.answer_question)

                #bert nlu:
                # df = pd.DataFrame({'text': [reference.reference_response, answer.answer_question]})
                # similarity_score = calculate_similarity_nlu(reference.reference_response, answer.answer_question)
                # predictions = pipe.predict(df.text, output_level='document')
                # e_col = 'sentence_embeddings'

                # similarity_matrix = calculate_similarity_ebd(predictions, e_col)
                # similarity_score = similarity_matrix[0, 1]

                #bert mini:
                # embedding1 = get_sentence_embedding(reference.reference_response, tokenizer, model)
                # embedding2 = get_sentence_embedding(answer.answer_question, tokenizer, model)
                # similarity_score = cosine_similarity(embedding1, embedding2)[0][0]

                #bert ptbr:
                # embedding1 = get_embedding(reference.reference_response)
                # embedding2 = get_embedding(answer.answer_question)
                # bertScore = tnf.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))

                # normalizedBert = normalize(bertScore, -1, 1, 0, 3)

                # similarities.append(similarity[0])
                # livDistances.append(livDistance)
                similarities.append(normalizedFreq)
                livDistances.append(normalizedDistance)
                # bertSim.append(normalizedBert)

            # embedding1 = get_embedding(question.reference_responses[0].reference_response)
            # embedding2 = get_embedding(answer.answer_question)
            # bertScore = tnf.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))

            # normalizedBert = normalize(bertScore.item(), -1, 1, 0, 3)
            
            answerParamsList.append(AnswerParams(answersNum,
                answer,
                # 0,
                # 0,
                max(similarities),
                max(livDistances),
                # max(bertSim)
                # normalizedBert
                # 0
                max(normalized_bert_scores)
                ))
            answersNum += 1
    
    answerParamsDict = [asdict(answerParams) for answerParams in answerParamsList]
    write_to_json(answerParamsDict, '../normalizedData/ptbrDataset/answersParams3.json')
                
if __name__ == "__main__":
    main()