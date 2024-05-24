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

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from mpi4py import MPI
import json

from sentence_transformers import SentenceTransformer

model_name = "neuralmind/bert-large-portuguese-cased"
model = SentenceTransformer(model_name)

def write_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def calculate_cosine_similarity(reference_response, student_responses):
    vectors = model.encode([reference_response] + student_responses)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0][1:]

def normalize(value, left_min, left_max, right_min, right_max):
    return right_min + ((value - left_min) / (left_max - left_min)) * (right_max - right_min)

def process_questions(questionList):
    answersNum = 0
    answerParamsList = []
    for question in questionList:
        for answer in question.responses_students:
            bertSim = []
            for reference in question.reference_responses:
                bertScore = calculate_cosine_similarity(reference.reference_response, [answer.answer_question])
                normalizedBert = normalize(bertScore, -1, 1, 0, 3)
                bertSim.append(normalizedBert)
            
            answerParamsList.append((answersNum, answer, 0, 0, max(bertSim)))
            answersNum += 1
    
    return answerParamsList
                
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        questionList = dcf.loadQuestions()[0:4]
        chunks = np.array_split(questionList, size)
    else:
        chunks = None
    
    local_chunk = comm.scatter(chunks, root=0)

    local_results = process_questions(local_chunk)

    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        flattened_results = [item for sublist in all_results for item in sublist]
        answerParamsDict = [{"answersNum": a, "answer": b, "bertSim": d} for (a, b, c, d, e) in flattened_results]
        write_to_json(answerParamsDict, './normalizedData/ptbrDataset/answersParamsLarge.json')
                
if __name__ == "__main__":
    main()
