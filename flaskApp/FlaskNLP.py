from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from FlaskModels import AnswerParams
from Levenshtein import distance
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as tnf

def get_embedding_device(texts):
    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()

def get_embedding_device_en(texts):
    model_name = "google-bert/bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()

def get_embedding_device_es(texts):
    model_name = "dccuchile/bert-base-spanish-wwm-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()

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

def calcular_nota_final(answer):
    peso1 = 0.47001387
    peso2 = 0.27656985
    peso3 = 0.51803824
    
    fator1 = peso1 * answer.frequencia_termos
    fator2 = peso2 * answer.levenshtein_distancia
    fator3 = peso3 * answer.bert_semantica
    
    soma = fator1 + fator2 + fator3
    
    peso = peso1 + peso2 + peso3
    
    nota = soma / peso
    
    return nota
    #return normalize(nota, 0, 3, 0, 10)
    
def calcular_nota_final_en(answer):
    peso1 = 0.41559304
    peso2 = -0.16848201
    peso3 = 0.15370208
    
    fator1 = peso1 * answer.frequencia_termos
    fator2 = peso2 * answer.levenshtein_distancia
    fator3 = peso3 * answer.bert_semantica
    
    soma = fator1 + fator2 + fator3
    
    peso = peso1 + peso2 + peso3
    
    nota = soma / peso
    
    return nota

def calcular_nota_final_es(answer):
    peso1 = 0.10336491
    peso2 = -0.04858511
    peso3 = 0.45042231
    
    fator1 = peso1 * answer.frequencia_termos
    fator2 = peso2 * answer.levenshtein_distancia
    fator3 = peso3 * answer.bert_semantica
    
    soma = fator1 + fator2 + fator3
    
    peso = peso1 + peso2 + peso3
    
    nota = soma / peso
    
    return nota

def process_questions(answer, referenceList):
    reference_embeddings = get_embedding_device([ref.reference_response for ref in referenceList])

    student_embedding = get_embedding_device([answer]).squeeze(0)
    bert_scores = [tnf.cosine_similarity(student_embedding, ref_emb.unsqueeze(0)).item() for ref_emb in reference_embeddings]
    normalized_bert_scores = [normalize(score, -1, 1, 0, 3) for score in bert_scores]

    freqSimilarity = []
    livDistances = []
            
    with ThreadPoolExecutor() as executor:
        future_to_ref = {executor.submit(calculate_freq_and_liv, reference, answer): reference for reference in referenceList}
        for future in future_to_ref:
            normalizedFreq, normalizedDistance = future.result()
            freqSimilarity.append(normalizedFreq)
            livDistances.append(normalizedDistance)
        
    answerParams = AnswerParams(
        max(freqSimilarity),
        max(livDistances),
        max(normalized_bert_scores),
        0
    )
    
    answerParams.nota_final = calcular_nota_final(answerParams)
    
    return answerParams

def process_questions_en(answer, referenceList):
    reference_embeddings = get_embedding_device_en([ref.reference_response for ref in referenceList])

    student_embedding = get_embedding_device_en([answer]).squeeze(0)
    bert_scores = [tnf.cosine_similarity(student_embedding, ref_emb.unsqueeze(0)).item() for ref_emb in reference_embeddings]
    normalized_bert_scores = [normalize(score, -1, 1, 0, 5) for score in bert_scores]

    freqSimilarity = []
    livDistances = []
            
    with ThreadPoolExecutor() as executor:
        future_to_ref = {executor.submit(calculate_freq_and_liv_en, reference, answer): reference for reference in referenceList}
        for future in future_to_ref:
            normalizedFreq, normalizedDistance = future.result()
            freqSimilarity.append(normalizedFreq)
            livDistances.append(normalizedDistance)
        
    answerParams = AnswerParams(
        max(freqSimilarity),
        max(livDistances),
        max(normalized_bert_scores),
        0
    )
    
    answerParams.nota_final = calcular_nota_final(answerParams)
    
    return answerParams

def process_questions_es(answer, referenceList):
    reference_embeddings = get_embedding_device_es([ref.reference_response for ref in referenceList])

    student_embedding = get_embedding_device_es([answer]).squeeze(0)
    bert_scores = [tnf.cosine_similarity(student_embedding, ref_emb.unsqueeze(0)).item() for ref_emb in reference_embeddings]
    normalized_bert_scores = [normalize(score, -1, 1, 0, 5) for score in bert_scores]

    freqSimilarity = []
    livDistances = []
            
    with ThreadPoolExecutor() as executor:
        future_to_ref = {executor.submit(calculate_freq_and_liv_es, reference, answer): reference for reference in referenceList}
        for future in future_to_ref:
            normalizedFreq, normalizedDistance = future.result()
            freqSimilarity.append(normalizedFreq)
            livDistances.append(normalizedDistance)
        
    answerParams = AnswerParams(
        max(freqSimilarity),
        max(livDistances),
        max(normalized_bert_scores),
        0
    )
    
    answerParams.nota_final = calcular_nota_final(answerParams)
    
    return answerParams