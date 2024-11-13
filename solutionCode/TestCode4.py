from transformers import BertModel, BertTokenizer
import torch
from torch.nn.functional import cosine_similarity

model_name = "neuralmind/bert-large-portuguese-cased"
#model_name = "google-bert/bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

text1 = "A célula animal e vegetal apresentam formato diferenciado. A célula animal possui formato irregular, enquanto a célula vegetal apresenta uma forma fixa."
text2 = "As células animais são todas aquelas que compõem os seres vivos do reino animália, composto por membrana plasmática, citoplasma e núcleo verdadeiro. As células vegerais contem estruturas como parede celulares, plastídios e grandes vacúolos"

# text1 = "Dor de barriga"
# text2 = "Aula na FEI"

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()

embedding1 = get_embedding(text1)
embedding2 = get_embedding(text2)

texts = [text1, text2]

for text in texts:
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    print(f"Embeddings:\n '{text}':")
    print(embeddings.shape)
    #print(len(embeddings))

similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
print(f"Similaridade de Cosseno: {similarity.item()}")