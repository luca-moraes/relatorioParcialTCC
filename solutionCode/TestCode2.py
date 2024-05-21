import nlu
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

pipe = nlu.load('pt.embed_sentence')

text1 = "A célula animal e vegetal apresentam formato diferenciado. A célula animal possui formato irregular, enquanto a célula vegetal apresenta uma forma fixa."
text2 = "As células animais são todas aquelas que compõem os seres vivos do reino animália, composto por membrana plasmática, citoplasma e núcleo verdadeiro. As células vegerais contem estruturas como parede celulares, plastídios e grandes vacúolos"

import pandas as pd
df = pd.DataFrame({'text': [text1, text2]})

predictions = pipe.predict(df.text, output_level='document')

e_col = 'sentence_embeddings'

def calculate_similarity(predictions, e_col):
    embed_mat = np.array([x for x in predictions[e_col]])
    sim_mat = cosine_similarity(embed_mat, embed_mat)
    return sim_mat

similarity_matrix = calculate_similarity(predictions, e_col)
similarity_score = similarity_matrix[0, 1]

print(f"A similaridade entre os textos é: {similarity_score:.4f}")