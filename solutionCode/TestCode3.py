# import nlu
# import sparknlp
# from pyspark.sql import SparkSession

# spark = SparkSession.builder \
#     .appName("TestApp") \
#     .config("spark.jars.repositories", "https://repo1.maven.org/maven2,https://repository.apache.org/content/groups/public/") \
#     .getOrCreate()

# spark = sparknlp.start()

# pipe = nlu.load('xx.embed_sentence')

# text1 = "A célula animal e vegetal apresentam formato diferenciado. A célula animal possui formato irregular, enquanto a célula vegetal apresenta uma forma fixa."
# text2 = "As células animais são todas aquelas que compõem os seres vivos do reino animália, composto por membrana plasmática, citoplasma e núcleo verdadeiro. As células vegerais contem estruturas como parede celulares, plastídios e grandes vacúolos"

# import pandas as pd
# df = pd.DataFrame({'text': [text1, text2]})

# predictions = pipe.predict(df.text, output_level='document')
# print(predictions)

# from pyspark.sql import SparkSession

# spark = SparkSession.builder \
#     .appName("NLU") \
#     .getOrCreate()

# print("Spark Session Initialized Successfully")

# import nlu

# # Lista todos os modelos disponíveis
# all_models = nlu.all_components()
# print(all_models)

from transformers import BertModel, BertTokenizer
import torch

# Carregar o modelo e o tokenizer
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Texto de exemplo
texts = ["Este é um exemplo de texto.", "Outro exemplo de texto para testar."]

# Tokenizar e obter as incorporações
for text in texts:
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    print(f"Incorporação para '{text}':")
    print(embeddings)