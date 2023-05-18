import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration,T5Tokenizer


models = ['distilbert-base-nli-mean-tokens','paraphrase-distilroberta-base-v1', 'paraphrase-MiniLM-L6-v2']
model_name ='t5-base'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
data = pd.read_csv('preprocessed_dataset.csv')
question = input('Please enter your question: ') 

def get_sim_quest():
    num_models = len(models)
    corpus_tfidf = []
    for model_name in models:
        model = SentenceTransformer(model_name)
        corpus_tfidf.append(model.encode(data['questions']))


    question_tfidf = []
    for model_name in models:
        model = SentenceTransformer(model_name)
        question_tfidf.append(model.encode([question]))

    cos_similarities = [cosine_similarity(q_tfidf, c_tfidf) for q_tfidf, c_tfidf in zip(question_tfidf, corpus_tfidf)]
    avg_similarities = sum(cos_similarities) / num_models

    sim_quest_idx = avg_similarities.argmax()
    sim_quest = data['questions'][sim_quest_idx]
    sim_answer = data['answers'][sim_quest_idx]

    return cos_similarities,sim_answer


def get_top_10(cos_similarities):

    highest_indices = np.argsort(-cos_similarities[0])[:10]

    for i in highest_indices:
        best_doc =data['questions'][i]
        print(best_doc)

cos_similarities,context = get_sim_quest()

def answer_generation(context):
    
    input_text = f"question:{question} context : {context}"
    input_ids = tokenizer.encode(input_text , return_tensors = "pt")

    output= model.generate(input_ids)
    answer = tokenizer.decode(output[0],skip_special_token=True)
    return answer


print("Question : " , question)
get_top_10(cos_similarities)
print("Answer : ",answer_generation(context))