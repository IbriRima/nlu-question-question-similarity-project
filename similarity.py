import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import T5ForConditionalGeneration,T5Tokenizer
import argparse


def get_sim_quest(models,data,question):
    num_models = len(models)
    corpus_tfidf_list = []
    question_tfidf_list = []
    for model_name in models:
        model = SentenceTransformer(model_name)
        corpus_tfidf_list.append(model.encode(data['questions']))
        question_tfidf_list.append(model.encode([question]))
    cos_similarities = [cosine_similarity(q_tfidf, c_tfidf) for q_tfidf, c_tfidf in zip(question_tfidf_list, corpus_tfidf_list)]
    avg_similarities = sum(cos_similarities) / num_models
    print(cos_similarities[0].shape)
    sim_quest_idx = avg_similarities.argmax()
    sim_quest = data['questions'][sim_quest_idx]
    sim_answer = data['answers'][sim_quest_idx]
    return cos_similarities,sim_answer


def get_top_10(cos_similarities,data):
    highest_indices = np.argsort(-cos_similarities[0])[:10]
    for i in highest_indices:
        best_doc =data['questions'][i]
        print(best_doc)


def answer_generation(context,model,question,tokenizer):
    input_text = f"question:{question} context : {context}"
    input_ids = tokenizer.encode(input_text , return_tensors = "pt")
    output= model.generate(input_ids,max_new_tokens=20)
    answer = tokenizer.decode(output[0],skip_special_token=True)
    return answer

def main():
    parser = argparse.ArgumentParser(description="Question Answering Script")
    parser.add_argument("--question", required=True, type=str, help="Input question")
    parser.add_argument("--data-file", required=True, type=str, help="CSV file containing questions and answers")
    args = parser.parse_args()
    #Load data 
    data = pd.read_csv(args.data_file)
    models = ['distilbert-base-nli-mean-tokens', 'paraphrase-distilroberta-base-v1', 'paraphrase-MiniLM-L6-v2']
    model_name = 't5-base'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    cos_similarities, context = get_sim_quest(models, data, args.question)
    print("Question : ", args.question)
    get_top_10(cos_similarities, data)
    answer = answer_generation(context, model, args.question, tokenizer)
    print("Answer : ", answer)

if __name__=='__main__':
    main()