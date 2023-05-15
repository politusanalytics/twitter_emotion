"""
This script is used to predict big files with a trained model from train.py.
You can already do prediction in train.py, but this script lazy loads a file
so that we can process bigger files.
"""

from transformers import AutoModel, AutoTokenizer, AutoConfig
from commons import MLP
import sys
import numpy as np
import json
import torch
import gzip

# INPUTS
database_or_input_filename = sys.argv[1]

# MUST SET THESE VALUES
output_filename = "out.json"
pretrained_transformers_model = "dbmdz/bert-base-turkish-128k-cased"
max_seq_length = 64
batch_size = 1536
repo_path = "/home/username/twitter_emotion"

# # GoEmotions
# idx_to_label = ['admiration','amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
#                 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
#                 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
#                 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
# encoder_path = "{}/models/best_models/go_emotions_encoder_cardiffnlp_twitter-xlm-roberta-base_43.pt".format(repo_path)
# classifier_path = "{}/models/best_models/go_emotions_classifier_cardiffnlp_twitter-xlm-roberta-base_43.pt".format(repo_path)

# # Affect in Tweets
# idx_to_label = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism",
#                 "sadness", "surprise", "trust"]
# encoder_path = "{}/models/best_models/affect_in_tweets_encoder_cardiffnlp_twitter-xlm-roberta-base_45.pt".format(repo_path)
# classifier_path = "{}/models/best_models/affect_in_tweets_classifier_cardiffnlp_twitter-xlm-roberta-base_45.pt".format(repo_path)

# Politus
idx_to_label = ["notr", "mutluluk", "sevgi", "umut", "minnet", "saskinlik", "uzuntu", "kaygi", "korku",
                "umutsuzluk", "utanc", "pismanlik", "ofke", "igrenme", "arzu", "onaylama", "onaylamama"]
encoder_path = "{}/models/best_models/politus_encoder_dbmdz_bert-base-turkish-128k-cased_71.pt".format(repo_path)
classifier_path = "{}/models/best_models/politus_classifier_dbmdz_bert-base-turkish-128k-cased_71.pt".format(repo_path)

query = {"text": {"$nin": ["", None]}, "emotions": None}

# See if there is anything to predict
if database_or_input_filename == "database":
    import pymongo
    # Connect to mongodb
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["politus_twitter"]
    tweet_col = db["tweets"]

    num_tweets_to_predict = tweet_col.count_documents(query)
    if num_tweets_to_predict == 0:
        print("No documents to predict. Exiting...")
        sys.exit(0)


device = torch.device("cuda")

# OPTIONS
return_probabilities = False
positive_threshold = 0.5

# LOAD MODEL
tokenizer = AutoTokenizer.from_pretrained(pretrained_transformers_model)
config = AutoConfig.from_pretrained(pretrained_transformers_model)
has_token_type_ids = config.type_vocab_size > 1

encoder = AutoModel.from_pretrained(pretrained_transformers_model)
encoder.to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device))
classifier = MLP(encoder.config.hidden_size, encoder.config.hidden_size*4, len(idx_to_label))
classifier.to(device)
classifier.load_state_dict(torch.load(classifier_path, map_location=device))

encoder = torch.nn.DataParallel(encoder)
encoder.eval()
classifier.eval()

def model_predict(batch):
    if has_token_type_ids:
        input_ids, input_mask, token_type_ids = tuple(t.to(device) for t in batch.values())
        embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[1]
    else:
        input_ids, input_mask = tuple(t.to(device) for t in batch.values())
        embeddings = encoder(input_ids, attention_mask=input_mask)[1]

    out = classifier(embeddings)

    all_preds = torch.sigmoid(out).detach().cpu().numpy().tolist()
    if return_probabilities:
        all_preds = [[round(float(x), 4) for x in preds] for preds in all_preds]
    else:
        all_preds = [[idx_to_label[i] for i, x in enumerate(preds) if x >= positive_threshold] for preds in all_preds]

    return all_preds

def preprocess(text): # Preprocess text (username and link placeholders)
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# !!!IMPORTANT!!!
# Change this according to the json line format.
def read_json_line(data):
    data_id = data["id"]
    text = preprocess(str(data["text"]))

    return data_id, text

if __name__ == "__main__":
    total_processed = 0
    if database_or_input_filename == "database": # if database
        # NOTE: This find can be changed according to the task.
        tweets_to_predict = tweet_col.find(query, ["_id", "text"])

        curr_batch = []
        for i, tweet in enumerate(tweets_to_predict):
            id_str = tweet["_id"]
            text = preprocess(tweet["text"])

            if len(text) > 0:
                total_processed += 1
                curr_batch.append({"_id": id_str, "text": text})

            if len(curr_batch) == batch_size:
                texts = [d["text"] for d in curr_batch]
                inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                                   max_length=max_seq_length)
                preds = model_predict(inputs)

                curr_updates = [UpdateOne({"_id": curr_batch[pred_idx]["_id"]}, {"$set": {task_name: pred}}) for pred_idx, pred in enumerate(preds)]
                tweet_col.bulk_write(curr_updates, ordered=False)

                curr_batch = []

        # Last incomplete batch, if any
        if len(curr_batch) != 0:
            texts = [d["text"] for d in curr_batch]
            inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=max_seq_length)
            preds = model_predict(inputs)

            curr_updates = [UpdateOne({"_id": curr_batch[pred_idx]["_id"]}, {"$set": {task_name: pred}}) for pred_idx, pred in enumerate(preds)]
            tweet_col.bulk_write(curr_updates, ordered=False)

    else: # if filename
        output_file = open(output_filename, "w", encoding="utf-8")
        if database_or_input_filename.endswith(".json.gz"):
            input_file = gzip.open(database_or_input_filename, "rt", encoding="utf-8")
        elif database_or_input_filename.endswith(".json"):
            input_file = open(database_or_input_filename, "r", encoding="utf-8")
        else:
            raise("File extension should be 'json' or 'json.gz'!")

        curr_batch = []
        for i, line in enumerate(input_file):
            data = json.loads(line)
            data_id, text = read_json_line(data)

            if len(text) > 0:
                total_processed += 1
                curr_batch.append({"id": data_id, "text": text})

            if len(curr_batch) == batch_size:
                texts = [d["text"] for d in curr_batch]
                inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                                   max_length=max_seq_length)
                preds = model_predict(inputs)
                for pred_idx, pred in enumerate(preds):
                    curr_d = curr_batch[pred_idx]
                    curr_d.pop("text") # No need for text in the output.
                    curr_d["prediction"] = pred
                    output_file.write(json.dumps(curr_d, ensure_ascii=False) + "\n")

                curr_batch = []

        input_file.close()

        # Last incomplete batch, if any
        if len(curr_batch) != 0:
            texts = [d["text"] for d in curr_batch]
            inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=max_seq_length)
            preds = model_predict(inputs)
            for pred_idx, pred in enumerate(preds):
                curr_d = curr_batch[pred_idx]
                curr_d.pop("text") # No need for text in the output.
                curr_d["prediction"] = pred
                output_file.write(json.dumps(curr_d, ensure_ascii=False) + "\n")

        output_file.close()

    print("Processed {} tweets in total.".format(str(total_processed)))
