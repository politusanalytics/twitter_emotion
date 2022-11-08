"""
This script is used to predict big files with a trained model from train.py.
You can already do prediction in train.py, but this script lazy loads a file
so that we can process bigger files.
"""

from transformers import AutoModel, AutoTokenizer, AutoConfig
import sys
import numpy as np
import json
import torch
import gzip

# INPUTS
input_filename = sys.argv[1] # Should be json or json.gz file in json line format.
output_filename = sys.argv[2]

# MUST SET THESE VALUES
pretrained_transformers_model = "xlm-roberta-base"
max_seq_length = 512
batch_size = 64
idx_to_label = ["category1", "category2", "category3"]
encoder_path = ""
classifier_path = ""
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
classifier = torch.nn.Linear(encoder.config.hidden_size, 1 if len(idx_to_label) == 2 else len(idx_to_label))
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

    if len(idx_to_label) == 2:
        preds = torch.sigmoid(out).detach().cpu().numpy().flatten()
        if return_probabilities:
            preds = [round(float(x), 4) for x in preds]
        else:
            preds = [idx_to_label[int(x >= positive_threshold)] for x in preds]

    else:
        out = out.detach().cpu().numpy()
        if return_probabilities:
            preds = [probs for probs in softmax(out, axis=1).tolist()] # a list of lists(of probabilities)
        else:
            preds = [idx_to_label[pred] for pred in np.argmax(out, axis=1).tolist()] # a list of labels

    return preds

# !!!IMPORTANT!!!
# Change this according to the json line format.
# Here the format for every line is like:
# {id_str: text}
def read_json_line(data):
    def preprocess(text): # Preprocess text (username and link placeholders)
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    id_str = list(data.keys())[0]
    text = preprocess(data[id_str])

    return id_str, text

if __name__ == "__main__":
    output_file = open(output_filename, "w", encoding="utf-8")
    if input_filename.endswith(".json.gz"):
        input_file = gzip.open(input_filename, "rt", encoding="utf-8")
    elif input_filename.endswith(".json"):
        input_file = open(input_filename, "r", encoding="utf-8")
    else:
        raise("File extension should be 'json' or 'json.gz'!")

    curr_batch = []
    for i, line in enumerate(input_file):
        data = json.loads(line)
        id_str, text = read_json_line(data)

        if len(text) > 0:
            curr_batch.append({"id_str": id_str, "text": text})

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
