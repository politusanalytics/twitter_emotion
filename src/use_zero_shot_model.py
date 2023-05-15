from transformers import pipeline
import json

classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

candidate_labels = ["nervousness", "relief", "pride", "embarrassment", "grief"]
hypothesis_template = "This text has {}"

input_filename = "data/annotation/turkey_tweets_wPredictions_using_go_emotions.json"
out_filename = "data/annotation/turkey_tweets_wPredictions_using_go_emotions_wZeroShot.json"
out_file = open(out_filename, "w", encoding="utf-8")

with open(input_filename, "r", encoding="utf-8") as f:
    for line in f:
        curr_d = json.loads(line)

        pred = classifier(curr_d["text"], candidate_labels,# hypothesis_template=hypothesis_template,
                          multi_label=True)

        # print(pred["scores"])
        out_preds = []
        for i,score in enumerate(pred["scores"]):
            if score > 0.95:
                out_preds.append(candidate_labels[i])

        curr_d["zero_shot_preds"] = out_preds

        out_file.write(json.dumps(curr_d, ensure_ascii=False, sort_keys=True) + "\n")

out_file.close()
