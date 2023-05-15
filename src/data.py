import json
import torch
# import pandas as pd

class TransformersData(torch.utils.data.Dataset):
    def __init__(self, examples, label_map, tokenizer, max_seq_length=512, has_token_type_ids=False, with_label=True):
        self.examples = examples
        self.label_map = label_map

        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.with_label = with_label
        self.has_token_type_ids = has_token_type_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoded_input = self.tokenizer(ex[0], padding="max_length", truncation=True, max_length=self.max_seq_length)
        input_ids = torch.tensor(encoded_input["input_ids"], dtype=torch.long)
        input_mask = torch.tensor(encoded_input["attention_mask"], dtype=torch.long)
        if self.has_token_type_ids:
            token_type_ids = torch.tensor(encoded_input["token_type_ids"], dtype=torch.long)

        if self.with_label:
            label_ids = torch.zeros(len(self.label_map))
            for label_name in ex[1]:
                label_ids[self.label_map[label_name]] = 1.0

            if self.has_token_type_ids:
                return input_ids, input_mask, token_type_ids, label_ids
            else:
                return input_ids, input_mask, label_ids

        if self.has_token_type_ids:
            return input_ids, input_mask, token_type_ids
        else:
            return input_ids, input_mask


def get_examples(filename, with_label=True):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    examples = []
    for (i, line) in enumerate(lines):
        line = json.loads(line)
        text = str(line["text"])
        if with_label:
            labels = line["labels"] # should be a list of label names
            examples.append([text, labels])
        else:
            examples.append([text])

    # examples = pd.read_json(filename, orient="records", lines=True)
    # examples = [[a[1]] for a in examples.values.tolist()]

    return examples
