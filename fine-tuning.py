# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import pipeline
import pandas as pd
import os
import pickle
import spacy
import torch
from spacy.training import offsets_to_biluo_tags   
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification
from tqdm import trange
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
nlp = spacy.load("en_core_web_lg")

with open("artifacts/annotated_data_dict.pkl", "rb") as f:
    annotated_data_dict: dict = pickle.load(f)

df = pd.DataFrame(annotated_data_dict)
df.drop(columns="annotations", inplace=True)

def verify_annotations(annot: list, article_text: str):
    for entry in annot:
        for point in entry["points"]:
            text_to_verify = point["text"]
            start = point["start"]
            end = point["end"]
            actual_text = article_text[start:end]
            if text_to_verify != actual_text:
                print(f"Actual text = {actual_text}")
                print(f"Text = {text_to_verify}")

for idx, row in df.iterrows():
    verify_annotations(row["correct_annotations"], row["news_articles"])

LABEL_MAP = {
    'Startup Name': 'STARTUP',
    'Founder Name': 'FOUNDER',
    'Investment Company': 'INVESTOR',
    'Funding Amount': 'INVESTMENT',
    'Revenue': 'ARR',
    'Percent change in revenue': 'PERCHNG',
    'Company Buying Startup': 'BUY',
    "Valuation": "VAL"
}

def mergeIntervals(intervals):
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            if higher[0] <= lower[1]:
                if lower[2] is higher[2]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (lower[0], upper_bound, lower[2])
                else:
                    if lower[1] > higher[1]:
                        merged[-1] = lower
                    else:
                        merged[-1] = (lower[0], higher[1], higher[2])
            else:
                merged.append(higher)

    return merged

print(df.shape)
entities = []
for idx, row in df.iterrows():
    entity = []
    for annot in row["correct_annotations"]:
        ent = LABEL_MAP[annot["label"][0]]
        start = annot["points"][0]["start"]
        end = annot["points"][0]["end"]
        entity.append((start, end, ent))

    entity = mergeIntervals(entity)
    entities.append(entity)

df["entities"] = entities   

# print(df.head())
sentences = []
tags = []
for i in range(len(df)):
    doc = nlp(df["news_articles"][i])
    # print(type(doc))
    ent = df["entities"][i]
    # print(ent)
    tag = offsets_to_biluo_tags(doc, ent)
    # print(len(doc))
    # print(len(tag))
    tmp = pd.DataFrame([list(doc), tag]).T
    # print(tmp)

    loc = []
    for i in range(len(tmp)):
        if tmp[0][i].text == "." and tmp[1][i] == 'O':
            loc.append(i)

    loc.append(len(doc))

    # print(loc)
    last = 0
    data=[]
    for pos in loc:
        data.append([list(doc)[last:pos], tag[last:pos]])
        last = pos

    for d in data:
        tag = ['O' if t == '-' else t for t in d[1]]
        if len(set(tag))>1:
            sentences.append(d[0])
            tags.append(tag)

    

print("\n")
print(len(sentences))
print(len(tags))
# print(sentences)
# print(tags)


tag_vals = set(["X", "[CLS]", "[SEP]"])
for i in range(len(tags)):
    tag_vals = tag_vals.union(tags[i])

idx2tag = {i: t for i, t in enumerate(tag_vals)}
print(idx2tag)

tag2idx = {t: i for i, t in enumerate(tag_vals)}
print(tag2idx)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

tokenized_texts = []
word_piece_labels = []

for word_list, label in zip(sentences, tags):
    # word_list is a list of words (sentence), and label is it's respective list of labels
    temp_lable = ['[CLS]']
    temp_token = ['[CLS]']

    for word, lab in zip(word_list, label):
        # word is a word, lab is the respective tag
        token_list = tokenizer.tokenize(word.text)
        for m, token in enumerate(token_list):
            # List of tokens for 1 word
            temp_token.append(token) # All tokens in list appended to temp_token
            if m==0:
                temp_lable.append(lab) # The tag of the word is appended in the first index
            else:
                temp_lable.append('X') # X is appended for later tokens
    
    temp_lable.append("[SEP]")
    temp_token.append("[SEP]")
    # temp_lable is a list of all the labels for a single sentence
    # temp_token is a list of all tokens for a single sentence

    tokenized_texts.append(temp_token)
    word_piece_labels.append(temp_lable)

print("\n")
print("List of tokens for 1st sentence")
print(tokenized_texts[0])
print(len(tokenized_texts[0]))
print("\n")
print("List of labels for 1st sentence")
print(word_piece_labels[0])
print(len(word_piece_labels[0]))

max_len = max([len(txt) for txt in tokenized_texts])
print("Maximum length of tokens for a sentence")
print(max_len)

MAX_LEN = 512

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts], maxlen=MAX_LEN, dtype="long", padding="post")
print("Number of sentences: ", len(tokenized_texts))
print("Number of sentences in padded token ids: ", len(input_ids))
print(input_ids)
print(len(input_ids[0]))
print(len(input_ids[1]))


tag_ids = pad_sequences([[tag2idx.get(l) for l in lab] for lab in word_piece_labels], maxlen=MAX_LEN, value = tag2idx['O'], padding="post", dtype="long")
print("Number of setences in padded tag ids: ", len(tag_ids))
print(tag_ids)
print(len(tag_ids[0]))
print(len(tag_ids[1]))

attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
print("Attention masks: ")
print(len(attention_masks))
print(len(attention_masks[0]))

tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks = train_test_split(input_ids, tag_ids, attention_masks, random_state=42, test_size=0.3)

print(len(tr_inputs), len(val_inputs), len(tr_tags), len(val_tags), len(tr_masks), len(val_masks))



tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags, dtype=torch.long)
val_tags = torch.tensor(val_tags, dtype=torch.long)
tr_masks = torch.tensor(tr_masks, dtype=torch.long)
val_masks = torch.tensor(val_masks, dtype=torch.long)
print("After tensor")
print(len(tr_inputs), len(val_inputs), len(tr_tags), len(val_tags), len(tr_masks), len(val_masks))

bs=4
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(tag2idx))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

no_decay = ["bias", "gamma", "beta"]
el = list(model.named_parameters())[4]
print(el[0])
print(any(i in el[0] for i in no_decay))
    


FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],  # if there are no ["bias", "gamma", "beta"] in parameter apply weight decay to prevent overfitting force the model to use smaller weights
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], # if there are ["bias", "gamma", "beta"] in parameter we do not apply weight decay
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

epochs = 10
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps)) 
    print(nb_tr_examples)


model_dir = "./model/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_to_save = model.module if hasattr(model, 'module') else model

# Save the weights
torch.save(model_to_save.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))

# Save the configuration (so we know how many labels/layers to load later)
model_to_save.config.to_json_file(os.path.join(model_dir, "config.json"))

# 3. Save the tokenizer (so we process text the exact same way later)
tokenizer.save_vocabulary(model_dir)

torch.save(valid_dataloader, "./artifacts/val_dataloader.pt")
print("Saved Validation dataloader object to ./artifacts/val_dataloader.pt")
print(f"Model saved to {model_dir}")