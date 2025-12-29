import torch
from pytorch_pretrained_bert import BertForTokenClassification, BertTokenizer
import numpy as np
from seqeval.metrics import classification_report, f1_score, accuracy_score 

model_dir = "./model/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tag2idx = {'L-INVESTOR': 0, 'L-INVESTMENT': 1, 'B-INVESTOR': 2, 'L-STARTUP': 3, 'L-PERCHNG': 4, 'L-VAL': 5, 'B-VAL': 6, 'B-FOUNDER': 7, 'U-STARTUP': 8, 'L-FOUNDER': 9, 'O': 10, 'U-BUY': 11, 'U-INVESTOR': 12, 'B-STARTUP': 13, 'I-VAL': 14, 'X': 15, 'I-INVESTMENT': 16, '[CLS]': 17, 'I-ARR': 18, 'B-INVESTMENT': 19, '[SEP]': 20, 'B-PERCHNG': 21, 'B-ARR': 22, 'L-ARR': 23}
idx2tag = {0: 'L-INVESTOR', 1: 'L-INVESTMENT', 2: 'B-INVESTOR', 3: 'L-STARTUP', 4: 'L-PERCHNG', 5: 'L-VAL', 6: 'B-VAL', 7: 'B-FOUNDER', 8: 'U-STARTUP', 9: 'L-FOUNDER', 10: 'O', 11: 'U-BUY', 12: 'U-INVESTOR', 13: 'B-STARTUP', 14: 'I-VAL', 15: 'X', 16: 'I-INVESTMENT', 17: '[CLS]', 18: 'I-ARR', 19: 'B-INVESTMENT', 20: '[SEP]', 21: 'B-PERCHNG', 22: 'B-ARR', 23: 'L-ARR'}

tokenizer = BertTokenizer.from_pretrained("model/vocab.txt", do_lower_case=False)
model = BertForTokenClassification.from_pretrained("model", num_labels = len(tag2idx))
model.to(device=device)
model.eval() # Set to evaluation mode (turns off dropout)

def predict_sentence(sentence):
    tokenized_sentence = tokenizer(sentence)

    # Convert tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids()


y_true = []
y_pred = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
valid_dataloader = torch.load("./artifacts/val_dataloader.pt", weights_only=False)

for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, label_ids = batch

    with torch.no_grad():
        logits = model(input_ids, token_type_ids=None, attention_mask=input_mask,)  # models prediction

    logits = logits.detach().cpu().numpy()
    logits = [list(p) for p in np.argmax(logits, axis=2)] # only pick the index of the highest probability
    
    label_ids = label_ids.to('cpu').numpy()
    input_mask = input_mask.to('cpu').numpy()
    
    for i,mask in enumerate(input_mask):
        temp_1 = [] # Real one
        temp_2 = [] # Predict one
        
        for j, m in enumerate(mask):
            # Mark=0, meaning its a pad word, dont compare, if it is not padding
            if m:
                # if its a valid tag not X, CLS, SEP
                if idx2tag[label_ids[i][j]] != "X" and idx2tag[label_ids[i][j]] != "[CLS]" and idx2tag[label_ids[i][j]] != "[SEP]" : # Exclude the X label
                    temp_1.append(idx2tag[label_ids[i][j]]) # Actual Tag
                    temp_2.append(idx2tag[logits[i][j]]) # Predicted tag
            else:
                break
        y_true.append(temp_1)
        y_pred.append(temp_2)


# print("f1 score: %f"%(f1_score(y_true, y_pred)))
# print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))

print(classification_report(y_true, y_pred,digits=4))        
