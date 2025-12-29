# Startup News NER (Named Entity Recognition)

A domain-specific Named Entity Recognition (NER) system designed to extract critical business intelligence from startup news articles.
This project fine-tunes a **BERT (Bidirectional Encoder Representations from Transformers)** model to identify high-value entities such as startups, founders, funding amounts, and valuations.

## 📌 Project Overview

Standard NER models often fail to capture the nuances of financial news, confusing general organizations with specific startups or missing complex monetary values.

This repository implements a custom training pipeline that:

1. **Preprocesses Data:** Uses **Spacy** to align character-level annotations to token-level tags (BILUO scheme).
2. **Fine-Tunes BERT:** Adapts a pre-trained `bert-base-cased` model to understand startup-specific context.
3. **Evaluates Performance:** Uses **Seqeval** to calculate strict F1-scores and classification reports.

## 🏷️ Supported Entities

The model is trained to recognize and classify the following entities:

| Label | Description | Example |
| --- | --- | --- |
| **STARTUP** | The startup company being discussed | *Harness, Stripe, OpenAI* |
| **FOUNDER** | Name of the startup founder(s) | *Jyoti Bansal, Elon Musk* |
| **INVESTOR** | VC firms, angel investors, or banks | *Sequoia, Menlo Ventures* |
| **INVESTMENT** | Amount raised in a funding round | *$240 million, $50M* |
| **VAL** | Valuation of the company | *$5.5 billion* |
| **ARR** | Annual Recurring Revenue figures | *$250 million ARR* |
| **BUY** | Acquiring company (M&A) | *Cisco, Microsoft* |

## 📂 Repository Structure

```bash
ner_startup/
├── artifacts/              # Stores processed data dictionaries and DataLoaders
│   ├── annotated_data_dict.pkl
│   └── val_dataloader.pt   # Saved validation batch for consistent evaluation
├── model/                  # Saved model weights (pytorch_model.bin) and config
├── fine-tuning.py          # Main script for data processing and training
├── evaluation.py           # Script for generating F1 scores and classification reports
├── annotated_data.csv      # Raw dataset containing news text and annotations
└── requirements.txt        # Python dependencies

```

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/arnavxx7/ner_startup.git
cd ner_startup
pip install -r requirements.txt
python -m spacy download en_core_web_lg

```

### 2. Training the Model (`fine-tuning.py`)

The `fine-tuning.py` script handles the entire training lifecycle:

* **Data Preparation:** Loads annotated data, uses Spacy to generate tags, and converts words into BERT-compatible tokens.
* **Tokenization:** Handles sub-word tokenization and padding using `keras.preprocessing`.
* **Training Loop:** Runs a custom PyTorch training loop for 10 epochs using the Adam optimizer.
* **Artifact Saving:** Saves the trained model weights to `./model/` and the validation dataloader to `./artifacts/`.

```bash
python fine-tuning.py

```

### 3. Evaluation (`evaluation.py`)

Once training is complete, run the evaluation script to test the model's performance on the validation set. This script:

* Loads the saved model and the validation dataloader.
* Performs inference on the validation batch.
* Filters out special tokens (`[CLS]`, `[SEP]`) and sub-word padding (`X`).
* Generates a detailed **Classification Report** and **F1 Score**.

```bash
python evaluation.py

```

## 🧠 Technical Details

* **Model Architecture:** `BertForTokenClassification` (Pre-trained `bert-base-cased`).
* **Optimizer:** Adam with custom parameter grouping (weight decay applied to weights, but not bias/gamma/beta).
* **Metrics:** Precision, Recall, and F1-score via `seqeval`.
* **Data Handling:** Uses `TensorDataset` and `DataLoader` for efficient batch processing on GPU/CPU.

## 📊 Sample Results

*Example output from `evaluation.py`:*

```text
              precision    recall  f1-score   support

         ARR     0.6000    0.6000    0.6000         5
     FOUNDER     0.6667    1.0000    0.8000         2
  INVESTMENT     1.0000    0.1667    0.2857         6
    INVESTOR     0.9709    0.8696    0.9174       115
     PERCHNG     0.0000    0.0000    0.0000         3
     STARTUP     1.0000    0.2500    0.4000         4

   micro avg     0.8045    0.7698    0.7868       139
   macro avg     0.5297    0.3608    0.3754       139
weighted avg     0.9064    0.7698    0.8160       139

```

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

[MIT](https://choosealicense.com/licenses/mit/)
