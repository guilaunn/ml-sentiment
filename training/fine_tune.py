import os
import torch
from datasets import load_dataset, Features, Value
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Configurações
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
OUTPUT_DIR = "./models/sentiment_model"
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5

# 1. Definição explícita do schema, com o nome de coluna correto "overall_rating"
features = Features({
    "reviewer_id": Value("string"),
    "review_text": Value("string"),
    "overall_rating": Value("int32")
})

# 2. Carregar dataset CSV com schema
dataset = load_dataset(
    "csv",
    data_files="https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv",
    features=features,
)

# 3. Filtra o dataset para remover as linhas com review_text nulo ou vazio
dataset = dataset.filter(lambda example: example["review_text"] is not None and len(example["review_text"].strip()) > 0)

# 4. Mapear as estrelas para rótulos binários (0 ou 1)
def map_labels(example):
    if example["overall_rating"] in [1, 2]:
        example["labels"] = 0
    elif example["overall_rating"] in [4, 5]:
        example["labels"] = 1
    else:
        example["labels"] = -1
    return example

dataset = dataset.map(map_labels)

# 5. Filtra as notas neutras (onde labels é -1)
dataset = dataset.filter(lambda example: example["labels"] != -1)

# 6. Dividir em treino/teste (80/20)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

# Tokenizador
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    return tokenizer(batch["review_text"], truncation=True, padding=True, max_length=128)

# 7. Mapear o preprocessamento
dataset_processed = dataset.map(preprocess, batched=True)

# Mapear labels para id2label e label2id
id2label = {0: "negativo", 1: "positivo"}
label2id = {"negativo": 0, "positivo": 1}

# Configurar modelo
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# Argumentos de treino
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_processed["train"],
    eval_dataset=dataset_processed["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Treinar
trainer.train()

# Salvar modelo e tokenizer
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Modelo salvo em {OUTPUT_DIR}")