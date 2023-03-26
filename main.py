from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

transformers.set_seed(8833)

model_file_name = 'saved_model'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != 'cuda':
    print('No GPU found')

emotions = load_dataset('emotion')

model_ckpt = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

num_labels = 6
already_trained = False
if Path(model_file_name).exists():
    print('Loading from saved model')
    model = (AutoModelForSequenceClassification.from_pretrained(model_file_name).to(device))
    already_trained = True
else:
    print('Resuming from checkpointed model')
    model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=False,
                                  log_level="error")

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)

if not already_trained:
    training_metrics = trainer.train()
    print(training_metrics)
    trainer.save_model(model_file_name)
    already_trained = True


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


labels = emotions["train"].features["label"].names

preds_output = trainer.predict(emotions_encoded["validation"])
print(f'Prediction metrics\n{preds_output.metrics}')

y_preds = np.argmax(preds_output.predictions, axis=1)
y_valid = np.array(emotions_encoded["validation"]["label"])

plot_confusion_matrix(y_preds, y_valid, labels)
