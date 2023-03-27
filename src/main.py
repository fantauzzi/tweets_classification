from pathlib import Path

import hydra
import mlflow as mf
import numpy as np
import torch
import transformers
from datasets import load_dataset
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    # Absolute path to the repo root in the local filesystem
    repo_root = Path('..').resolve()

    # Absolute path to the MLFlow tracking dir, currently supported only in the local filesystem
    tracking_uri = repo_root / params.mlflow.tracking_uri

    models_dir = (repo_root / params.main.models_dir).resolve()
    mf.set_tracking_uri(tracking_uri)  # set_tracking_uri() expects an absolute path

    # If there is no active run then start one
    mf.start_run()
    mf_run = mf.active_run()
    print(f"Active run name is: {mf_run.info.run_name}")

    if params.transformers.seed is not None:
        transformers.set_seed(params.transformers.seed)
    model_file_name = '../models/saved_model'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print('No GPU found')

    emotions = load_dataset('emotion')

    pretrained_model = params.transformers.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

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
        model = (AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels).to(device))

    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    batch_size = 64
    logging_steps = len(emotions_encoded["train"]) // batch_size
    model_name = f"{pretrained_model}-finetuned-emotion"
    output_dir = str(models_dir / model_name)
    training_args = TrainingArguments(output_dir=output_dir,
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

    preds_output_val = trainer.predict(emotions_encoded["validation"])
    print(f'Prediction metrics\n{preds_output_val.metrics}')

    y_preds_val = np.argmax(preds_output_val.predictions, axis=1)
    y_valid = np.array(emotions_encoded["validation"]["label"])

    plot_confusion_matrix(y_preds_val, y_valid, labels)

    preds_output_test = trainer.predict(emotions_encoded["test"])
    print(f'Prediction metrics\n{preds_output_test.metrics}')

    y_preds_test = np.argmax(preds_output_test.predictions, axis=1)
    y_test = np.array(emotions_encoded["test"]["label"])

    plot_confusion_matrix(y_preds_test, y_test, labels)


if __name__ == '__main__':
    main()

"""
TODO
Add a test step -> Done 
Turn the script into an MLFlow project -> Done
Store hyperparameters in a config. file -> Done
Draw charts of the training and validation loss and the confusion matrix under TensorFlow
Tune hyperparameters (using some framework/library)
Version the model
Give the model an API, deploy it
Make a GUI via gradio and/or streamlit

"""
