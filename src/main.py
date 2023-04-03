import logging
from pathlib import Path

import hydra
import mlflow as mf
import numpy as np
import optuna as opt
import torch
import transformers
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    """
    Tune the hyperparameters for best fine-tuning the model
    :param params: the configuration parameters passed by Hydra
    """

    ''' Set-up logging and Hydra '''

    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()
    info = logger.info
    warning = logger.warning
    info(f'Current working directory is {Path.cwd()}')
    hydra_output_dir = OmegaConf.to_container(HydraConfig.get().runtime)['output_dir']
    info(f'Output dir is {hydra_output_dir}')

    ''' Set the RNG seed to make runs reproducible '''

    if params.transformers.seed is not None:
        transformers.set_seed(params.transformers.seed)

    ''' Set various directories '''

    # Absolute path to the repo root in the local filesystem
    repo_root = Path('..').resolve()

    # Absolute path to the MLFlow tracking dir; currently supported only in the local filesystem
    tracking_uri = repo_root / params.mlflow.tracking_uri
    mf.set_tracking_uri(tracking_uri)  # set_tracking_uri() expects an absolute path

    # Absolute path to the directory where model and model checkpoints are be saved and loaded from
    models_path = (repo_root / params.main.models_dir).resolve()

    ''' If there is no MLFlow run currently ongoing, then start one '''

    # If the script has been started from shell with `mflow run` then a run is ongoing already, no need to start it
    mf.start_run()
    mf_run = mf.active_run()
    info(f"Active run name is: {mf_run.info.run_name}")

    # Model to be loaded; if not found, a model is optimized and then saved there
    fine_tuned_model_path = models_path / params.main.fine_tuned_model_dir

    # Check GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        warning(f'No GPU found, device type is {device.type}')

    emotions = load_dataset('emotion')
    pretrained_model = params.transformers.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

    num_labels = 6
    already_trained = False

    if Path(fine_tuned_model_path).exists():
        info(f'Loading model from directory f{fine_tuned_model_path}')
        model = (AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path).to(device))
        already_trained = True
    else:
        print(f'Model f{fine_tuned_model_path} not found, going to download pre-trained model for fine-tuning')
        model = None
        # model = (AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels).to(device))

    print(model)

    # logging_steps = len(emotions_encoded["train"]) // params.transformers.batch_size
    info(f'Training set contains {len(emotions_encoded["train"])} samples')
    model_name = f"{pretrained_model}-finetuned-emotion"
    output_dir = str(models_path / model_name)

    training_args = TrainingArguments(output_dir=output_dir,
                                      num_train_epochs=params.transformers.epochs,
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=params.transformers.batch_size,
                                      per_device_eval_batch_size=params.transformers.batch_size,
                                      weight_decay=0.01,
                                      evaluation_strategy="epoch",
                                      disable_tqdm=False,
                                      # logging_steps=logging_steps,
                                      push_to_hub=False,
                                      log_level="error",
                                      logging_strategy='epoch',
                                      report_to=['mlflow'],
                                      logging_first_step=True,
                                      save_strategy='epoch')

    def model_init(trial: opt.trial.Trial):
        the_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels).to(
            device)
        return the_model

    def compute_metrics(pred):
        ground_truth = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(ground_truth, preds, average="weighted")
        acc = accuracy_score(ground_truth, preds)
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(model=model,
                      args=training_args,
                      compute_metrics=compute_metrics,
                      train_dataset=emotions_encoded["train"],
                      eval_dataset=emotions_encoded["validation"],
                      tokenizer=tokenizer,
                      model_init=model_init)

    if not already_trained:
        def hp_space(trial: opt.trial.Trial) -> dict:
            res = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
            }
            return res

        def compute_objective(metrics: dict) -> float:
            return metrics['eval_f1']

        best_run = trainer.hyperparameter_search(hp_space=hp_space,
                                                 n_trials=3,
                                                 direction='maximize',
                                                 compute_objective=compute_objective)
        # print(training_metrics)
        info(f'Best run: {best_run}')
        trainer.save_model(fine_tuned_model_path)
        already_trained = True

    def plot_confusion_matrix(y_preds, y_true, labels, show=True):
        cm = confusion_matrix(y_true, y_preds, normalize="true")
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
        plt.title("Normalized confusion matrix")
        if show:
            plt.show()
        return fig

    labels = emotions["train"].features["label"].names

    info(f'Validation set contains {len(emotions_encoded["validation"])} samples')
    preds_output_val = trainer.predict(emotions_encoded["validation"])
    info(f'Validation metrics\n{preds_output_val.metrics}')

    y_preds_val = np.argmax(preds_output_val.predictions, axis=1)
    y_valid = np.array(emotions_encoded["validation"]["label"])

    fig_val = plot_confusion_matrix(y_preds_val, y_valid, labels, False)
    mf.log_figure(fig_val, 'validation_confusion_matrix.png')

    info(f'Test set contains {len(emotions_encoded["test"])} samples')
    preds_output_test = trainer.predict(emotions_encoded["test"])
    info(f'Test metrics\n{preds_output_test.metrics}')
    mf.log_metrics(preds_output_test.metrics)

    y_preds_test = np.argmax(preds_output_test.predictions, axis=1)
    y_test = np.array(emotions_encoded["test"]["label"])

    fig_test = plot_confusion_matrix(y_preds_test, y_test, labels, False)
    mf.log_figure(fig_test, 'test_confusion_matrix.png')


if __name__ == '__main__':
    main()

"""
TODO
Add a test step -> Done 
Turn the script into an MLFlow project -> Done
Store hyperparameters in a config. file -> Done
Introduce proper logging -> Done
Tune hyperparameters (using some framework/library) -> Done
(Re-)train and save the best model after hyperparameters tuning
Version the saved model (also the dataset?)
Draw charts of the training and validation loss and the confusion matrix under MLFlow -> Done
Follow Andrej recipe
Plot charts to MLFlow for debugging of the training process, as per Andrej's lectures
Implement early stopping
Give the model an API, deploy it
Make a GUI via gradio and/or streamlit

"""
