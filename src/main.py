from os import system
from pathlib import Path
from shutil import copytree, rmtree

import hydra
import mlflow as mf
import numpy as np
import optuna as opt
import torch
import transformers
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from optuna.pruners import NopPruner
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline, \
    DistilBertForSequenceClassification
from transformers.integrations import MLflowCallback

from utils import info, warning, info_active_run, MLFlowTrialCB, compute_metrics, get_name_for_run, implies, \
    plot_confusion_matrix


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    """
    Tune the hyperparameters for best fine-tuning the model
    :param params: the configuration parameters passed by Hydra
    """

    ''' Set-up Hydra '''
    info(f'Current working directory is {Path.cwd()}')
    hydra_output_dir = OmegaConf.to_container(HydraConfig.get().runtime)['output_dir']
    info(f'Output dir is {hydra_output_dir}')

    ''' Set the RNG seed to make runs reproducible '''

    seed = params.transformers.get('seed')
    if seed is not None:
        transformers.set_seed(params.transformers.seed)

    ''' Set various path '''

    # Absolute path to the repo root in the local filesystem
    repo_root = Path('..').resolve()

    # Absolute path to the MLFlow tracking dir; currently supported only in the local filesystem
    tracking_uri = repo_root / params.mlflow.tracking_uri
    mf.set_tracking_uri(tracking_uri)  # set_tracking_uri() expects an absolute path

    # Absolute path to the directory where model and model checkpoints are to be saved and loaded from
    models_path = (repo_root / params.main.models_dir).resolve()

    # Absolute path the fine-tuned model are saved to/loaded from
    tuned_model_path = models_path / params.main.fine_tuned_model_dir

    # Absolute path where the hyperparameter values for the fine-tuned model are saved to/loaded from
    best_trial_path = models_path / 'best_trial.yaml'

    ''' If there is no MLFlow run currently ongoing, then start one. Note that is this script has been started from
     shell with `mlflow run` then a run is ongoing already, no need to start it'''

    if mf.active_run() is None:
        info('No active MLFlow run, starting one now')
        mf.start_run(run_name=get_name_for_run())

    info_active_run()

    if not models_path.exists():
        models_path.mkdir()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        warning(f'No GPU found, device type is {device.type}')

    # Save the output of nvidia-smi (GPU info) into a text file, log it with MLFlow then delete the file
    nvidia_info_filename = 'nvidia-smi.txt'
    nvidia_info_path = repo_root / nvidia_info_filename
    system(f'nvidia-smi -q > {nvidia_info_path}')
    mf.log_artifact(str(nvidia_info_path))
    nvidia_info_path.unlink(missing_ok=True)

    emotions = load_dataset('emotion')
    pretrained_model = params.transformers.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

    num_labels = 6  # Yuck!

    info(f'Training set contains {len(emotions_encoded["train"])} samples')
    model_name = f"{pretrained_model}-finetuned-emotion"
    output_dir = str(models_path / model_name)

    def optimize(model_init,
                 early_stopping_patience,
                 overriding_params=None,
                 hp_space=None,
                 optuna_sampler=None):
        assert implies(hp_space is None, optuna_sampler is None)

        training_args = TrainingArguments(output_dir=output_dir,
                                          num_train_epochs=params.transformers.epochs,
                                          learning_rate=2e-5,
                                          per_device_train_batch_size=params.transformers.batch_size,
                                          per_device_eval_batch_size=params.transformers.test_batch_size,
                                          weight_decay=0.01,
                                          evaluation_strategy="epoch",
                                          disable_tqdm=False,
                                          push_to_hub=False,
                                          log_level="error",
                                          logging_strategy='epoch',
                                          report_to=['mlflow'],
                                          logging_first_step=False,
                                          save_strategy='epoch',
                                          load_best_model_at_end=True,
                                          seed=params.transformers.get('seed'))

        if overriding_params is not None:
            for key, value in overriding_params.items():
                setattr(training_args, key, value)

        callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]

        trainer = Trainer(model=model_init() if hp_space is None else None,
                          args=training_args,
                          compute_metrics=compute_metrics,
                          train_dataset=emotions_encoded["train"],
                          eval_dataset=emotions_encoded["validation"],
                          tokenizer=tokenizer,
                          model_init=model_init if hp_space is not None else None,
                          callbacks=callbacks)

        if hp_space is not None:
            def compute_objective(metrics: dict) -> float:
                return metrics['eval_f1']

            trainer.remove_callback(MLflowCallback)
            trainer.add_callback(MLFlowTrialCB())
            study_name = params.fine_tuning.study_name
            trials_storage = f'sqlite:///../db/{study_name}.db'
            ''' Careful: at the end of the hyperparameters tuning process, `trainer` contains the model as trained
            by the *last* trial, not by the *best* trial '''
            res = trainer.hyperparameter_search(hp_space=hp_space,
                                                n_trials=params.fine_tuning.n_trials,
                                                direction='maximize',
                                                compute_objective=compute_objective,
                                                sampler=optuna_sampler,
                                                study_name=study_name,
                                                storage=trials_storage,
                                                load_if_exists=params.fine_tuning.resume_previous,
                                                pruner=NopPruner())
            info(f'Best run: {res}')
        else:
            res = trainer.train()
            info(f'Fine tuning results: {res}')

        return res, trainer

    ''' Find the best value for hyperparameters that govern the model's fine-tuning, if requested '''

    def get_model(trial: opt.trial.Trial | None = None) -> DistilBertForSequenceClassification:
        """
        Returns the model. The signature of the function is such that it can be invoked by Optuna during
        hyperparameters tuning.
        :param trial: an Optuna Trial object, or None.
        :return: the model instance. It is already in the GPU memory if a GPU is available.
        """
        the_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model,
                                                                       num_labels=num_labels).to(device)
        return the_model

    labels = emotions["train"].features["label"].names

    if params.train.tune:
        with mf.start_run(run_name=get_name_for_run(), nested=True, description='hyperparemeters tuning'):
            info_active_run()

            def hp_space(trial: opt.trial.Trial) -> dict:
                res = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
                    'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size',
                                                                             [64, 128, 192, 256]),
                }
                return res

            optuna_sampler = None if seed is None else TPESampler(seed=seed)
            best_trial, trainer = optimize(model_init=get_model,
                                           early_stopping_patience=params.transformers.early_stopping_patience,
                                           overriding_params=None,
                                           hp_space=hp_space,
                                           optuna_sampler=optuna_sampler)
            info(f'Hyperparameters tuning completed')
            info(f'  Best model has optimizing metric {trainer.state.best_metric}')
            info(f'  achieved at epoch {trainer.state.epoch}')
            info(f'  of global step {trainer.state.global_step}')
            checkpoint_path = trainer.state.best_model_checkpoint
            info(f'  saved in checkpoint {checkpoint_path}')
            info('  it has the following tuned hyperparameters value:')
            for key, value in trainer.state.trial_params.items():
                info(f'    {key} = {value}')
            for log_entry in trainer.state.log_history:
                if log_entry['step'] == trainer.state.global_step and log_entry.get('eval_loss') is not None:
                    info('  evaluation stats:')
                    for stat_name, stat_value in log_entry.items():
                        info(f'    {stat_name} = {stat_value}')
                    assert log_entry['eval_loss'] == trainer.state.best_metric

            # TODO see how much of the state.log_history you want to log with MLFlow, also to make charts
            OmegaConf.save(best_trial.hyperparameters, best_trial_path)  # Could also save how many epochs
            if Path(tuned_model_path).exists():
                info(
                    f'Overwriting {tuned_model_path} with model with newly tuned hyperparameters')
                rmtree(tuned_model_path)
            else:
                info(f'Saving model with tuned hyperparameters into {tuned_model_path}')
            copytree(checkpoint_path, tuned_model_path)

    ''' Fine-tune the model if requested. Default hyperparameter values are taken from the config/params.yaml file, but 
    are then overridden by values taken from the models/saved_models/best_trial.yaml file if such file exists '''

    if params.train.fine_tune:
        with mf.start_run(run_name=get_name_for_run(), nested=True, description='pre-trained model fine-tuning'):
            info_active_run()
            best_trial_params = None
            if Path(best_trial_path).exists():
                info(f'Loading tuned hyper-parameters from {best_trial_path}')
                best_trial_params = dict(OmegaConf.load(best_trial_path))
            training_metrics, trainer = optimize(model_init=get_model,
                                                 early_stopping_patience=params.transformers.early_stopping_patience,
                                                 overriding_params=best_trial_params)

            trainer.save_model(tuned_model_path)

        ''' Validate the model that has just been fine-tuned'''

        def test_model(trainer: Trainer, dataset: Dataset, description: str, confusion_matrix_filename: str) -> None:
            with mf.start_run(run_name=get_name_for_run(), nested=True, description=description):
                info_active_run()
                info(f'{description} - dataset contains {len(dataset)} samples')
                preds_output_val = trainer.predict(dataset)
                info(f'Result metrics:\n{preds_output_val.metrics}')

                y_preds_val = np.argmax(preds_output_val.predictions, axis=1)
                y_valid = np.array(dataset["label"])

                fig_val = plot_confusion_matrix(y_preds_val, y_valid, labels, False)
                mf.log_figure(fig_val, confusion_matrix_filename)

        test_model(trainer=trainer,
                   dataset=emotions_encoded["validation"],
                   description='Model validation after fine-tuning',
                   confusion_matrix_filename='validation_confusion_matrix.png')

        ''' Test the model that has just been fine-tuned'''

        test_model(trainer=trainer,
                   dataset=emotions_encoded["test"],
                   description='Model testing after fine tuning',
                   confusion_matrix_filename='test_confusion_matrix.png')

    ''' Test the saved fine-tuned model if required. That is the same model that would be used for inference '''

    if params.train.test:
        with mf.start_run(run_name=get_name_for_run(), nested=True, description='inference testing with saved model'):
            info_active_run()
            model = AutoModelForSequenceClassification.from_pretrained(tuned_model_path)
            tokenizer = AutoTokenizer.from_pretrained(tuned_model_path)
            pipe = pipeline(model=model, task='text-classification', tokenizer=tokenizer, device=0)

            val_pred = pipe(emotions['validation']['text'])
            val_pred_labels = np.array([int(item['label'][-1]) for item in val_pred])
            y_val = np.array(emotions["validation"]["label"])
            f1 = f1_score(y_val, val_pred_labels, average="weighted")
            acc = accuracy_score(y_val, val_pred_labels)
            info(
                f'Validating inference pipeline with model loaded from {tuned_model_path} - dataset contains {len(y_val)} samples')
            info(f'Validation f1 is {f1} and validation accuracy is {acc}')

            test_pred = pipe(emotions['test']['text'])
            test_pred_labels = np.array([int(item['label'][-1]) for item in test_pred])
            y_test = np.array(emotions["test"]["label"])
            f1 = f1_score(y_test, test_pred_labels, average="weighted")
            acc = accuracy_score(y_test, test_pred_labels)
            info(
                f'Testing inference pipeline with model loaded from {tuned_model_path} - dataset contains {len(y_test)} samples')
            info(f'Test f1 is {f1} and test accuracy is {acc}')
            # TODO log these metrics with MLFlow ---^

            fig_test = plot_confusion_matrix(test_pred_labels, y_test, labels, False)
            mf.log_figure(fig_test, 'pipeline_test_confusion_matrix.png')


if __name__ == '__main__':
    main()

"""
TODO Add a test step -> Done
Turn the  script into an MLFlow project -> Done
Store hyperparameters in a config.file -> Done
Introduce proper logging -> Done
Tune hyperparameters(using some framework / library) -> Done
(Re -)train and save -> Done
Do proper testing of inference from saved model -> Done
Draw charts of the training and validation loss and the confusion matrix under MLFlow -> Done
Implement early-stopping -> Done
Make a nested run for every Optuna trial -> Done
Send the training to the cloud -> Done
Make sure GPU info is logged -> Done
Ensure parameters for every trial are logged, at least the changing ones -> Done
Split MLFlowTrialCB() in its own file -> Done
Try GPU on Amazon/google free service -> Done
Have actually random run names even with a set random seed -> Done
Fix reproducibility -> Done
Make sure hyperparameters search works correctly -> Done
Can fine-tuning be interrupted and resumed? -> Done, yes!
Fix up call to optimize() -> Done

Provide an easy way to coordinate the trial info (in the SQLite DB) with the run info in MLFlow
Log with MLFlow the Optuna trial id of every nested run, also make sure the study name is logged

Tag the best nested run as such, will have to remove and re-assign the tag of best nested run as needed 
Optimize hyper-parameters tuning such that it saves the best model so far at every trial, so it doesn't have to be
    computed again later (is it even possible?)
Log computation times
Make a GUI via gradio and / or streamlit
Version the saved model(also the dataset?)
Follow Andrej recipe
Plot charts to MLFlow for debugging of the training process, as per Andrej's lectures
Give the model an API, deploy it, unit-test it
"""
