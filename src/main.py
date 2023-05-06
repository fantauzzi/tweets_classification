from os import system
from pathlib import Path
from shutil import copytree, rmtree

import hydra
import mlflow as mf
import numpy as np
import optuna
import torch
import transformers
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline, \
    DistilBertForSequenceClassification

from utils import info, warning, info_active_run, compute_metrics, get_name_for_run, plot_confusion_matrix, \
    get_eval_f1_from_state_log


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

    """seed = params.transformers.get('seed')
    if seed is not None:
        transformers.set_seed(params.transformers.seed)"""

    ''' Set various paths '''

    # Absolute path to the repo root in the local filesystem
    repo_root = Path('..').resolve()

    # Absolute path to the MLFlow tracking dir; currently supported only in the local filesystem
    tracking_uri = repo_root / params.mlflow.tracking_uri
    mf.set_tracking_uri(tracking_uri)  # set_tracking_uri() expects an absolute path

    # Absolute path to the directory where model and model checkpoints are to be saved and loaded from
    models_path = (repo_root / params.main.models_dir).resolve()

    # Absolute path the fine-tuned model is saved to/loaded from
    tuned_model_path = models_path / params.main.fine_tuned_model_dir

    # Absolute path where the hyperparameter values for the fine-tuned model are saved to/loaded from
    params_override_path = models_path / 'params_override.yaml'

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

    num_labels = 6

    info(f'Training set contains {len(emotions_encoded["train"])} samples')
    model_name = f"{pretrained_model}-finetuned-emotion"
    output_dir = str(models_path / model_name)

    def train(model_init, params_ovveride=None):
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

        if params_ovveride is not None:
            for key, value in params_ovveride.items():
                setattr(training_args, key, value)

        callbacks = [
            transformers.EarlyStoppingCallback(early_stopping_patience=params.transformers.early_stopping_patience)]

        trainer = Trainer(model=model_init(),
                          args=training_args,
                          compute_metrics=compute_metrics,
                          train_dataset=emotions_encoded["train"],
                          eval_dataset=emotions_encoded["validation"],
                          tokenizer=tokenizer,
                          callbacks=callbacks)

        res = trainer.train()
        info(f'Model tuning completed with results {res}')
        return res, trainer

    def get_model() -> DistilBertForSequenceClassification:
        """
        Returns the pre-trained model.
        :return: the model instance. It is already in the GPU memory if a GPU is available.
        """
        the_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model,
                                                                       num_labels=num_labels).to(device)
        return the_model

    labels = emotions["train"].features["label"].names

    ''' Hyperparameters tuning, starting from a pre-trained model '''

    if params.train.tune:
        info('Starting hyperparameters tuning')
        with mf.start_run(run_name=get_name_for_run(), nested=True, description='hyperparemeters tuning'):
            info_active_run()

            def trial_CB(trial: optuna.trial.Trial) -> float:
                """
                Callback meant to be passed to Optuna for its optimization process, performs the computation for
                one trial and returns the value of its resulting evaluation metric (F1).
                :param trial: instance of Optuna Trial that Optuna will pass to the callback during the optimization
                process.
                :return: the evaluation metric for the current trial
                """

                # random_state1 = trial.study.sampler._rng._bit_generator.state
                # random_state2 = trial.study.sampler._random_sampler._rng._bit_generator.state
                info(
                    f'Starting trial No. {trial.number} of study {trial.study.study_name}, wich contains a total of {len(trial.study.trials)} trial(s) so far')
                with mf.start_run(run_name=get_name_for_run(),
                                  nested=True,
                                  description='trial for hyperparameters tuning'):
                    trial_params = {
                        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
                        'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size',
                                                                                 [64, 128, 192, 256])}
                    info('Parameters drawn for this trial:')
                    for key, value in trial_params.items():
                        info(f'  {key} = {value}')
                    res, trainer = train(model_init=get_model, params_ovveride=trial_params)
                    ''' Update information on the best trial so far as needed, and ensure the best trained model so
                    far is saved '''

                    eval_f1 = get_eval_f1_from_state_log(trainer.state.log_history, res.global_step)
                    if not trial.study.best_trials or eval_f1 > trial.study.best_value:
                        # if self._best_eval_f1 is None or self._best_eval_f1 < eval_f1:
                        if not trial.study.best_trials:
                            info(f'First trial completed with evaluation metric {eval_f1}')
                        else:
                            info(
                                f'Current trial (No. {trial.number}) improved the evaluation metric from {trial.study.best_value} to {eval_f1}')
                        # self._best_eval_f1 = eval_f1
                        if Path(tuned_model_path).exists():
                            info(
                                f'Overwriting {tuned_model_path} with model with best tuned hyperparameters so far')
                            rmtree(tuned_model_path)
                        else:
                            info(
                                f'Checkpoint with best tuned hyperparameters so far is {trainer.state.best_model_checkpoint}')
                            info(
                                f'Saving model (checkpoint) with best tuned hyperparameters so far into {tuned_model_path}')
                        copytree(trainer.state.best_model_checkpoint, tuned_model_path)
                        info(f'Saving best choice of hyperparemeters so far into {params_override_path}')
                        OmegaConf.save(trial_params, params_override_path)
                    else:
                        info(f'Trial No. {trial.number} completed with evaluation metric {eval_f1}')
                    # trial.study.sampler._rng._bit_generator.state = random_state1
                    # trial.study.sampler._random_sampler._rng._bit_generator.state = random_state2
                    return eval_f1

            study_name = params.fine_tuning.study_name
            optuna_db = params.fine_tuning.optuna_db
            trials_storage = f'sqlite:///../db/{optuna_db}'
            sampler = optuna.samplers.TPESampler()
            pruner = optuna.pruners.NopPruner()
            study = optuna.create_study(study_name=study_name,
                                        storage=trials_storage,
                                        load_if_exists=params.fine_tuning.resume_previous,
                                        sampler=sampler,
                                        pruner=pruner,
                                        direction='maximize')
            study.optimize(func=trial_CB, n_trials=params.fine_tuning.n_trials)

            mf.log_param('study_name', study.study_name)
            mf.log_param('best_trial', study.best_trial.number)
            mf.log_param('best_value', study.best_value)
            mf.log_param('best_params', study.best_params)

            info(f'Hyperparameters tuning completed')
            info(f'  study_name {study.study_name}')
            info(f'  best_trial {study.best_trial.number}')
            info(f'  best_value {study.best_value}')
            info(f'  best_params {study.best_params}')

    ''' Fine-tune the model if requested. Default hyperparameter values are taken from the config/params.yaml file, but 
    are then overridden by values taken from the models/saved_models/best_trial.yaml file if such file exists '''

    if params.train.fine_tune:
        with mf.start_run(run_name=get_name_for_run(), nested=True, description='pre-trained model fine-tuning'):
            info_active_run()
            params_override = None
            if Path(params_override_path).exists():
                info(f'Loading parameters overried from {params_override_path}')
                params_override = dict(OmegaConf.load(params_override_path))
            res, trainer = train(model_init=get_model, params_ovveride=params_override)
            info(f'Mode fine tuning results: {res}')
            trainer.save_model(tuned_model_path)

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

        ''' Validate the model that has just been fine-tuned'''

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
        info('Starting validation and test of the inference pipeline')
        with mf.start_run(run_name=get_name_for_run(), nested=True, description='inference testing with saved model'):
            info_active_run()
            model = AutoModelForSequenceClassification.from_pretrained(tuned_model_path)
            tokenizer = AutoTokenizer.from_pretrained(tuned_model_path)
            pipe = pipeline(model=model, task='text-classification', tokenizer=tokenizer, device=0)

            val_pred = pipe(emotions['validation']['text'])
            val_pred_labels = np.array([int(item['label'][-1]) for item in val_pred])
            y_val = np.array(emotions["validation"]["label"])
            eval_f1 = f1_score(y_val, val_pred_labels, average="weighted")
            eval_acc = accuracy_score(y_val, val_pred_labels)
            info(
                f'Validating inference pipeline with model loaded from {tuned_model_path} - dataset contains {len(y_val)} samples')
            info(f'Validation f1 is {eval_f1} and validation accuracy is {eval_acc}')
            mf.log_params({'eval_f1': eval_f1, 'eval_acc': eval_acc})

            fig_test = plot_confusion_matrix(val_pred_labels, y_val, labels, False)
            mf.log_figure(fig_test, 'pipeline_validation_confusion_matrix.png')

            test_pred = pipe(emotions['test']['text'])
            test_pred_labels = np.array([int(item['label'][-1]) for item in test_pred])
            y_test = np.array(emotions["test"]["label"])
            test_f1 = f1_score(y_test, test_pred_labels, average="weighted")
            test_acc = accuracy_score(y_test, test_pred_labels)
            info(
                f'Testing inference pipeline with model loaded from {tuned_model_path} - dataset contains {len(y_test)} samples')
            info(f'Test f1 is {test_f1} and test accuracy is {test_acc}')
            mf.log_params({'test_f1': test_f1, 'test_acc': test_acc})

            fig_test = plot_confusion_matrix(test_pred_labels, y_test, labels, False)
            mf.log_figure(fig_test, 'pipeline_test_confusion_matrix.png')

            info('Validation and test of the inference pipeline completed')


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
