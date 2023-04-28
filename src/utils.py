import json
import logging
import os

import mlflow as mf
import transformers
from transformers import TrainingArguments
from transformers.integrations import MLflowCallback
from transformers.utils import flatten_dict
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
_logger = logging.getLogger()
info = _logger.info
warning = _logger.warning


def xor(a: bool, b: bool) -> bool:
    return (a and not b) or (b and not a)


def info_active_run():
    """
    Logs an info message with the indication of the currently active MLFlow run, or that there is no active run.
    """
    mf_run = mf.active_run()
    if mf_run is None:
        info('No MLFlow run is active')
    else:
        info(f"Active MLFlow run has name {mf_run.info.run_name} and ID {mf_run.info.run_id}")


class MLFlowTrialCB(MLflowCallback):
    """ A callback that starts and stops an MLFlow nested run at the beginning and end of training, meant to
    encompass one Optuna trial for hyperparameters tuning """

    def __init__(self):
        super().__init__()
        self._nested_run_id = None

    def _log_params_with_mlflow(self, model, args, state):
        assert self._initialized
        if not state.is_world_process_zero:
            return
        combined_dict = args.to_dict()
        if hasattr(model, "config") and model.config is not None:
            model_config = model.config.to_dict()
            combined_dict = {**model_config, **combined_dict}
        combined_dict = flatten_dict(combined_dict) if self._flatten_params else combined_dict

        # remove params that are too long for MLflow
        for name, value in list(combined_dict.items()):
            # internally, all values are converted to str in MLflow
            if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                warning(
                    f'Trainer is attempting to log a value of "{value}" for key "{name}" as a parameter. MLflow\'s'
                    " log_param() only accepts values no longer than 250 characters so we dropped this attribute."
                    " You can use `MLFLOW_FLATTEN_PARAMS` environment variable to flatten the parameters and"
                    " avoid this message."
                )
                del combined_dict[name]
        # MLflow cannot log more than 100 values in one go, so we have to split it
        combined_dict_items = list(combined_dict.items())
        for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
            self._ml_flow.log_params(dict(combined_dict_items[i: i + self._MAX_PARAMS_TAGS_PER_BATCH]))
        mlflow_tags = os.getenv("MLFLOW_TAGS", None)
        if mlflow_tags:
            mlflow_tags = json.loads(mlflow_tags)
            self._ml_flow.set_tags(mlflow_tags)

    def on_train_begin(self,
                       args: TrainingArguments,
                       state: transformers.TrainerState,
                       control: transformers.TrainerControl,
                       **kwargs):
        super().on_train_begin(args=args, state=state, control=control, **kwargs)
        run = mf.start_run(nested=True, description='hyperparemeters tuning trial')
        info_active_run()
        assert self._nested_run_id is None
        self._nested_run_id = run.info.run_id
        self._log_params_with_mlflow(model=kwargs['model'], args=args, state=state)  # TODO check this

    def on_train_end(self,
                     args: TrainingArguments,
                     state: transformers.TrainerState,
                     control: transformers.TrainerControl,
                     **kwargs):
        super().on_train_end(args=args, state=state, control=control, **kwargs)
        run = mf.active_run()
        assert run.info.run_id == self._nested_run_id
        mf.end_run()
        self._nested_run_id = None

def compute_metrics(pred: transformers.trainer_utils.EvalPrediction) -> dict[str, np.float64]:
    ground_truth = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(ground_truth, preds, average="weighted")
    acc = accuracy_score(ground_truth, preds)
    return {"accuracy": acc, "f1": f1}