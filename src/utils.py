import json
import logging
import os
import random

import mlflow as mf
import numpy as np
import transformers
from matplotlib import pyplot as plt
from mlflow.utils.name_utils import _generate_random_name
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from transformers import TrainingArguments
from transformers.integrations import MLflowCallback
from transformers.utils import flatten_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
_logger = logging.getLogger()
info = _logger.info
warning = _logger.warning


def xor(a: bool, b: bool) -> bool:
    return (a and not b) or (b and not a)


def implies(a: bool, b: bool) -> bool:
    return not a or b


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
        run = mf.start_run(run_name=get_name_for_run(), nested=True, description='hyperparemeters tuning trial')
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


def get_name_for_run() -> str:
    """
    Returns a random string suitable as the name for an MLFlow run. The returned string remains random even if the
    RNG seed has been set with random.seed(). The state of the RNG in module `random` when the function returns is the
    same as when the function was called. This allows to set a seed for the RNG (for other purposes) while still getting
    random (unpredictable) strings to be used as probably unique run names.
    :return:
    """
    curr_state = random.getstate()
    random.seed()
    name_for_run = _generate_random_name()
    random.setstate(curr_state)
    return name_for_run


def plot_confusion_matrix(y_preds, y_true, labels, show=True):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    if show:
        plt.show()
    return fig
