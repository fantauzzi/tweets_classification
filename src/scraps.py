import json
from pathlib import Path


def load_run_checkpoint(run_id: int, output_dir: str | Path, metric: str, direction: str) -> (str, float):
    assert direction in ('minimize', 'maximize')
    run_dir = '/' / Path(output_dir) / f'run-{run_id}'
    items = run_dir.glob('checkpoint-*')
    res = None
    res_metric = None
    for item in items:
        if not item.is_dir():
            continue
        with open(item / 'trainer_state.json', 'rt') as trainer_state_file:
            trainer_state = json.load(trainer_state_file)
            metric_value = None
            for log_history_item in trainer_state['log_history']:
                metric_value = log_history_item.get(metric)
                if metric_value is not None:
                    if res_metric is None or (metric_value > res_metric and direction == 'maximize') or (
                            metric_value < res_metric and direction == 'minimize'):
                        res_metric = metric_value
                        res = str(item)
    return res, res_metric


res, metric_value = load_run_checkpoint(0,
                                        'home/fanta/workspace/tweets_classification/models/distilbert-base-uncased-finetuned-emotion',
                                        'eval_f1',
                                        'maximize')

print(res)
print(metric_value)
