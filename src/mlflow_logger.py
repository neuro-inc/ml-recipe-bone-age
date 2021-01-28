import logging
from typing import List, Dict, Union, Any

from catalyst.dl.callbacks.logging import LoggerCallback, RunnerState

import mlflow
import mlflow.pytorch


class MLFlowLogging(LoggerCallback):
    def __init__(self, metric_names: List[str] = None, extra_params: Dict[str, Any] = None, *args, **kwargs):
        super(MLFlowLogging, self).__init__(*args, **kwargs)
        self.mlflow_run: mlflow.ActiveRun = mlflow.start_run()
        self.metrics_to_log = metric_names
        self.extra_params = extra_params

    def _log_metrics(self, metrics: Dict[str, float], step: int, suffix: str = ""):
        if self.metrics_to_log is None:
            metrics_to_log = sorted(list(metrics.keys()))
        else:
            metrics_to_log = self.metrics_to_log
        for name in metrics_to_log:
            if name in metrics:
                mlflow.log_metric(f"{name}{suffix}", metrics[name], step)

    def _log_params(self, params: Dict[str, Union[int, float, str]]):
        for param, value in params.items():
            mlflow.log_param(param, str(value))
        for param, value in self.extra_params.items():
            mlflow.log_param(param, str(value))

    def on_stage_end(self, state: RunnerState):
        try:
            mlflow.pytorch.log_model(state.model, "model")
            mlflow.end_run(status="FINISHED")
        except Exception as e:
            logging.exception(f"Unable to log model into MLFlow: {e}", exc_info=True)
            mlflow.end_run("FAILED")

    def on_batch_end(self, state: RunnerState):
        """Send batch metrics to MLFlow"""
        # mode = state.loader_name
        metrics_ = state.metrics.batch_values
        self._log_metrics(metrics=metrics_, step=int(state.step / state.batch_size), suffix="/batch")

    def on_epoch_end(self, state: RunnerState):
        """Send epoch metrics to MLFlow"""
        mode = state.loader_name
        metrics_ = state.metrics.epoch_values[mode]
        self._log_metrics(
            metrics=metrics_,
            step=state.epoch,
            suffix="/epoch",
        )

    def on_stage_start(self, state: RunnerState):
        self._log_params(state.model.init_params)
