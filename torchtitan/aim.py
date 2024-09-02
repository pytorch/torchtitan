import os
from typing import Any, Dict, Optional

from aim.ext.resource.configs import DEFAULT_SYSTEM_TRACKING_INT
from aim.sdk.repo import Repo
from aim.sdk.run import Run
from aim.sdk.utils import clean_repo_path, get_aim_repo_name


class AimLogger():
    def __init__(
        self,
        repo: Optional[str] = None,
        experiment: Optional[str] = None,
        system_tracking_interval: Optional[int] = DEFAULT_SYSTEM_TRACKING_INT,
        log_system_params: Optional[bool] = True,
        capture_terminal_logs: Optional[bool] = True,
        run_name: Optional[str] = None,
        run_hash: Optional[str] = None,
        train_metric_prefix: Optional[str] = 'train_',
        val_metric_prefix: Optional[str] = 'val_',
        test_metric_prefix: Optional[str] = 'test_',
    ):
        super().__init__()

        self._experiment_name = experiment
        self._run_name = run_name
        self._repo_path = repo

        self._system_tracking_interval = system_tracking_interval
        self._log_system_params = log_system_params
        self._capture_terminal_logs = capture_terminal_logs

        self._run = None
        self._run_hash = run_hash

        self._train_metric_prefix = train_metric_prefix
        self._val_metric_prefix = val_metric_prefix
        self._test_metric_prefix = test_metric_prefix

    @property
    def experiment(self) -> Run:
        if self._run is None:
            if self._run_hash:
                self._run = Run(
                    self._run_hash,
                    repo=self._repo_path,
                    system_tracking_interval=self._system_tracking_interval,
                    capture_terminal_logs=self._capture_terminal_logs,
                    force_resume=True,
                )
            else:
                self._run = Run(
                    repo=self._repo_path,
                    experiment=self._experiment_name,
                    system_tracking_interval=self._system_tracking_interval,
                    log_system_params=self._log_system_params,
                    capture_terminal_logs=self._capture_terminal_logs,
                )
                self._run_hash = self._run.hash
            if self._run_name is not None:
                self._run.name = self._run_name
        return self._run

    def log_hyperparams(self, params: Dict[str, Any]):
        for key, value in params.items():
            self.experiment.set(('hparams', key), value, strict=False)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):

        metric_items: Dict[str:Any] = {k: v for k, v in metrics.items()} # for modifications to metric_items without affecting the original metrics
        for k, v in metric_items.items():
            name = k
            context = {}
            if self._train_metric_prefix and name.startswith(self._train_metric_prefix):
                name = name[len(self._train_metric_prefix) :]
                context['subset'] = 'train'
            elif self._test_metric_prefix and name.startswith(self._test_metric_prefix):
                name = name[len(self._test_metric_prefix) :]
                context['subset'] = 'test'
            elif self._val_metric_prefix and name.startswith(self._val_metric_prefix):
                name = name[len(self._val_metric_prefix) :]
                context['subset'] = 'val'
            self.experiment.track(v, name=name, step=step, context=context)

    def finalize(self) -> None:
        if self._run:
            self._run.close()
            del self._run
            self._run = None

    def __del__(self):
        self.finalize()

    @property
    def save_dir(self) -> str:
        repo_path = clean_repo_path(self._repo_path) or Repo.default_repo_path()
        return os.path.join(repo_path, get_aim_repo_name())

    @property
    def name(self) -> str:
        return self._experiment_name

    @property
    def version(self) -> str:
        return self.experiment.hash