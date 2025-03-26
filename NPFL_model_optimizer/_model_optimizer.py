#!/usr/bin/env python3
# 3b65dab7-9b72-43fc-90c5-9bbbfb304ea9
# 3037cdec-8857-4d1e-8707-4b3916e39158

import os
from optparse import Option

import yaml
from pathlib import Path
import importlib
import re
from collections import defaultdict

from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset

import npfl138
import torch
import torch.nn as nn

import optuna

from typing import Final, TypeAlias, Any, Type, Optional, TypeVar, Literal, Callable

from yaml import YAMLError, ScalarNode

Categorical: TypeAlias = Any
HyperparamMapping: TypeAlias = dict[str, dict[str, Any]]
Self: TypeVar = TypeVar("Self", bound="ModelOptimizer")

if os.name == 'posix':
    hypertune_path = '~/.hypertune'
elif os.name == 'nt':
    hypertune_path = os.path.join(os.getenv('APPDATA'), 'hypertune')
else:
    raise OSError("Unknown OS (Mac :P)")

CHAMPION_MODEL_DIR: Final[Path] = Path(hypertune_path) / "models"

# YAML optuna config parsion utils ---------------------------------------------------------------------
def safe_eval(expression, **kwargs):
    """Evaluates a safe expression while blocking multi-line input and dangerous modules."""
    # Reject multi-line input
    if "\n" in expression or ";" in expression:
        raise ValueError("Multi-line expressions are not allowed")

    # Reject dangerous words like 'os', '__import__', etc.
    if re.search(r"\b(os|sys|subprocess|eval|exec|import|importlib)\b", expression):
        raise ValueError("Use of restricted modules/functions is not allowed")
    local_context = {**kwargs}
    return eval(expression, globals(), local_context)

def find_parent_key(buffer, start_index, line_no):
    """Find the closest parent key before 'start_index' based on indentation."""
    before_text = buffer[:start_index]  # Get everything before '!class'
    lines = before_text.splitlines()  # Split into lines

    indent = len(lines[line_no]) - len(lines[line_no].lstrip())   # Indentation of '!class'

    # Iterate backward
    for i in range(line_no-1, -1, -1):
        line = lines[i]

        match = re.match(r"^\s*(\w+):", line)
        if match:
            key_indent = len(line) - len(line.lstrip())  # Get key indentation
            if key_indent < indent:
                return match.group(1)  # Return the key name

    raise YAMLError("Unparsable yaml config")

def find_key(buffer, line_no):
    line = buffer.splitlines()[line_no]
    match = re.match(r"^\s*(\w+):", line)
    return match.group(1)

def suggest_float_constructor(loader, node, trial):
    value = list(map(lambda x: float(loader.construct_scalar(x)), node.value))
    name = find_key(node.start_mark.buffer, node.start_mark.line)
    return trial.suggest_float(name, *value)

def suggest_int_constructor(loader, node, trial):
    value = list(map(lambda x: int(loader.construct_scalar(x)), node.value))
    name = find_key(node.start_mark.buffer, node.start_mark.line)
    return trial.suggest_int(name, *value)

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def try_mapping_list_to_numeric(lst):
    new_lst = []
    for elem in lst:
        if isinstance(elem, list):
            new_lst.append(try_mapping_list_to_numeric(elem))
        elif is_int(elem):
            new_lst.append(int(elem))
        elif is_float(elem):
            new_lst.append(float(elem))
        else:
            new_lst.append(elem)
    return new_lst

def suggest_categorical_constructor(loader, node, trial):
    value = list(map(loader.construct_scalar, node.value)) if isinstance(node.value[0], ScalarNode) else list(map(loader.construct_sequence, node.value))
    value = try_mapping_list_to_numeric(value)
    name = find_key(node.start_mark.buffer, node.start_mark.line)
    return trial.suggest_categorical(name, value)

def eval_constructor(loader, node, trial, model: "ModelOptimizer"):
    value = loader.construct_scalar(node)
    return safe_eval(value, trial=trial, model=model)

def registered_constructor(loader, node, registered: dict):
    value = loader.construct_scalar(node)
    return registered[value]

def class_constructor(loader, node, trial, model) -> Any:
    class_ = loader.construct_scalar(node)

    if not class_:
        # None for optional classes like scheduler
        return None

    name = find_parent_key(node.start_mark.buffer, node.start_mark.index, node.start_mark.line)
    if isinstance(class_, list):
        cls = trial.suggest_categorical(name, class_)

    if isinstance(class_, Type):
        cls = class_
    elif isinstance(class_, str):
        if class_ == "{{resolve}}":
            cls = model.__getattr__(name).__class__
        else:
            rsp = class_.rsplit(".", 1)
            module_name, class_name = rsp
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)

    return cls

def parse_class(cls_dict: dict, constructed_kwargs: dict, model: "ModelOptimizer"):
    class_kwargs = {}
    class_ = cls_dict.pop("class")
    if not class_:
        # optional classes like lr scheduler
        return None
    for k, v in cls_dict.items():
        if isinstance(v, str) and "^hook:" in v:
            v = constructed_kwargs[v[6:]]
        if isinstance(v, str) and "!eval:" in v:
            v = safe_eval(v[6:], model=model)
        class_kwargs[k] = v
    return class_(**class_kwargs)

def parse_config(conf: dict, model: "ModelOptimizer"):
    constructed_kwargs = {}
    for k, v in conf.items():
        if isinstance(v, dict) and "class" in v:
            v = parse_class(v, constructed_kwargs, model)
        constructed_kwargs[k] = v
    return constructed_kwargs
# YAML optuna config parsion utils end -------------------------------------------------------------------


# predefined callbacks -----------------------------------------------------------------------------------
def lt(curr_x, best_x):
    return curr_x < best_x

def gt(curr_x, best_x):
    return curr_x > best_x

def callback_early_stop(model, epochs, logs):
    """
    Early stop according to optimized metric
    """
    metric_name = model.metric
    metric_type = metric_name.split("_")[-1]
    best_metric = model.__getattribute__("best_" + metric_type)
    curr_metric = logs[metric_name].cpu().item()
    comparison = lt if model.direction == "minimize" else gt

    stop = False
    if comparison(curr_metric, best_metric):
        model.__setattr__("best_" + metric_type, curr_metric)
        model.last_best_epoch = epochs
        for m in model.metrics.keys():
            model.__setattr__("best_" + m, logs["dev_"+m].cpu().item())
    if model.last_best_epoch + model.patience <= epochs:
        stop = model.STOP_TRAINING
    return stop

def callback_save_champion(model, epochs, logs):
    """
    Save champion if current model is the best, always return True
    """
    model.save_champion()
    return False

def callback_clip_grad_norm(model, epochs, logs):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    return False
# callbacks end -------------------------------------------------------------------------------------------


class ModelOptimizer(npfl138.TrainableModule):
    def __init__(self,
                 module: torch.nn.Module,
                 model_name: str,
                 early_stop=True,
                 patience: int=5,
                 metric: str="dev_loss",
                 direction: str="minimize",
                 batch_size: int = 64,
                 model_dir: Path=CHAMPION_MODEL_DIR,
                 ):
        super().__init__(module)
        self._path_model_weights = Path(model_dir) / f"{model_name}_champion.pth"
        self._path_model_config = Path(model_dir) / f"{model_name}_champion.cfg"
        self._path_model_optim = Path(model_dir) / f"{model_name}_optim.cfg"
        self._path_best_loss = Path(model_dir) / f"{model_name}_best_loss.txt"
        self.best_loss = float('inf')

        # register superclass methods
        self._fit = super().fit
        self._configure = super().configure
        self._predict = super().predict

        # early stopping configuration
        self.early_stop = early_stop
        self.patience = patience
        self.last_best_epoch = 0
        self._callbacks = []

        self.batch_size = batch_size

        # metric for optimization (specified in self.optimize) and early stopping
        self.metric = metric
        self.direction = direction

        # register dict allows users to register
        # code generated params to the optimizer pipeline
        self.register_dict = {"params": self.parameters(recurse=True)}


    def load_champion(self):
        assert self._path_model_weights.exists() \
               and self._path_model_config.exists() \
               and self._path_model_optim.exists(), \
            "No model to load."
        self.load_config(str(self._path_model_config))
        self.load_weights(str(self._path_model_weights))
        return self

    def configure(self, **kwargs):
        """ kwargs are same as for npfl138.TrainableModule.configure() """
        self._configure(**kwargs)
        for metric in self.metrics.keys():
            self.__setattr__("best_" + metric, 0.)
        self._callbacks = []
        if self.early_stop:
            self._callbacks.append(callback_early_stop)
        self._callbacks.append(callback_save_champion)
        self._callbacks.append(callback_clip_grad_norm)

    def get_best_metrics(self):
        return {"loss": self.best_loss} | {m: self.__getattribute__("best_" + m) for m in self.metrics.keys()}

    def save_champion(self):
        """
        Save current model if it is the best model yet
        """
        if self._path_best_loss.exists():
            with open(self._path_best_loss, "r") as f:
                saved_best_loss = float(f.readline())
            if saved_best_loss <= self.best_loss:
                # Only save if it is the best model yet
                return

        print(f"Saving best model yet (epoch: {self.epoch}) with metrics: {self.get_best_metrics()} to {self._path_model_weights.parent}")
        self.save_config(str(self._path_model_config))
        self.save_weights(str(self._path_model_weights), str(self._path_model_optim))
        os.makedirs(self._path_best_loss.parent, exist_ok=True)
        with open(self._path_best_loss, "w") as f:
            f.truncate(0)
            f.write(str(self.best_loss))

    def fit(self, train: Dataset | DataLoader, *, load_train_progress=False, load_best_in_the_end=True, **kwargs):
        """
        Same args and kwargs as npfl138.TrainableModule.fit()
        """
        if load_train_progress and os.path.exists(self._path_model_weights):
            self.load_champion()

        assert self.loss_tracker is not None, "The TrainableModule has not been configured, or loaded"


        if isinstance(train, Dataset):
            train = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        dev = kwargs.pop("dev", None)
        if dev and isinstance(dev, Dataset):
            dev = DataLoader(dev, batch_size=self.batch_size, shuffle=False)
        kwargs["dev"] = dev

        self._callbacks.extend(kwargs.pop("callbacks", []))
        retval = self._fit(train, callbacks=self._callbacks, **kwargs)
        if load_best_in_the_end:
            self.load_champion()
        return retval

    def predict(self, test: Dataset | DataLoader, *, load_champion: bool=False, **kwargs):
        """
        If no model is loaded or load_champion=True,
        Trainer will loaded champion of that name and predict the test data
        otherwise current wrapped model will be evaluated

        Same keywords as npfl138.TrainableModule.predict()

        :param test: test dataset / dataloader
        :param load_champion: (bool) load champion of self.name

        :return: same as self.predict()
        """
        if self.loss_tracker is None or load_champion:
            self.load_champion()
        if isinstance(test, Dataset):
            test = DataLoader(test, batch_size=self.batch_size, shuffle=False)
        return self._predict(test, **kwargs)

    def register(self, name: str, param: Any):
        """
        register parameter of type 'registered' to the optimization pipeline
        Note: The name of the param specified has to be the same as in the config file
        Model params are registered by default under name 'params'
        """
        self.register_dict |= {name: param}

    def _optuna_trial(self,
                      trial: optuna.Trial,
                      metric: str,
                      optuna_config_stream: str,
                      train: DataLoader,
                      dev: DataLoader,
                      metrics: Optional[dict],
                      pre_trial_hooks: Optional[list[Callable[[optuna.Trial, "ModelOptimizer"], None], ...]],
                      post_trial_hooks: Optional[list[Callable[[optuna.Trial, "ModelOptimizer"], None], ...]]
                      ):
        yaml.SafeLoader.add_constructor("!float", lambda l, n: suggest_float_constructor(l, n, trial))
        yaml.SafeLoader.add_constructor("!int", lambda l, n: suggest_int_constructor(l, n, trial))
        yaml.SafeLoader.add_constructor("!categorical", lambda l, n: suggest_categorical_constructor(l, n, trial))
        yaml.SafeLoader.add_constructor("!registered", lambda l, n: registered_constructor(l, n, self.register_dict))
        yaml.SafeLoader.add_constructor("!eval", lambda l, n: eval_constructor(l, n, trial, self))
        yaml.SafeLoader.add_constructor("!class", lambda l, n: class_constructor(l, n, trial, self))

        config = yaml.safe_load(optuna_config_stream)
        self_params = parse_config(config["self"], self)
        self.__dict__.update(**self_params)

        fit_params = parse_config(config["functional"]["fit"], self)
        configure_params = parse_config(config["functional"]["configure"], self)
        self.unconfigure()
        self.configure(metrics=metrics, **configure_params)
        self.module.to(self.device)

        for hook in pre_trial_hooks:
            hook(trial, self, train, dev)

        print(f"Trial {trial.number} params: {trial.params}")
        logs = self.fit(train, dev=dev, **fit_params)

        for hook in post_trial_hooks:
            hook(trial, self)

        return logs[metric]

    def optimize(self,
                 optuna_config_path: os.PathLike,
                 optimized_metric: str,
                 direction: Literal["minimize", "maximize"],
                 n_trials: int,
                 train: Dataset | DataLoader,
                 dev: Dataset | DataLoader,
                 metrics: Optional[dict] = None,
                 pre_trial_hooks: Optional[list[Callable[[optuna.Trial, "ModelOptimizer"], None], ...]] = None,
                 post_trial_hooks: Optional[list[Callable[[optuna.Trial, "ModelOptimizer"], None], ...]] = None,):
        """
        Optimize wrapped model

        call order:
        self.configure()
        self.self_train()

        :param optuna_config_path: (str) path to yaml config
        :param optimized_metric: (str) name of metric to optimize
        :param direction: (str) either minimize or maximize
        :param n_trials: (int) number of trials
        :param train: (Dataset) train dataset
        :param dev: (Dataset) dev dataset
        :param metrics: (dict) metrics to log, has to contain optimized metric
        :param pre_trial_hooks: (list) callable with trial and self params that are called in the beginning of each trial
        :param post_trial_hooks: (list) callable with trial and self params that are called in the end of each trial

        :return: (TrainerModel) optimized model

        Args:
            pre_trial_hooks:
        """
        print("Welcome to Model Optimizer 1.0")

        if pre_trial_hooks is None:
            pre_trial_hooks = []
        if post_trial_hooks is None:
            post_trial_hooks = []

        self.metric = optimized_metric
        self.direction = direction

        with open(optuna_config_path, "r") as f:
            optuna_config_stream = f.read()

        study = optuna.create_study(direction=self.direction)
        study.optimize(
            lambda trial: self._optuna_trial(trial, self.metric, optuna_config_stream, train, dev, metrics, pre_trial_hooks, post_trial_hooks),
            n_trials=n_trials,
        )

        self.load_champion()
        return self
