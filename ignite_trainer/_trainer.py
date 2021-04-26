import io
import os
import glob
import json
import time
import tqdm
import signal
import argparse
import numpy as np

import torch
import torch.utils.data
import torch.nn.functional

import torchvision as tv

import ignite.engine as ieng
import ignite.metrics as imet
import ignite.handlers as ihan

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Optional

from termcolor import colored

from collections import defaultdict
from collections.abc import Iterable

from ignite_trainer import _utils
from ignite_trainer import _visdom
from ignite_trainer import _interfaces

VISDOM_HOST = 'localhost'
VISDOM_PORT = 8097
VISDOM_ENV_PATH = os.path.join(os.path.expanduser('~'), 'visdom_env')
BATCH_TRAIN = 128
BATCH_TEST = 1024
WORKERS_TRAIN = 0
WORKERS_TEST = 0
EPOCHS = 100
LOG_INTERVAL = 50
SAVED_MODELS_PATH = os.path.join(os.path.expanduser('~'), 'saved_models')


def run(experiment_name: str,
        visdom_host: str,
        visdom_port: int,
        visdom_env_path: str,
        model_class: str,
        model_args: Dict[str, Any],
        optimizer_class: str,
        optimizer_args: Dict[str, Any],
        dataset_class: str,
        dataset_args: Dict[str, Any],
        batch_train: int,
        batch_test: int,
        workers_train: int,
        workers_test: int,
        transforms: List[Dict[str, Union[str, Dict[str, Any]]]],
        epochs: int,
        log_interval: int,
        saved_models_path: str,
        performance_metrics: Optional = None,
        scheduler_class: Optional[str] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        model_suffix: Optional[str] = None,
        setup_suffix: Optional[str] = None,
        orig_stdout: Optional[io.TextIOBase] = None,
        skip_train_val: bool = False):

    with _utils.tqdm_stdout(orig_stdout) as orig_stdout:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transforms_train = list()
        transforms_test = list()

        for idx, transform in enumerate(transforms):
            use_train = transform.get('train', True)
            use_test = transform.get('test', True)

            transform = _utils.load_class(transform['class'])(**transform['args'])

            if use_train:
                transforms_train.append(transform)
            if use_test:
                transforms_test.append(transform)

            transforms[idx]['train'] = use_train
            transforms[idx]['test'] = use_test

        transforms_train = tv.transforms.Compose(transforms_train)
        transforms_test = tv.transforms.Compose(transforms_test)

        Dataset: Type = _utils.load_class(dataset_class)

        train_loader, eval_loader = _utils.get_data_loaders(
            Dataset,
            dataset_args,
            batch_train,
            batch_test,
            workers_train,
            workers_test,
            transforms_train,
            transforms_test
        )

        Network: Type = _utils.load_class(model_class)
        model: _interfaces.AbstractNet = Network(**model_args)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = model.to(device)

        Optimizer: Type = _utils.load_class(optimizer_class)
        optimizer: torch.optim.Optimizer = Optimizer(model.parameters(), **optimizer_args)

        if scheduler_class is not None:
            Scheduler: Type = _utils.load_class(scheduler_class)

            if scheduler_args is None:
                scheduler_args = dict()

            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = Scheduler(optimizer, **scheduler_args)
        else:
            scheduler = None

        model_short_name = ''.join([c for c in Network.__name__ if c == c.upper()])
        model_name = '{}{}'.format(
            model_short_name,
            '-{}'.format(model_suffix) if model_suffix is not None else ''
        )
        visdom_env_name = '{}_{}_{}{}'.format(
            Dataset.__name__,
            experiment_name,
            model_name,
            '-{}'.format(setup_suffix) if setup_suffix is not None else ''
        )

        vis, vis_pid = _visdom.get_visdom_instance(visdom_host, visdom_port, visdom_env_name, visdom_env_path)

        prog_bar_epochs = tqdm.tqdm(total=epochs, desc='Epochs', file=orig_stdout, dynamic_ncols=True, unit='epoch')
        prog_bar_iters = tqdm.tqdm(desc='Batches', file=orig_stdout, dynamic_ncols=True)

        tqdm.tqdm.write(f'\n{repr(model)}\n')
        tqdm.tqdm.write('Total number of parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

        def training_step(_: ieng.Engine, batch: _interfaces.TensorPair) -> torch.Tensor:
            model.train()

            optimizer.zero_grad()

            x, y = batch

            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)
            if loss.ndim > 0:
                loss = loss.mean()

            loss.backward(retain_graph=False)
            optimizer.step(None)

            return loss.item()

        def eval_step(_: ieng.Engine, batch: _interfaces.TensorPair) -> _interfaces.TensorPair:
            model.eval()

            with torch.no_grad():
                x, y = batch

                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)

            return y_pred, y

        trainer = ieng.Engine(training_step)
        validator_train = ieng.Engine(eval_step)
        validator_eval = ieng.Engine(eval_step)

        # placeholder for summary window
        vis.text(
            text='',
            win=experiment_name,
            env=visdom_env_name,
            opts={'title': 'Summary', 'width': 940, 'height': 416},
            append=vis.win_exists(experiment_name, visdom_env_name)
        )

        default_metrics = {
            "Loss": {
                "window_name": None,
                "x_label": "#Epochs",
                "y_label": model.loss_fn_name if not isinstance(model, torch.nn.DataParallel) else model.module.loss_fn_name,
                "width": 940,
                "height": 416,
                "lines": [
                    {
                        "line_label": "SMA",
                        "object": imet.RunningAverage(output_transform=lambda x: x),
                        "test": False,
                        "update_rate": "iteration"
                    },
                    {
                        "line_label": "Val.",
                        "train": not skip_train_val,
                        "object": imet.Loss(model.loss_fn if not isinstance(model, torch.nn.DataParallel) else model.module.loss_fn)
                    }
                ]
            }
        }

        performance_metrics = {**default_metrics, **performance_metrics}
        checkpoint_metrics = list()

        for scope_name, scope in performance_metrics.items():
            scope['window_name'] = scope.get('window_name', scope_name) or scope_name

            for line in scope['lines']:
                if 'object' not in line:
                    line['object']: imet.Metric = _utils.load_class(line['class'])(**line['args'])

                line['metric_label'] = '{}: {}'.format(scope['window_name'], line['line_label'])

                line['update_rate'] = line.get('update_rate', 'epoch')
                line_suffixes = list()
                if line['update_rate'] == 'iteration':
                    line['object'].attach(trainer, line['metric_label'])
                    line['train'] = False
                    line['test'] = False

                    line_suffixes.append(' Train.')

                if line.get('train', True):
                    line['object'].attach(validator_train, line['metric_label'])
                    line_suffixes.append(' Train.')
                if line.get('test', True):
                    line['object'].attach(validator_eval, line['metric_label'])
                    line_suffixes.append(' Eval.')

                    if line.get('is_checkpoint', False):
                        checkpoint_metrics.append(line['metric_label'])

                for line_suffix in line_suffixes:
                    _visdom.plot_line(
                        vis=vis,
                        window_name=scope['window_name'],
                        env=visdom_env_name,
                        line_label=line['line_label'] + line_suffix,
                        x_label=scope['x_label'],
                        y_label=scope['y_label'],
                        width=scope['width'],
                        height=scope['height'],
                        draw_marker=(line['update_rate'] == 'epoch')
                    )

        if checkpoint_metrics:
            score_name = 'performance'

            def get_score(engine: ieng.Engine) -> float:
                current_mode = getattr(engine.state.dataloader.iterable.dataset, dataset_args['training']['key'])
                val_mode = dataset_args['training']['no']

                score = 0.0
                if current_mode == val_mode:
                    for metric_name in checkpoint_metrics:
                        try:
                            score += engine.state.metrics[metric_name]
                        except KeyError:
                            pass

                return score

            model_saver = ihan.ModelCheckpoint(
                os.path.join(saved_models_path, visdom_env_name),
                filename_prefix=visdom_env_name,
                score_name=score_name,
                score_function=get_score,
                n_saved=3,
                save_as_state_dict=True,
                require_empty=False,
                create_dir=True
            )

            validator_eval.add_event_handler(ieng.Events.EPOCH_COMPLETED, model_saver, {model_name: model})

        @trainer.on(ieng.Events.EPOCH_STARTED)
        def reset_progress_iterations(engine: ieng.Engine):
            prog_bar_iters.clear()
            prog_bar_iters.n = 0
            prog_bar_iters.last_print_n = 0
            prog_bar_iters.start_t = time.time()
            prog_bar_iters.last_print_t = time.time()
            prog_bar_iters.total = len(engine.state.dataloader)

        @trainer.on(ieng.Events.ITERATION_COMPLETED)
        def log_training(engine: ieng.Engine):
            prog_bar_iters.update(1)

            num_iter = (engine.state.iteration - 1) % len(train_loader) + 1

            early_stop = np.isnan(engine.state.output) or np.isinf(engine.state.output)

            if num_iter % log_interval == 0 or num_iter == len(train_loader) or early_stop:
                tqdm.tqdm.write(
                    'Epoch[{}] Iteration[{}/{}] Loss: {:.4f}'.format(
                        engine.state.epoch, num_iter, len(train_loader), engine.state.output
                    )
                )

                x_pos = engine.state.epoch + num_iter / len(train_loader) - 1
                for scope_name, scope in performance_metrics.items():
                    for line in scope['lines']:
                        if line['update_rate'] == 'iteration':
                            line_label = '{} Train.'.format(line['line_label'])
                            line_value = engine.state.metrics[line['metric_label']]

                            if engine.state.epoch > 1:
                                _visdom.plot_line(
                                    vis=vis,
                                    window_name=scope['window_name'],
                                    env=visdom_env_name,
                                    line_label=line_label,
                                    x_label=scope['x_label'],
                                    y_label=scope['y_label'],
                                    x=np.full(1, x_pos),
                                    y=np.full(1, line_value)
                                )

            if early_stop:
                tqdm.tqdm.write(colored('Early stopping due to invalid loss value.', 'red'))
                trainer.terminate()

        def log_validation(engine: ieng.Engine,
                           train: bool = True):

            if train:
                run_type = 'Train.'
                data_loader = train_loader
                validator = validator_train
            else:
                run_type = 'Eval.'
                data_loader = eval_loader
                validator = validator_eval

            prog_bar_validation = tqdm.tqdm(
                data_loader,
                desc=f'Validation {run_type}',
                file=orig_stdout,
                dynamic_ncols=True,
                leave=False
            )
            validator.run(prog_bar_validation)
            prog_bar_validation.clear()
            prog_bar_validation.close()

            tqdm_info = [
                'Epoch: {}'.format(engine.state.epoch)
            ]
            for scope_name, scope in performance_metrics.items():
                for line in scope['lines']:
                    if line['update_rate'] == 'epoch':
                        try:
                            line_label = '{} {}'.format(line['line_label'], run_type)
                            line_value = validator.state.metrics[line['metric_label']]

                            _visdom.plot_line(
                                vis=vis,
                                window_name=scope['window_name'],
                                env=visdom_env_name,
                                line_label=line_label,
                                x_label=scope['x_label'],
                                y_label=scope['y_label'],
                                x=np.full(1, engine.state.epoch),
                                y=np.full(1, line_value),
                                draw_marker=True
                            )

                            tqdm_info.append('{}: {:.4f}'.format(line_label, line_value))
                        except KeyError:
                            pass

            tqdm.tqdm.write('{} results - {}'.format(run_type, '; '.join(tqdm_info)))

        if not skip_train_val:
            @trainer.on(ieng.Events.EPOCH_COMPLETED)
            def log_validation_train(engine: ieng.Engine):
                log_validation(engine, True)

        @trainer.on(ieng.Events.EPOCH_COMPLETED)
        def log_validation_eval(engine: ieng.Engine):
            log_validation(engine, False)

            if engine.state.epoch == 1:
                summary = _utils.build_summary_str(
                    experiment_name=experiment_name,
                    model_short_name=model_name,
                    model_class=model_class,
                    model_args=model_args,
                    optimizer_class=optimizer_class,
                    optimizer_args=optimizer_args,
                    dataset_class=dataset_class,
                    dataset_args=dataset_args,
                    transforms=transforms,
                    epochs=epochs,
                    batch_train=batch_train,
                    log_interval=log_interval,
                    saved_models_path=saved_models_path,
                    scheduler_class=scheduler_class,
                    scheduler_args=scheduler_args
                )
                _visdom.create_summary_window(
                    vis=vis,
                    visdom_env_name=visdom_env_name,
                    experiment_name=experiment_name,
                    summary=summary
                )

            vis.save([visdom_env_name])

            prog_bar_epochs.update(1)

            if scheduler is not None:
                scheduler.step(engine.state.epoch)

        trainer.run(train_loader, max_epochs=epochs)

        if vis_pid is not None:
            tqdm.tqdm.write('Stopping visdom')
            os.kill(vis_pid, signal.SIGTERM)

        del vis
        del train_loader
        del eval_loader

        prog_bar_iters.clear()
        prog_bar_iters.close()

        prog_bar_epochs.clear()
        prog_bar_epochs.close()

    tqdm.tqdm.write('\n')


def main():
    with _utils.tqdm_stdout() as orig_stdout:
        parser = argparse.ArgumentParser()

        parser.add_argument('-c', '--config', type=str, required=True)
        parser.add_argument('-H', '--visdom-host', type=str, required=False)
        parser.add_argument('-P', '--visdom-port', type=int, required=False)
        parser.add_argument('-E', '--visdom-env-path', type=str, required=False)
        parser.add_argument('-b', '--batch-train', type=int, required=False)
        parser.add_argument('-B', '--batch-test', type=int, required=False)
        parser.add_argument('-w', '--workers-train', type=int, required=False)
        parser.add_argument('-W', '--workers-test', type=int, required=False)
        parser.add_argument('-e', '--epochs', type=int, required=False)
        parser.add_argument('-L', '--log-interval', type=int, required=False)
        parser.add_argument('-M', '--saved-models-path', type=str, required=False)
        parser.add_argument('-R', '--random-seed', type=int, required=False)
        parser.add_argument('-s', '--suffix', type=str, required=False)
        parser.add_argument('-S', '--skip-train-val', action='store_true', default=False)

        args, unknown_args = parser.parse_known_args()

        if args.batch_test is None:
            args.batch_test = args.batch_train

        if args.random_seed is not None:
            args.suffix = '{}r-{}'.format(
                '{}_'.format(args.suffix) if args.suffix is not None else '',
                args.random_seed
            )

            np.random.seed(args.random_seed)
            torch.random.manual_seed(args.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.random_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        configs_found = list(sorted(glob.glob(os.path.expanduser(args.config))))
        prog_bar_exps = tqdm.tqdm(
            configs_found,
            desc='Experiments',
            unit='setup',
            file=orig_stdout,
            dynamic_ncols=True
        )

        for config_path in prog_bar_exps:
            config = json.load(open(config_path))

            if unknown_args:
                tqdm.tqdm.write('\nParsing additional arguments...')

            args_not_found = list()
            for arg in unknown_args:
                if arg.startswith('--'):
                    keys = arg.strip('-').split('.')

                    section = config
                    found = True
                    for key in keys:
                        if key in section:
                            section = section[key]
                        else:
                            found = False
                            break

                    if found:
                        override_parser = argparse.ArgumentParser()

                        section_nargs = None
                        section_type = type(section) if section is not None else str

                        if section_type is bool:
                            if section_type is bool:
                                def infer_bool(x: str) -> bool:
                                    return x.lower() not in ('0', 'false', 'no')

                                section_type = infer_bool

                        if isinstance(section, Iterable) and section_type is not str:
                            section_nargs = '+'
                            section_type = {type(value) for value in section}

                            if len(section_type) == 1:
                                section_type = section_type.pop()
                            else:
                                section_type = str

                        override_parser.add_argument(arg, nargs=section_nargs, type=section_type)
                        overridden_args, _ = override_parser.parse_known_args(unknown_args)
                        overridden_args = vars(overridden_args)

                        overridden_key = arg.strip('-')
                        overriding_value = overridden_args[overridden_key]

                        section = config
                        old_value = None
                        for i, key in enumerate(keys, 1):
                            if i == len(keys):
                                old_value = section[key]
                                section[key] = overriding_value
                            else:
                                section = section[key]

                        tqdm.tqdm.write(
                            colored(f'Overriding "{overridden_key}": {old_value} -> {overriding_value}', 'magenta')
                        )
                    else:
                        args_not_found.append(arg)

            if args_not_found:
                tqdm.tqdm.write(
                    colored(
                        '\nThere are unrecognized arguments to override: {}'.format(
                            ', '.join(args_not_found)
                        ),
                        'red'
                    )
                )

            config = defaultdict(None, config)

            experiment_name = config['Setup']['name']

            visdom_host = _utils.arg_selector(
                args.visdom_host, config['Visdom']['host'], VISDOM_HOST
            )
            visdom_port = int(_utils.arg_selector(
                args.visdom_port, config['Visdom']['port'], VISDOM_PORT
            ))
            visdom_env_path = _utils.arg_selector(
                args.visdom_env_path, config['Visdom']['env_path'], VISDOM_ENV_PATH
            )
            batch_train = int(_utils.arg_selector(
                args.batch_train, config['Setup']['batch_train'], BATCH_TRAIN
            ))
            batch_test = int(_utils.arg_selector(
                args.batch_test, config['Setup']['batch_test'], BATCH_TEST
            ))
            workers_train = _utils.arg_selector(
                args.workers_train, config['Setup']['workers_train'], WORKERS_TRAIN
            )
            workers_test = _utils.arg_selector(
                args.workers_test, config['Setup']['workers_test'], WORKERS_TEST
            )
            epochs = _utils.arg_selector(
                args.epochs, config['Setup']['epochs'], EPOCHS
            )
            log_interval = _utils.arg_selector(
                args.log_interval, config['Setup']['log_interval'], LOG_INTERVAL
            )
            saved_models_path = _utils.arg_selector(
                args.saved_models_path, config['Setup']['saved_models_path'], SAVED_MODELS_PATH
            )

            model_class = config['Model']['class']
            model_args = config['Model']['args']

            optimizer_class = config['Optimizer']['class']
            optimizer_args = config['Optimizer']['args']

            if 'Scheduler' in config:
                scheduler_class = config['Scheduler']['class']
                scheduler_args = config['Scheduler']['args']
            else:
                scheduler_class = None
                scheduler_args = None

            dataset_class = config['Dataset']['class']
            dataset_args = config['Dataset']['args']

            transforms = config['Transforms']
            performance_metrics = config['Metrics']

            tqdm.tqdm.write(f'\nStarting experiment "{experiment_name}"\n')

            run(
                experiment_name=experiment_name,
                visdom_host=visdom_host,
                visdom_port=visdom_port,
                visdom_env_path=visdom_env_path,
                model_class=model_class,
                model_args=model_args,
                optimizer_class=optimizer_class,
                optimizer_args=optimizer_args,
                dataset_class=dataset_class,
                dataset_args=dataset_args,
                batch_train=batch_train,
                batch_test=batch_test,
                workers_train=workers_train,
                workers_test=workers_test,
                transforms=transforms,
                epochs=epochs,
                log_interval=log_interval,
                saved_models_path=saved_models_path,
                performance_metrics=performance_metrics,
                scheduler_class=scheduler_class,
                scheduler_args=scheduler_args,
                model_suffix=config['Setup']['suffix'],
                setup_suffix=args.suffix,
                orig_stdout=orig_stdout,
                skip_train_val=args.skip_train_val
            )

        prog_bar_exps.close()

    tqdm.tqdm.write('\n')
