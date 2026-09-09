import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner

from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger


class SubsetEvalHook(EvalHook):
    """EvalHook that tags its metrics with the name of the subset it ran on.

    The stock EvalHook writes bare 'mIoU'/'mAcc'/'aAcc' into the log buffer and
    clears that buffer before each run, so when several are registered only one
    survives into the .log.json and TensorBoard. This one prefixes its metrics
    with the subset name and re-publishes every subset seen so far in the
    current round (via the shared dict), so all of them are logged together and
    stay separable.
    """

    def __init__(self, dataloader, subset_name, shared_metrics, **kwargs):
        super(SubsetEvalHook, self).__init__(dataloader, **kwargs)
        self.subset_name = subset_name
        self.shared_metrics = shared_metrics

    # NOTE: the stock EvalHook calls runner.log_buffer.clear() here. That drops
    # 'data_time' (recorded back in before_train_iter) while IterTimerHook
    # re-adds 'time' immediately afterwards. TextLoggerHook then sees 'time',
    # decides the record is a training log, and dies with KeyError: 'data_time'
    # -- which is why runs crashed at the very first evaluation. Skipping the
    # clear keeps the buffer self-consistent and lets the training metrics and
    # every subset's metrics land in one record.
    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        runner.logger.info(
            f'evaluation on subset [{self.subset_name}] '
            f'({len(self.dataloader.dataset)} images)')
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            self.shared_metrics[f'{self.subset_name}/{name}'] = val
        runner.log_buffer.output.update(self.shared_metrics)
        runner.log_buffer.ready = True


class DistSubsetEvalHook(DistEvalHook):
    """Distributed counterpart of :class:`SubsetEvalHook`."""

    def __init__(self, dataloader, subset_name, shared_metrics, **kwargs):
        super(DistSubsetEvalHook, self).__init__(dataloader, **kwargs)
        self.subset_name = subset_name
        self.shared_metrics = shared_metrics

    # NOTE: the stock EvalHook calls runner.log_buffer.clear() here. That drops
    # 'data_time' (recorded back in before_train_iter) while IterTimerHook
    # re-adds 'time' immediately afterwards. TextLoggerHook then sees 'time',
    # decides the record is a training log, and dies with KeyError: 'data_time'
    # -- which is why runs crashed at the very first evaluation. Skipping the
    # clear keeps the buffer self-consistent and lets the training metrics and
    # every subset's metrics land in one record.
    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        import os.path as osp
        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            self.evaluate(runner, results)

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        import os.path as osp
        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            self.evaluate(runner, results)

    def evaluate(self, runner, results):
        runner.logger.info(
            f'evaluation on subset [{self.subset_name}] '
            f'({len(self.dataloader.dataset)} images)')
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            self.shared_metrics[f'{self.subset_name}/{name}'] = val
        runner.log_buffer.output.update(self.shared_metrics)
        runner.log_buffer.ready = True



def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
    
def train_segmentor_4subset(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        # Which subsets to evaluate on, in order. Prefer an explicit
        # `eval_sets` list in the config; otherwise auto-discover every
        # `test_*` key under cfg.data; otherwise fall back to cfg.data.val.
        eval_sets = cfg.get('eval_sets', None)
        if eval_sets is None:
            eval_sets = [k for k in cfg.data.keys() if k.startswith('test_')]
        if not eval_sets:
            eval_sets = ['val']

        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistSubsetEvalHook if distributed else SubsetEvalHook

        # All subset hooks share one dict so a full evaluation round lands in
        # a single log record instead of overwriting each other.
        shared_metrics = dict()
        for key in eval_sets:
            if key not in cfg.data:
                raise KeyError(
                    f"eval subset '{key}' is listed in eval_sets but missing "
                    f"from cfg.data (have: {sorted(cfg.data.keys())})")
            subset_name = key[len('test_'):] if key.startswith('test_') else key
            dataset_i = build_dataset(cfg.data[key], dict(test_mode=True))
            loader_i = build_dataloader(
                dataset_i,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            logger.info(f'eval subset [{subset_name}]: {len(dataset_i)} images '
                        f'from {cfg.data[key]["data_root"]}')
            runner.register_hook(
                eval_hook(
                    loader_i,
                    subset_name=subset_name,
                    shared_metrics=shared_metrics,
                    **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)