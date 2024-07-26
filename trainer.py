import math
import random
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy, SingleDeviceStrategy
from lightning.fabric.strategies.ddp import DDPStrategy
from lightning_utilities.core.apply_func import apply_to_collection

from omegaconf import DictConfig


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        precision: Union[str, int] = "32-true",
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        validation_frequency: Union[float, int] = 1,    
        use_distributed_sampler: bool = True,
        gradient_clip_val: float = 1.0,
        deterministic: bool = False,
        benchmark: bool = False,
        seed: int = -1,
        log_skip: int = 0,
        devices: Union[List[int], str, int] = "auto",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        criterion=None,
        postprocessor=None,
        evaluation=None,
    ) -> None:
        """
        * Args:
            * accelerator: 
                The hardware to run on. Possible choices are:
                ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
            * strategy: 
                Strategy for how to run across multiple devices. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            * precision: 
                _PRECISION_INPUT_INT = Literal[16, 32, 64]
                _PRECISION_INPUT_STR_ALIAS = Literal["16", "bf16", "32", "64"]
                _PRECISION_INPUT_STR = Literal["16-mixed", "bf16-mixed", "32-true", "64-true"]
            * max_epochs: 
                The maximum number of epochs to train
            * max_steps: 
                The maximum number of (optimizer) steps to train
            * grad_accum_steps: 
                How many batches to process before each optimizer step
            * validation_frequency: 
                How many epochs to run before each validation epoch.
            * use_distributed_sampler: 
                Wraps the sampler of each dataloader with a respective distributed-aware sampler
                in case of distributed training.
            * gradient_clip_val:
                Gradient clipping value to prevent gradient explosion.
            * deterministic:
                Flag to enable deterministic behavior in the training process.
            * benchmark:
                Flag to enable benchmark mode for optimized performance during training.
            * seed:
                Seed value for random number generation to ensure reproducibility of training results.
                if seed == -1, Use random seed.
            * devices: 
                Number of devices to train on (``int``),
                which GPUs to train on (``list`` or ``str``), or ``"auto"``.
                The value applies per node.
            * plugins: One or several custom plugins
            * callbacks: 
                A single callback or a list of callbacks. The following hooks are supported:
                    - on_train_epoch_start
                    - on train_epoch_end
                    - on_train_batch_start
                    - on_train_batch_end
                    - on_before_backward
                    - on_after_backward
                    - on_before_zero_grad
                    - on_before_optimizer_step
                    - on_validation_model_eval
                    - on_validation_model_train
                    - on_validation_epoch_start
                    - on_validation_epoch_end
                    - on_validation_batch_start
                    - on_validation_batch_end
            * loggers: 
                A single logger or a list of loggers. See :meth:`~lightning.fabric.fabric.Fabric.log` for more
                information.


        Warning:
            callbacks written for the lightning trainer (especially making assumptions on the trainer), won"t work!
        """
        self.cfg = cfg
        if strategy == "ddp_find_unused_parameters_true":
            strategy = DDPStrategy(find_unused_parameters=True)
        if len(cfg.trainer.devices) == 1:
            strategy = SingleDeviceStrategy(device=cfg.trainer.devices[0])
        if loggers is None:
            wandb.finish()
            loggers = WandbLogger(**cfg.logger, group=cfg.dataset_name)
            
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )        
        self.criterion = criterion 
        self.postprocessor = postprocessor 
        self.evaluation = evaluation 
    
        self.max_epochs = max_epochs
        self.max_steps = max_steps    
        self.grad_accum_steps = grad_accum_steps
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self.gradient_clip_val = gradient_clip_val

        torch.backends.cudnn.determinstic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        self.seed = seed
        self.set_seed()
        self.log_skip = log_skip
        
        self.global_step = 0
        self.current_epoch = 0
        self.val_count = 0
        self.should_stop = False
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}
        
        self.checkpoint_dir = cfg.checkpoint.checkpoint_dir
        self._train_monitor_logs = {}
        self._train_monitor_best = {}
        for monitor, mode in zip(cfg.checkpoint.monitor, cfg.checkpoint.mode):
            self._train_monitor_logs[monitor] = [
                {
                    "steps" : -1,
                    "epoch" : -1,
                    "value" : -1 if mode=="max" else torch.inf, 
                    "mode" : mode
                }
            ]
            self._train_monitor_best[monitor] = {
                                                "path":"",
                                                "value" : -1 if mode=="max" else torch.inf
                                                }
        self._train_monitor_best['last'] = {"path":""}
        self._val_metric_returns = []
        if loggers is not None:
            self.save_hyperparameters()
    
    
    def set_seed(self):
        if self.seed == -1:
            self.seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
        seed_tensor = torch.tensor(self.seed)
        seed_tensor = self.fabric.broadcast(seed_tensor, src=0)
        self.seed = seed_tensor.item()
        self.fabric.seed_everything(self.seed)
        
    
    def save_hyperparameters(self):
        if hasattr(self.fabric.logger, "log_hyperparams"):
            self.fabric.logger.log_hyperparams(
                dict(
                    seed=self.seed,
                    logger_name=self.cfg.logger.name,
                    max_epochs=self.cfg.trainer.max_epochs,
                    **self.cfg.model,
                    **self.cfg.data,
                )
            )
  
    
    def configure_optimizers(
        self,
        model: torch.nn.Module,
        num_training_samples: int
    ) -> dict: 
        param_optimizer = list(model.named_parameters())
        no_decay = [
                    "bias", "LayerNorm.weight", 
                    # "norm.weight", #? torch.nn.TransformerDecoder
                    # "norm1.weight", "norm2.weight", "norm3.weight", #? torch.nn.TransformerDecoderLayer
                    ]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
            "weight_decay": self.cfg.optim.weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0}
        ]
        optimizer = torch.optim.AdamW(
                        params=optimizer_grouped_parameters,
                        lr=self.cfg.optim.lr,
                        eps=self.cfg.optim.adam_epsilon,
                    )
        
        scheduler_name = self.cfg.optim.lr_scheduler
        self.num_training_steps = math.ceil(num_training_samples/self.cfg.data.total_batch_size)
        total_training_steps = self.num_training_steps*self.cfg.trainer.max_epochs
        num_warmup_steps = total_training_steps*self.cfg.optim.warmup_ratio
        self.fabric.print(f"num_warmup_steps : {num_warmup_steps}")
        if scheduler_name == "c":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                            optimizer, 
                            milestones=self.cfg.optim.lr_drop_milestones,
                            gamma=self.cfg.optim.lr_drop_gamma,
                        )
        elif scheduler_name == "oc":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                self.cfg.optim.lr,
                total_steps=total_training_steps
            )
        elif scheduler_name == "wl":     ## /\\
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps, total_training_steps
            )
        elif scheduler_name == "wc":     ## /-
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
        else:
            self.fabric.print("No scheduler_fn")
            scheduler = None
        return optimizer, scheduler
    
        
    def fit(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        ckpt_path: Optional[str] = None,
        with_logger: bool = True
    ) -> None:
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the torch.nn.Module to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.
            with_logger: Load model with wandb logger. If False, use initialized logger.
        """
        model.train()
        self.fabric.launch()
        optimizer, scheduler = self.configure_optimizers(
            model, train_loader.dataset.total_dataset_size
        )
        assert optimizer is not None
        model, optimizer = self.fabric.setup(model, optimizer)
        
        # assemble state (current epoch and global step will be added in save)
        state = {"model": model, "optim": optimizer, "scheduler": None}

        # load last checkpoint if available
        if ckpt_path is not None:
            latest_checkpoint_path = ckpt_path
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path, with_logger)

                # check if we even need to train here
                if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                    self.should_stop = True
        else:
            state['scheduler'] = scheduler
        print(self.seed)
        self.set_seed()
                
        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(train_loader, use_distributed_sampler=self.use_distributed_sampler)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler)
            
        while not self.should_stop:
            self.current_epoch += 1
            self.train_loop(
                model, 
                optimizer, 
                train_loader, 
                val_loader,
                scheduler=scheduler
            )
                
            if self.should_validate_after_epoch:
                self.val_loop(state, val_loader)
            
            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True
        
        if self.fabric.is_global_zero:
            self.save(state, 'last')
        
        # reset for next fit call
        self.should_stop = False


    def train_loop(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> None:
        """The training loop running a single training epoch.

        Args:
            model: the torch.nn.Module to train
            optimizer: the optimizer, optimizing the LightningModule.
            train_loader: The dataloader yielding the training batches.
            scheduler: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightninModule.configure_optimizers` for supported values.
        """
        #self.fabric.call("on_train_epoch_start")
        
        for i in range(train_loader.dataset.chunk_count):    
            iterable = self.progbar_wrapper(
                train_loader, 
                total=len(train_loader), 
                desc=f"Ep{self.current_epoch}.{str(i).zfill(2)}"
            )
            for batch_idx, batch in enumerate(iterable):
                # end epoch if stopping training completely or max batches for this epoch reached
                if self.should_stop:
                    #self.fabric.call("on_train_epoch_end")
                    return

                #self.fabric.call("on_train_batch_start", batch, batch_idx)

                # check if optimizer should step in gradient accumulation
                should_optim_step = self.global_step % self.grad_accum_steps == 0
                if should_optim_step:
                    # currently only supports a single optimizer
                    #self.fabric.call("on_before_optimizer_step", optimizer, 0)

                    # optimizer step runs train step internally through closure
                    self.training_step(model=model, batch=batch, batch_idx=batch_idx)
                    
                    self.fabric.clip_gradients(model, 
                                               optimizer, 
                                               max_norm=self.gradient_clip_val, 
                                               error_if_nonfinite=False)
                    optimizer.step()
                    # optimizer.step(partial(self.training_step, model=model, batch=batch, batch_idx=batch_idx))
                    #self.fabric.call("on_before_zero_grad", optimizer)

                    optimizer.zero_grad()

                else:
                    # gradient accumulation -> no optimizer step
                    self.training_step(model=model, batch=batch, batch_idx=batch_idx)

                #self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)

                # this guard ensures, we only step the scheduler once per global step
                if should_optim_step:
                    if scheduler is not None:
                        scheduler.step()

                # add output values to progress bar
                if self.global_step%5==0:
                    self._format_iterable(iterable, self._current_train_return)


                # only increase global step if optimizer stepped
                self.global_step += int(should_optim_step)
            
                if self.should_validate_in_train_loop:
                    self.val_loop({"model": model, 
                                    "optim": optimizer, 
                                    "scheduler": scheduler}, val_loader)
                    
                # stopping criterion on step level
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    self.should_stop = True
                    break
                    
            #? iterable end
            train_loader = self.reload_dataloader(train_loader)

        #self.fabric.call("on_train_epoch_end")
        


    def reload_dataloader(self, dataloader, stage="TRAIN"):
        if dataloader.dataset.chunk_count > 1:
            dataloader.dataset.next_chunk()
            dataloader = DataLoader(
                dataloader.dataset,        
                batch_size=dataloader.batch_size,
                shuffle=stage=="TRAIN",
                collate_fn=dataloader.collate_fn,
                num_workers=self.cfg.data.dataloader_num_workers,
            )
            dataloader = self.fabric.setup_dataloaders(
                dataloader, 
                use_distributed_sampler=self.use_distributed_sampler
            )
        return dataloader


    def val_loop(
        self,
        state: Mapping,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ) -> None:
        """The validation loop ruunning a single validation epoch.

        Args:
            model: the torch.nn.Module to evaluate
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater then the number of batches in the ``val_loader``, this has no effect.
        """
        # no validation if val_loader wasn"t passed
        if val_loader is None:
            return

        # self.fabric.call("on_validation_model_eval")  # calls `model.eval()`
        model = state['model']
        model.eval()
        torch.set_grad_enabled(False)
        
        # self.fabric.call("on_validation_epoch_start")
        for i in range(val_loader.dataset.chunk_count):    
            iterable = self.progbar_wrapper(
                val_loader, 
                total=len(val_loader), 
                desc=f"Val.{str(i).zfill(2)}"
            )
            for batch_idx, batch in enumerate(iterable):
                # end epoch if stopping training completely or max batches for this epoch reached
                if self.should_stop or batch_idx >= limit_batches:
                    break

                # self.fabric.call("on_validation_batch_start", batch, batch_idx)
                output: Union[torch.Tensor, Mapping[str, Any]] = model.inference(batch)
                results = self.postprocessor(batch, output)
                self.evaluation.update(results["pred_labels"], batch["multi_hot_labels"])
                # self.fabric.call("on_validation_batch_end", results, batch, batch_idx)
                
            val_loader = self.reload_dataloader(val_loader, stage="VAL")
        
        # sync confusion matrix over all devices
        self.evaluation.set_data(self.fabric.all_reduce(self.evaluation.get_data(), reduce_op="sum"))
        if self.fabric.is_global_zero:
            result = self.evaluation.compute()
            result = apply_to_collection(result, torch.Tensor, lambda x: x.detach())
            result = {f"val_{k}":v.cpu().item() for k,v in result.items()}
            self.fabric.print(result)
            result['epoch'] = self.current_epoch
            self._val_metric_returns.append(result)
            if hasattr(self.fabric.logger, "log_metrics"):
                self.fabric.log_dict(result, step=self.global_step)
            
        self.evaluation.reset()
    
        # self.fabric.call("on_validation_epoch_end")

        # self.fabric.call("on_validation_model_train")    # calls `model.train()`
        model.train()
        torch.set_grad_enabled(True)
        
        if self.fabric.is_global_zero:
            for monitor in self._train_monitor_logs.keys():
                new_log = self._train_monitor_logs[monitor][-1].copy()
                new_log["epoch"] = self.current_epoch
                new_log["steps"] = self.global_step
                new_log["value"] = self._val_metric_returns[-1][monitor]
                self._train_monitor_logs[monitor].append(new_log)
                valid_prefix = f"Ep{self.current_epoch}/step{self.global_step}"
                
                best_value = self.compare_values(
                    new_log["value"], 
                    self._train_monitor_best[monitor]["value"], 
                    new_log["mode"]
                )
                if best_value != self._train_monitor_best[monitor]["value"]:
                    self._train_monitor_best[monitor]["value"] = best_value
                    self.fabric.print(f"{valid_prefix} Best {monitor} : {best_value}")
                    self.save(state, monitor, best_value)
                else:
                    self.fabric.print(f"{valid_prefix} {monitor} : {new_log['value']}")
    
    
    def compare_values(self, current_value, best_value, operation="max"):
        if operation == "max":
            if current_value > best_value:
                return current_value
        elif operation == "min":
            if current_value < best_value:
                return current_value
        return best_value
    
    
    def training_step(self, model: torch.nn.Module, batch: Any, batch_idx: int) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this
        is given as a closure to the optimizer step.

        Args:
            model: the lightning module to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch
        """
        output: Union[torch.Tensor, Mapping[str, Any]] = model(batch)
        results = self.postprocessor(batch, output)
        results.update(self.criterion(output, batch))  #*default {"loss" : Torch.tensor(value)}
        loss = results["loss"]

        # self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        # self.fabric.call("on_after_backward")  
        
        loss = self.fabric.all_reduce(loss.detach(), reduce_op="mean")
        
        if self.global_step%50==0:
            log_data = {
                "train_loss" : loss.item(),
                "epoch" : self.current_epoch
            }
            
            if hasattr(self.fabric.logger, "log_metrics"):
                self.fabric.log_dict(log_data, step=self.global_step)

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(
            loss, 
            dtype=torch.Tensor, 
            function=lambda x: x.detach()
        )

        return loss
    
    
    @property
    def should_validate_after_epoch(self) -> bool:
        """Whether to currently run validation when validation_frequency is int."""
        if isinstance(self.validation_frequency, int):
            should_validate = self.current_epoch % self.validation_frequency == 0
            if should_validate:
                return self.should_skip_validate
        return False
    
    
    @property
    def should_validate_in_train_loop(self) -> bool:
        """Whether to currently run validation when validation_frequency is float."""
        if isinstance(self.validation_frequency, float):
            if not hasattr(self, "val_freq_steps"):
                self.val_freq_steps = int(self.num_training_steps*self.validation_frequency)
            should_validate = self.global_step % self.val_freq_steps == 0
            if should_validate:
                return self.should_skip_validate
        return False
    
    
    @property
    def should_skip_validate(self) -> bool:
        if self.log_skip > self.val_count:
            self.val_count+=1
            return False
        else:
            return True


    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.
        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable


    def load(self, state: Optional[Mapping], path: str, with_logger: bool=True) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from
        """
        if state is None:
            raise RuntimeError(f"State is not served")
        
        exist_optim = False
        if "optim" in state:
            optim = state['optim']
            del state['optim']
            exist_optim = True
        
        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")
        self._train_monitor_logs=remainder.pop("_train_monitor_logs")
        self._train_monitor_best=remainder.pop("_train_monitor_best")
        self._val_metric_returns=remainder.pop("_val_metric_returns")
        self.seed=remainder.pop("seed")
        self.val_count=remainder.pop("val_count")
        if with_logger and "logger_id" in remainder:
            try:
                int(remainder['logger_id'])
            except:
                wandb.finish()
                loggers = WandbLogger(**self.cfg.logger, 
                                    group=self.cfg.dataset_name, 
                                    id=remainder.pop("logger_id"))
                self.fabric._loggers = [loggers]
            
        self.fabric.print(f"Load model from {path}")
        self.fabric.print(f"Unused Checkpoint Values: {remainder.keys()}")
        
        if exist_optim:
            state['optim'] = optim        
        # if remainder:
        #     raise RuntimeError(f"Unused Checkpoint Values: {remainder}")


    def save(self, state: Optional[Mapping], monitor:str = "", value:float=0.0) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.
        """
        whole_part, decimal_part = f"{value*100:.2f}".split(".")
        
        prefix = f"{monitor}_{whole_part.zfill(2)}_{decimal_part}"
        if state is None:
            state = {}
        if hasattr(self.fabric.logger, 'experiment'):
            logger_id = self.fabric.logger.experiment.id
        else:
            logger_id = str(self.seed)
        state.update(
            global_step=self.global_step, 
            current_epoch=self.current_epoch,
            _train_monitor_logs=self._train_monitor_logs,
            _train_monitor_best=self._train_monitor_best,
            _val_metric_returns=self._val_metric_returns,
            seed=self.seed,
            logger_id=logger_id,
            val_count=self.val_count
        )
        ckpt_name = f"{prefix}_ep_{self.current_epoch:03d}_steps_{self.global_step}.ckpt"
        path = Path(self.checkpoint_dir, 
                    self.cfg.logger.project,
                    self.cfg.dataset_name,
                    self.cfg.logger.name,
                    logger_id)
        path.mkdir(parents=True, exist_ok=True)
        path = path/ckpt_name
        self.fabric.save(path, {k:v for k,v in state.items() if k != "optim"})
        self.fabric.print(f"Save {self.cfg.logger.name} to {ckpt_name}")
        best_before_path = self._train_monitor_best[monitor]["path"]
        if best_before_path:
            self.fabric._strategy.checkpoint_io.remove_checkpoint(best_before_path)
        self._train_monitor_best[monitor]["path"] = path


    @staticmethod
    def _format_iterable(
        prog_bar, 
        candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], 
        prefix: str=""
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.
        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)
                
                
    def test(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        ckpt_path: Optional[str] = None,
        with_logger: bool = True,
        monitor: str = "val_micro_f1"
    ) -> None:
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the torch.nn.Module to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            test_loader: the training dataloader. Has to be an iterable returning batches.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.
            with_logger: load model with wandb logger flag
        """
        if test_loader is None:
            return
        
        self.fabric.launch()
       
        # setup dataloaders
        test_loader = self.fabric.setup_dataloaders(
            test_loader, 
            use_distributed_sampler=self.use_distributed_sampler
        )
        
        model = self.fabric.setup(model)
        
        # assemble state (current epoch and global step will be added in save)
        state = {"model": model, "optim": None, "scheduler": None}
        
        # load last checkpoint if available
        if ckpt_path is None:
            ckpt_path = self._train_monitor_best[monitor]['path']
            ckpt_path = self.fabric.broadcast(ckpt_path)
        
        self.load({"model": model}, ckpt_path, with_logger)
                
        self.set_seed()
        model.eval()
        self.evaluation.reset()
        torch.set_grad_enabled(False)
        
        test_result = dict()
            
        for i in range(test_loader.dataset.chunk_count):    
            iterable = self.progbar_wrapper(
                test_loader, 
                total=len(test_loader), 
                desc=f"TEST.{str(i).zfill(2)}"
            )
            for batch_idx, batch in enumerate(iterable):
                output: Union[torch.Tensor, Mapping[str, Any]] = model.inference(batch)
                results = self.postprocessor(batch, output)
                self.evaluation.update(results["pred_labels"], batch["multi_hot_labels"])
                   
            test_loader = self.reload_dataloader(test_loader, stage="TEST")
        self.evaluation.set_data(self.fabric.all_reduce(self.evaluation.get_data(), reduce_op="sum"))
        if self.fabric.is_global_zero:
            result = self.evaluation.compute()
            torch.save(self.evaluation._confmat.confmat.cpu(), ckpt_path.parent/'confmat.pt')
            result = apply_to_collection(result, torch.Tensor, lambda x: x.detach())
            result = {f"test_{k}":v for k,v in result.items()}
            if hasattr(self.fabric.logger, "log_metrics"):
                self.fabric.log_dict(result)
            test_result.update(result)
        self.fabric.print(test_result)
        self.evaluation.reset()
        model.train()
        torch.set_grad_enabled(True)
        return test_result
    
    
    def predict(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        ckpt_path: Optional[str] = None,
        with_logger: bool = False,
        monitor: str = "val_micro_f1"
    ) -> None:
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: the torch.nn.Module to train.
                Can have the same hooks as :attr:`callbacks` (see :meth:`MyCustomTrainer.__init__`).
            test_loader: the training dataloader. Has to be an iterable returning batches.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.
            with_logger: load model with wandb logger flag
        """
        if test_loader is None:
            return
        
        self.fabric.launch()
       
        # setup dataloaders
        test_loader = self.fabric.setup_dataloaders(
            test_loader, 
            use_distributed_sampler=self.use_distributed_sampler
        )
        
        model = self.fabric.setup(model)
        
        # assemble state (current epoch and global step will be added in save)
        state = {"model": model, "optim": None, "scheduler": None}
        
        # load last checkpoint if available
        if ckpt_path is None:
            ckpt_path = self._train_monitor_best[monitor]['path']
            ckpt_path = self.fabric.broadcast(ckpt_path)

        self.load({"model": model}, ckpt_path, with_logger)
        
        self.set_seed()
        model.eval()
        self.evaluation.reset()
        torch.set_grad_enabled(False)
        
        test_result = dict()
        
        test_result['logits'] = []
        test_result['pred_labels'] = []
        test_result['multi_hot_labels'] = []
            
        for i in range(test_loader.dataset.chunk_count):    
            iterable = self.progbar_wrapper(
                test_loader, 
                total=len(test_loader), 
                desc=f"TEST.{str(i).zfill(2)}"
            )
            for batch_idx, batch in enumerate(iterable):
                output: Union[torch.Tensor, Mapping[str, Any]] = model.inference(batch)
                results = self.postprocessor(batch, output)
                self.evaluation.update(results["pred_labels"], batch["multi_hot_labels"])
                
                if 'logits' in results:
                    test_result['logits'].append(results['logits'])
                if 'logits' in results:
                    test_result['pred_labels'].append(results['pred_labels'])
                test_result['multi_hot_labels'].append(batch['multi_hot_labels'])
            test_loader = self.reload_dataloader(test_loader, stage="TEST")
        self.evaluation.set_data(self.fabric.all_reduce(self.evaluation.get_data(), reduce_op="sum"))
        if self.fabric.is_global_zero:
            result = self.evaluation.compute()
            result = apply_to_collection(result, torch.Tensor, lambda x: x.detach())
            result = {f"test_{k}":v for k,v in result.items()}
            
            test_result.update(result)
        self.fabric.print(test_result)
        model.train()
        torch.set_grad_enabled(True)
        return test_result

