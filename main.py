from pathlib import Path
from datetime import datetime
import gc

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig

from src.model.model import get_model
from src.dataset.dataset import get_dataset
from src.dataset.collator import get_collator
from src.utils.hierarchy import get_hierarchy
from src.utils.criterion import get_criterion
from src.utils.postprocess import get_postprocess
from src.utils.evaluation import EvaluationConfusionMatrix
from trainer import Trainer

@hydra.main(version_base="1.1", config_name="main", config_path='config')
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('medium')
    for _ in range(cfg.trials):
        cfg.data.batch_size = cfg.data.total_batch_size//len(str(cfg.devices))
        cfg.trainer.devices = [int(d) for d in str(cfg.devices)]
        cfg.data.data_dir = f"{get_original_cwd()}/{cfg.data.data_dir}"
        cfg.data.cache_dir = f"{get_original_cwd()}/{cfg.data.cache_dir}"
        
        hierarchy = get_hierarchy(cfg.hierarchy)
        cfg.num_labels = len(hierarchy.label2idx)
        cfg.num_target_labels = len(hierarchy.target_labels)
        
        model = get_model(cfg, hierarchy)    
        criterion =  get_criterion(cfg)
        postprocessor = get_postprocess(cfg, hierarchy)
        evaluation = EvaluationConfusionMatrix(cfg.dataset_name, hierarchy.target_metric_mask)

        dataset_cls = get_dataset(cfg.model.dataset_cls)
        if cfg.do_train:
            train_dataset = dataset_cls(cfg, stage="TRAIN", hierarchy=hierarchy)
            val_dataset = dataset_cls(cfg, stage="VAL", hierarchy=hierarchy)   
            collator = get_collator(cfg.model.collator_cls, train_dataset.num_labels)
            model.collator = collator
            train_dataloader = DataLoader(
                train_dataset,        
                batch_size=cfg.data.batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=cfg.data.dataloader_num_workers,
            )
            val_dataloader = DataLoader(
                val_dataset,        
                batch_size=cfg.data.batch_size,
                collate_fn=collator,
                num_workers=cfg.data.dataloader_num_workers,
            )
            
        if cfg.do_test:
            test_dataset = dataset_cls(cfg, stage="TEST", hierarchy=hierarchy)
            collator = get_collator(cfg.model.collator_cls, test_dataset.num_labels)
            model.collator = collator
            test_dataloader = DataLoader(
                test_dataset,        
                batch_size=256,
                collate_fn=collator,
                num_workers=cfg.data.dataloader_num_workers,
            )
        loggers = None
        with_logger = hasattr(cfg.logger, 'save_dir')
        if not with_logger or cfg.do_train==False:
            loggers = [0]
            cfg.logger.name = cfg.name
            log_dir = Path(get_original_cwd()+"/logs")
        else:
            cfg.logger.name = cfg.name
            log_dir = Path(get_original_cwd(),cfg.logger.save_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = Trainer(
            cfg,
            **cfg.trainer,
            criterion=criterion,
            postprocessor=postprocessor,
            evaluation=evaluation,
            loggers=loggers,
        )
        try:
            if with_logger:
                cfg.logger_id = trainer.fabric.logger.experiment.id
            else:
                cfg.logger_id = str(trainer.seed)
            save_config(cfg)
        except:
            pass
        
        if cfg.do_train:
            trainer.fit(model, train_dataloader, val_dataloader, cfg.ckpt_path)
        
        if cfg.do_test:
            if cfg.do_train:
                trainer.test(model, test_dataloader, None)
            elif cfg.ckpt_path is not None:
                trainer.test(model, test_dataloader, cfg.ckpt_path)
            else:
                for ckpt_path in Path(cfg.checkpoint.checkpoint_dir,
                                        cfg.logger.project,
                                        cfg.data.dataset,
                                        cfg.logger.name).iterdir():
        
                    best_ckpt_path = sorted(ckpt_path.iterdir(),reverse=True)[0]
                    print(best_ckpt_path)
                    trainer.test(model, test_dataloader, best_ckpt_path)
            
        gc.collect()
        
def save_config(cfg: DictConfig):
    now = datetime.now()
    config_dir = Path("config/run_logs/",
                      cfg.dataset_name,
                      cfg.name,
                      f"{now:%Y-%m-%d-%H%M%S}")
    config_dir = get_original_cwd()/config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "config.yaml"
    config_file.write_text(OmegaConf.to_yaml(cfg))
    hydra_file = config_dir / "hydra.yaml"
    hydra_file.write_text(OmegaConf.to_yaml(HydraConfig.instance().cfg))
    overrides_file = config_dir / "overrides.yaml"
    overrides_file.write_text(OmegaConf.to_yaml(HydraConfig.get().overrides.task))


if __name__ == "__main__":
    main()