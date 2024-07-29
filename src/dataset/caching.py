import json
import math
import shutil

from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm
from transformers import AutoTokenizer

from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig


stages = {"TRAIN" : "train_file",
          "VAL" : "val_file",
          "TEST": "test_file"}

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def job(i, cfg, lines, tokenizer):
    tmp_file = f"/tmp/tmp_for_cache/{str(i).zfill(3)}.tmp"
    f = Path(tmp_file).open("w")

    if i == 0:
        lines = tqdm(lines, desc=f"{i}-th worker")

    for line in lines:
        instance = json.loads(line)

        # tokenizing
        input_ids = tokenizer(
                                instance["text"],
                                max_length=cfg.data.max_seq_length,
                                truncation=True
                                )["input_ids"]
        data = {
                "did" : instance["did"],
                "input_ids" : input_ids,
                "labels" : instance["labels"],
                }
        f.write(f"{json.dumps(data, ensure_ascii=False)}\n")


def caching_mp(cfg):
    stage = cfg.stage
    num_workers = cfg.num_workers
    chunk_size = cfg.chunk_size
    encoder_name = cfg.model.encoder.pretrained_model_name_or_path
    cache_dir = Path(cfg.data.cache_dir, f"{cfg.dataset_name}-{encoder_name.split('/')[-1]}", stage)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.dataset_name} caching at {cache_dir}")

    tmp_dir = Path("/tmp/tmp_for_cache")
    if not tmp_dir.exists():
        tmp_dir.mkdir()

    print(f"{cfg.dataset_name} caching start...")
    print(f"{cfg.dataset_name} {stage.lower()} data loading...", end="")
    corpus = Path(cfg.data.data_dir, cfg.data[stages[stage]]).open().readlines()
    num_lines_per_proecess = math.ceil(len(corpus)/num_workers)
    chunked_corpus = list_chunk(corpus,num_lines_per_proecess)
    del corpus
    print("END")

    print(f"{encoder_name} tokenizer loading...", end="")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    print("END")

    print(f"Prepare multiprocessing...", end="")
    jobs = [[i, cfg, chunk, tokenizer] for i, chunk in enumerate(chunked_corpus)]
    del chunked_corpus
    print("END")

    print(f"Multiprocessing start...")
    p = Pool(num_workers)
    p.starmap(job, jobs)
    p.close()
    p.join()
    del jobs
    print(f"Multiprocessing end...")

    result = sorted(list(tmp_dir.iterdir()))

    print(f"Write preprocessed result to {cache_dir}... ")
    chunk_file_idx = 0
    current_chunk_file = (cache_dir/f"{str(chunk_file_idx).zfill(3)}.jsonl").open("w")
    i = 0
    for tmp in result:
        tmp_lines = tmp.open().readlines()
        for line in tqdm(tmp_lines, total=len(tmp_lines), desc=f"{tmp} write"):
            current_chunk_file.write(line)
            i+=1
            if i == chunk_size:
                current_chunk_file.flush()
                current_chunk_file.close()
                chunk_file_idx+=1
                current_chunk_file = (cache_dir/f"{str(chunk_file_idx).zfill(3)}.jsonl").open("w")
                i = 0

    shutil.rmtree(tmp_dir)

    print(f"{cfg.dataset_name} caching end")


@hydra.main(version_base="1.1", config_name="cache", config_path='../../config')
def main(cfg: DictConfig):
    if cfg.stage == "ALL":
        for stage in ["TRAIN", "VAL", "TEST"]:
            cfg.stage = stage
            caching_mp(cfg)
    else:
        caching_mp(cfg)

if __name__ == "__main__":
    main()