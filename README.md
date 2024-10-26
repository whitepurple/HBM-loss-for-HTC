# HBM loss
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchy-aware-biased-bound-margin-loss/hierarchical-multi-label-classification-on-17)](https://paperswithcode.com/sota/hierarchical-multi-label-classification-on-17?p=hierarchy-aware-biased-bound-margin-loss)\
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchy-aware-biased-bound-margin-loss/hierarchical-multi-label-classification-on-18)](https://paperswithcode.com/sota/hierarchical-multi-label-classification-on-18?p=hierarchy-aware-biased-bound-margin-loss)\
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchy-aware-biased-bound-margin-loss/hierarchical-multi-label-classification-on-19)](https://paperswithcode.com/sota/hierarchical-multi-label-classification-on-19?p=hierarchy-aware-biased-bound-margin-loss)

This repository contains code for the ACL24 (findings) paper: [Hierarchy-aware Biased Bound Margin Loss Function for Hierarchical Text Classification
](https://aclanthology.org/2024.findings-acl.457/)

Keywords: Hierarchical Text Classification, Classification Loss, Label Imbalance

TL;DR: This paper presents the Hierarchy-aware Biased Bound Margin (HBM) loss, a novel approach for unit-based hierarchical text classification models, addressing static thresholding and label imbalance.

![HBM loss](./figure/fig2.svg)

## Requirements
```
torch
lightning==2.0.0
torchmetrics==1.2.1
transformers
wandb
ipykernel
anytree
hydra-core
omegaconf
```

## Download data
Move the downloaded data to ```./data/{dataset name}/raw``` according to the license of each dataset.

## Data Preprocess
- RCV1-v2

    ```bash
    python src/preprocess.py \
            --name=RCV1v2 \
            --raw_dir={raw_data_dir} \
            --save_dir={save_dir}
    ```
- NYT
    ```bash
    python src/preprocess.py \
            --name=NYT \
            --raw_dir={raw_data_dir} \
            --save_dir={save_dir}
    ```
- EURLEX57K
    ```bash
    python src/preprocess.py \
            --name=EURLEX57K \
            --raw_dir={raw_data_dir} \
            --save_dir={save_dir}  \
            --hierarchy_file={EURLEX57K.json file path}
    ```
    + run ```HBM-loss-for-HTC/src/preprocess/tree.ipynb``` to create DAG to Tree hierarchy file

## Caching data

- Train data
    ```bash
    python src/dataset/caching.py data=NYT stage=TRAIN
    ```

- Val data
    ```bash
    python src/dataset/caching.py data=EURLEX57K stage=VAL
    ```

- Test data with num_workers=4, chunk_size=100000
    ```bash
    python src/dataset/caching.py data=RCV1v2 stage=TEST num_workers=4 chunk_size=100000
    ```

- or Run ```caching.sh```

## Training model with HBM loss

- in RCV1v2, with no log_skip

    ```bash
    python main.py data=RCV1v2 name=HiDEC-HBM trainer.log_skip=0
    ```

- in NYT, devices=[0,1]

    ```bash
    python main.py data=NYT name=HiDEC-HBM devices=01
    ```

- in EURLEX57K with wandb logger

    ```bash
    python main.py data=EURLEX57K name=HiDEC-HBM logger=wandb
    ```

## Inference model
- with best model

    ```bash
    python main.py data=EURLEX57K name=HiDEC-HBM do_train=false
    ```

- with specific model weight
    
    ```bash
    python main.py data=EURLEX57K name=HiDEC-HBM do_train=false ckpt_path={~.ckpt file path}
    ```

## Main table

Micro F1 (MiF1) and Macro F1 (MaF1) are the average performances of 10 runs with random weight initialization.

| Model | RCV1v2 MiF1 | RCV1v2 MaF1 | NYT MiF1 | NYT MaF1 | EURLEX57K MiF1 | EURLEX57K MaF1 |
|------------|----------|-----------|--------|----------|---------|--------|
|  [HPT](https://aclanthology.org/2022.emnlp-main.246/) ([ZLPR](https://arxiv.org/abs/2208.02955)) | 87.26 | 69.53 | 80.42 | **70.42** | - | - |
| [HiDEC](https://ojs.aaai.org/index.php/AAAI/article/view/26520) (BCE) | **87.96** | 69.97 | 79.99 | 69.64 | **75.29** | - |

| HPT | RCV1v2 MiF1 | RCV1v2 MaF1 | NYT MiF1 | NYT MaF1 | EURLEX57K MiF1 | EURLEX57K MaF1 |
|-------|----------|-----------|--------|----------|---------|--------|
| with BCE  | 87.65±0.11 | 69.87±0.40 | 79.49±0.22 | 68.66±0.30 | 71.57±0.58 | 25.34±0.59 |
| with ZLPR | **87.82**±0.14 | 70.23±0.31 | 80.04±0.23 | 69.69±0.49 | 75.54±0.20 | 28.46±0.26 |
| with **HBM**  | **87.82**±0.06 | **70.55**±0.13 | **80.42**±0.12 | **70.23**±0.18 | **75.78**±0.15 | **28.70**±0.22 |

| HiDEC | RCV1v2 MiF1 | RCV1v2 MaF1 | NYT MiF1 | NYT MaF1 | EURLEX57K MiF1 | EURLEX57K MaF1 |
|-------|----------|-----------|--------|----------|---------|--------|
| with BCE   | 87.70±0.12 | 70.82±0.20 | 80.13±0.16 | 69.80±0.24 | 75.14±0.19 | 27.91±0.11 |
| with ZLPR  | 87.59±0.18 | 70.61±0.36 | 80.25±0.21 | 70.14±0.23 | 76.16±0.16 | 28.68±0.15 |
| with **HBM**   | **87.81**±0.09 | **71.47**±0.20 | **80.52**±0.18 | **70.69**±0.19 | **76.48**±0.12 | **28.77**±0.11 |


## Cite

```bigquery
@inproceedings{kim-etal-2024-hierarchy,
    title = "Hierarchy-aware Biased Bound Margin Loss Function for Hierarchical Text Classification",
    author = "Kim, Gibaeg  and
      Im, SangHun  and
      Oh, Heung-Seon",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.457",
    pages = "7672--7682",
    abstract = "Hierarchical text classification (HTC) is a challenging problem with two key issues: utilizing structural information and mitigating label imbalance. Recently, the unit-based approach generating unit-based feature representations has outperformed the global approach focusing on a global feature representation. Nevertheless, unit-based models using BCE and ZLPR losses still face static thresholding and label imbalance challenges. Those challenges become more critical in large-scale hierarchies. This paper introduces a novel hierarchy-aware loss function for unit-based HTC models: Hierarchy-aware Biased Bound Margin (HBM) loss. HBM integrates learnable bounds, biases, and a margin to address static thresholding and mitigate label imbalance adaptively. Experimental results on benchmark datasets demonstrate the superior performance of HBM compared to competitive HTC models.",
}
```
