## Improving Bias Mitigation through Bias Experts in Natural Language Understanding
We propose a new debiasing framework that introduces binary classifiers between the auxiliary model and the main model, coined bias experts. Specifically, each bias expert is trained on a binary classification task derived from the multi-class classification task via the One-vs-Rest approach. Experimental results demonstrate that our proposed strategy effectively reduces the gap and consistently improves the state-of-the-art on various challenge datasets such as HANS.

This repository contains code for our paper: Improving Bias Mitigation through Bias Experts in Natural Language Understanding. For a detailed description and experimental results, please refer to the paper.

## Results
| Method | MNLI-dev  | HANS | FEVER-dev | FEVER-symm | QQP-dev | PAWS |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| BERT-base | 84.5 | 62.4 | 85.6 | 63.1 | 91.0 | 33.5 |
| PoE (Sanh et al., 2021)  | 83.3 | 67.9 | 84.8 | 65.7 | 88.0 | 46.4 |
| Bias Experts (ours)  | 82.7 | **72.6** | 85.6 | **68.1** | 86.8 | **58.1** |

## Requirements
- Python 3
- Transformers
- Numpy
- PyTorch

## Data
Our experiments use MNLI dataset. Download the file from [here](https://dl.fbaipublicfiles.com/glue/data/MNLI.zip), and unzip under the directory ./dataset The dataset directory should be structured as the following:
```
└── dataset 
    └── MNLI
        ├── train.tsv
        ├── dev_matched.tsv
        ├── dev_mismatched.tsv
        ├── dev_mismatched.tsv
```

## Running Experiments
    # Training auxiliary model
    bash run_dynamics.sh

    # Training bias experts
    bash run_last_layer_biased.sh

    # Training the main model
    bash run_last_layer1.sh

## Contact Info
.
