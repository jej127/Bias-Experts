## Improving Bias Mitigation through Bias Experts in Natural Language Understanding
We propose a new debiasing framework that introduces binary classifiers between the auxiliary model and the main model, coined bias experts. Specifically, each bias expert is trained on a binary classification task derived from the multi-class classification task via the One-vs-Rest approach. Experimental results demonstrate that our proposed strategy effectively reduces the gap and consistently improves the state-of-the-art on various challenge datasets such as HANS.

This repository contains code for our paper: Improving Bias Mitigation through Bias Experts in Natural Language Understanding. For a detailed description and experimental results, please refer to the paper.

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

## Expected Results
Results on MNLI and HANS
| Seed | MNLI-dev  | HANS |
| ------------- | ------------- | ------------- |
| 206 | 82.9 | 73.8 |
| 211 | 82.8 | 73.1 |
| 222 | 82.7 | 72.6 |

## Contact Info
.
