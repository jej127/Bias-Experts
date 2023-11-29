## Improving Bias Mitigation through Bias Experts in Natural Language Understanding
We propose a new debiasing framework that introduces binary classifiers between the auxiliary model and the main model, coined bias experts. Specifically, each bias expert is trained on a binary classification task derived from the multi-class classification task via the One-vs-Rest approach. Experimental results demonstrate that our proposed strategy effectively reduces the gap and consistently improves the state-of-the-art on various challenge datasets such as HANS.

This repository contains code for our paper: Improving Bias Mitigation through Bias Experts in Natural Language Understanding. For a detailed description and experimental results, please refer to the paper.

## Requirements
- Python 3
- Transformers
- Numpy
- PyTorch

## Data
Our experiments use MNLI and HANS dataset. Download the file for MNLI from [here](https://dl.fbaipublicfiles.com/glue/data/MNLI.zip), and the file for HANS from [here](https://github.com/tommccoy1/hans). Unzip under the directory ./dataset. The dataset directory should be structured as the following:
```
└── Bias-Experts
    └── dataset 
        └── glue_multinli
            ├── train.tsv
            ├── dev_matched.tsv
            └── dev_mismatched.tsv
        └── hans
            ├── heuristics_evaluation_set.txt
```

## Running Experiments
    # Training auxiliary model
    bash run_dynamics.sh

    # Training bias experts
    bash run_last_layer_biased.sh

    # Training the main model
    bash run_last_layer1.sh

## Results
Results on MNLI and HANS
| Seed | MNLI-dev  | HANS |
| ------------- | ------------- | ------------- |
| 206 | 82.6 | 72.4 |
| 211 | 82.7 | 73.0 |
| 222 | 82.6 | 72.0 |
| 234 | 83.0 | 73.6 |

## Contact Info
For help or issues, please submit a GitHub issue.

For personal communication, please contact Eojin Jeon <skdlcm456@korea.ac.kr> or Mingyu Lee <decon9201@korea.ac.kr>.
