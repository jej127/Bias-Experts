## Improving Bias Mitigation through Bias Experts in Natural Language Understanding
We propose a new debiasing framework that introduces binary classifiers between the auxiliary model and the main model, coined bias experts. Specifically, each bias expert is trained on a binary classification task derived from the multi-class classification task via the One-vs-Rest approach. Experimental results demonstrate that our proposed strategy effectively reduces the gap and consistently improves the state-of-the-art on various challenge datasets such as HANS.

This repository contains code for our paper: Improving Bias Mitigation through Bias Experts in Natural Language Understanding. For a detailed description and experimental results, please refer to the paper.

## Results
| Method | MNLI-dev  | HANS | FEVER-dev | FEVER-symm | QQP-dev | PAWS |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| BERT-base | 84.5 | 62.4 | 85.6 | 63.1 | 91.0 | 33.5 |
| PoE (Sanh et al., 2021)  | 83.3 | 67.9 | 84.8 | 65.7 | 88.0 | 46.4 |
| Bias Experts (ours)  | 82.7 | **72.6** | 85.6 | **68.1** | 86.8 | **58.1** |

## Running experiments

    # Training bias experts
    bash run_last_layer_biased.sh

    # Training the main model
    bash run_last_layer1.sh

## Contact Info
For help or issues using bias experts, please submit a GitHub issue.

For personal communication related to our work, please contact Eojin Jeon `skdlcm456@korea.ac.kr` or Mingyu Lee `decon9201@korea.ac.kr`.
