"""
Script to train BERT on MNLI with our loss function

Modified from the old "run_classifier" script from
https://github.com/huggingface/pytorch-transformer
"""

from collections import namedtuple
import os
from os.path import join, exists
import random
from typing import List, Dict, Iterable, Union
import config
import numpy as np
import logging
import argparse
from transformers import BertTokenizer, WEIGHTS_NAME, CONFIG_NAME
import torch
from utils_cf import Processor, process_par
import utils_cf
import json
import re
from nltk import word_tokenize
from tqdm import trange, tqdm
from predictions_analysis import visualize_predictions
from bert_poe import BertForClassification, BertForPOE
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler

_IGNORED_TOKENS = [".", "?", "!", "-"]
#NEG_WORDS = set(["not", "no", "n't", "none", "nobody", "nothing", "neither", "nowhere", "never", "cannot", "nor"])
NEG_WORDS = set(["no", "nobody", "nothing", "never"])

import nltk
nltk.download('punkt')

HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"

NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}
REV_NLI_LABEL_MAP = {i: k for i, k in enumerate(NLI_LABELS)}
NLI_LABEL_MAP["hidden"] = NLI_LABEL_MAP["entailment"]

FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
FEVER_LABEL_MAP = {k: i for i, k in enumerate(FEVER_LABELS)}

TextPairExample = namedtuple("TextPairExample", ["id", "text_a", "text_b", "label"])

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, example_id, input_ids, segment_ids, label_id, bias):
        self.example_id = example_id
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.bias = bias

class ExampleConverter(Processor):
    def __init__(self, max_seq_length, tokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def process(self, data: Iterable):
        features = []
        tokenizer = self.tokenizer
        max_seq_length = self.max_seq_length

        for example in data:
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            features.append(
                InputFeatures(
                    example_id=example.id,
                    input_ids=np.array(input_ids),
                    segment_ids=np.array(segment_ids),
                    label_id=example.label,
                    bias=None
                ))
        return features

class SortedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, seed):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.seed = seed
        if batch_size == 1:
            raise NotImplementedError()
        self._epoch = 0

    def __iter__(self):
        rng = np.random.RandomState(self._epoch + 601767 + self.seed)
        n_batches = len(self)
        batch_lens = np.full(n_batches, self.batch_size, np.int32)

        # Randomly select batches to reduce by size 1
        extra = n_batches * self.batch_size - len(self.data_source)
        batch_lens[rng.choice(len(batch_lens), extra, False)] -= 1

        batch_ends = np.cumsum(batch_lens)
        batch_starts = np.pad(batch_ends[:-1], [1, 0], "constant")

        if batch_ends[-1] != len(self.data_source):
            print(batch_ends)
            raise RuntimeError()

        bounds = np.stack([batch_starts, batch_ends], 1)
        rng.shuffle(bounds)

        for s, e in bounds:
            yield np.arange(s, e)

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

class InputFeatureDataset(Dataset):
    def __init__(self, examples: List[InputFeatures]):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

def ensure_mnli_is_downloaded():
    mnli_source = config.GLUE_SOURCE
    print(config.GLUE_SOURCE)
    if exists(mnli_source) and len(os.listdir(mnli_source)) > 0:
        return
    else:
        raise Exception("Download MNLI from Glue and put files under glue_multinli")

def load_mnli(is_train, sample=None, custom_path=None) -> List[TextPairExample]:
    ensure_mnli_is_downloaded()
    if is_train:
        filename = join(config.GLUE_SOURCE, "train.tsv")
    else:
        if custom_path is None:
            filename = join(config.GLUE_SOURCE, "dev_matched.tsv")
        else:
            filename = join(config.GLUE_SOURCE, custom_path)

    logging.info("Loading mnli " + ("train" if is_train else "dev"))
    with open(filename) as f:
        f.readline()
        lines = f.readlines()

    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample, replace=False)
                                                                
    out = []
    for line in lines:
        line = line.split("\t")
        out.append(TextPairExample(line[0], line[8], line[9], NLI_LABEL_MAP[line[-1].rstrip()]))
    return out

def load_snli(split, sample=None, custom_path=None) -> List[TextPairExample]:
    if split == 'train' :
        filename = join("../dataset/SNLI/train.tsv")
    elif split == 'dev' :
        filename = join("../dataset/SNLI/dev.tsv")
    else :
        filename = join("../dataset/SNLI/test.tsv")

    logging.info("Loading snli " + split)
    with open(filename) as f:
        f.readline()
        lines = f.readlines()

    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample, replace=False)
                                                                
    out = []
    for line in lines:
        line = line.split("\t")
        out.append(TextPairExample("%s-%s" % (split, line[0]), line[7], line[8], NLI_LABEL_MAP[line[-1].rstrip()]))
    return out

def load_wanli(is_train, data_dir, sample=None):
    out = []
    file_path = 'train.jsonl' if is_train else 'test.jsonl'
    full_path = join(data_dir, file_path)
    logging.info("Loading jsonl from {}...".format(full_path))
    with open(full_path, 'r') as jsonl_file:
        for i, line in enumerate(jsonl_file):
            example = json.loads(line)
            label = example["gold"]
            if label == '-':
                continue
            if not "pairID" in example.keys():
                id = i
            else:
                id = example["pairID"]
            text_a = example["premise"]
            text_b = example["hypothesis"]
            out.append(TextPairExample(id, text_a, text_b, NLI_LABEL_MAP[label]))
    if sample:
        random.shuffle(out)
        out = out[:sample]
    return out

def load_fever(is_train, custom_path=None, sample=None):
    out = []
    if custom_path is not None :
        full_path = custom_path
    elif is_train :
        full_path = '../dataset/FEVER/nli.train.jsonl'
    else :
        full_path = '../dataset/FEVER/nli.dev.jsonl'
    logging.info("Loading jsonl from {}...".format(full_path))
    with open(full_path, 'r') as jsonl_file:
        for i, line in enumerate(jsonl_file):
            example = json.loads(line)
            id = i
            text_a = example["claim"]
            text_b = example["evidence"] if "evidence" in example.keys() else example["evidence_sentence"]
            label = example["gold_label"] if "gold_label" in example.keys() else example["label"]
            out.append(TextPairExample(id, text_a, text_b, FEVER_LABEL_MAP[label]))
    if sample:
        random.shuffle(out)
        out = out[:sample]
    return out

def load_jsonl(file_path, data_dir, sample=None):
    out = []
    full_path = join(data_dir, file_path)
    logging.info("Loading jsonl from {}...".format(full_path))
    with open(full_path, 'r') as jsonl_file:
        for i, line in enumerate(jsonl_file):
            example = json.loads(line)

            label = example["gold_label"]
            if label == '-':
                continue

            if not "pairID" in example.keys():
                id = i
            else:
                id = example["pairID"]
            text_a = example["sentence1"]
            text_b = example["sentence2"]

            out.append(TextPairExample(id, text_a, text_b, NLI_LABEL_MAP[label]))

    if sample:
        random.shuffle(out)
        out = out[:sample]

    return out

def load_all_test_jsonl():
    test_datasets = []
    test_datasets.append(("mnli_test_m", load_jsonl("multinli_0.9_test_matched_unlabeled.jsonl",
                                                    config.MNLI_TEST_SOURCE)))
    test_datasets.append(("mnli_test_mm", load_jsonl("multinli_0.9_test_mismatched_unlabeled.jsonl",
                                                     config.MNLI_TEST_SOURCE)))
    test_datasets.append(("mnli_test_hard_m", load_jsonl("multinli_0.9_test_matched_unlabeled_hard.jsonl",
                                                         config.MNLI_HARD_SOURCE)))
    test_datasets.append(("mnli_test_hard_mm", load_jsonl("multinli_0.9_test_mismatched_unlabeled_hard.jsonl",
                                                          config.MNLI_HARD_SOURCE)))
    return test_datasets

def load_hans(n_samples=None, filter_label=None, filter_subset=None) -> List[
    TextPairExample]:
    out = []

    if filter_label is not None and filter_subset is not None:
        logging.info("Loading hans subset: {}-{}...".format(filter_label, filter_subset))
    else:
        logging.info("Loading hans all...")

    src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
    if not exists(src):
        logging.info("Downloading source to %s..." % config.HANS_SOURCE)
        utils_cf.download_to_file(HANS_URL, src)

    with open(src, "r") as f:
        f.readline()
        lines = f.readlines()

    if n_samples is not None:
        lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples,
                                                                replace=False)

    for line in lines:
        parts = line.split("\t")
        label = parts[0]

        if filter_label is not None and filter_subset is not None:
            if label != filter_label or parts[-3] != filter_subset:
                continue

        if label == "non-entailment":
            label = 0
        elif label == "entailment":
            label = 1
        else:
            raise RuntimeError()
        s1, s2, pair_id = parts[5:8]
        out.append(TextPairExample(pair_id, s1, s2, label))
    return out

def load_hans_subsets():
    src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
    if not exists(src):
        logging.info("Downloading source to %s..." % config.HANS_SOURCE)
        utils_cf.download_to_file(HANS_URL, src)

    hans_datasets = []
    labels = ["entailment", "non-entailment"]
    subsets = set()
    with open(src, "r") as f:
        for line in f.readlines()[1:]:
            line = line.split("\t")
            subsets.add(line[-3])
    subsets = [x for x in subsets]

    for label in labels:
        for subset in subsets:
            name = "hans_{}_{}".format(label, subset)
            examples = load_hans(filter_label=label, filter_subset=subset)
            hans_datasets.append((name, examples))

    return hans_datasets

def convert_examples_to_features(
        examples: List[TextPairExample], max_seq_length, tokenizer, n_process=1):
    converter = ExampleConverter(max_seq_length, tokenizer)
    return process_par(examples, converter, n_process, chunk_size=2000, desc="featurize")

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def collate_input_features(batch: List[InputFeatures]):
    max_seq_len = max(len(x.input_ids) for x in batch)
    sz = len(batch)

    input_ids = np.zeros((sz, max_seq_len), np.int64)
    segment_ids = np.zeros((sz, max_seq_len), np.int64)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)
    for i, ex in enumerate(batch):
        input_ids[i, :len(ex.input_ids)] = ex.input_ids
        segment_ids[i, :len(ex.segment_ids)] = ex.segment_ids
        mask[i, :len(ex.input_ids)] = 1

    input_ids = torch.as_tensor(input_ids)
    segment_ids = torch.as_tensor(segment_ids)
    label_ids = torch.as_tensor(np.array([x.label_id for x in batch], np.int64))

    # include example ids for test submission
    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = torch.zeros(len(batch)).long()
    return example_ids, input_ids, mask, segment_ids, label_ids

def build_eval_dataloader(data: List[InputFeatures], batch_size):
    ds = InputFeatureDataset(data)
    return DataLoader(ds, batch_size=batch_size, sampler=SequentialSampler(ds),
                      collate_fn=collate_input_features)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def sentence_to_words(sent: Union[str, List[str]], ignored: List[str] = None, lowercase=True):
    if ignored is None:
        ignored = _IGNORED_TOKENS
    if isinstance(sent, str):
        sent = word_tokenize(sent)

    assert isinstance(sent, list)
    regex = re.compile('[' + "".join(ignored).replace('.', r'\.').replace('?', r'\?').replace('-', r'\-') + ']')
    if lowercase:
        return [regex.sub('', word.lower()) for word in sent if word not in ignored]
    else:
        return [regex.sub('', word) for word in sent if word not in ignored]

def _prem_hypothesis_to_words(premise: str, hypothesis: str, lowercase=True):
    prem_words = sentence_to_words(premise, lowercase=lowercase)
    hyp_words = sentence_to_words(hypothesis, lowercase=lowercase)
    return prem_words, hyp_words

def percent_lexical_overlap(premise: str, hypothesis: str, get_hans_new_features=False, lowercase=True):
    r"""Check if a given premise and hypothesis lexically overlap.
    :param premise: The premise
    :param hypothesis: The hypothesis
    :param get_hans_new_features: If True, the returned overlap percentage is calculated w.r.t. the hypothesis.
    Otherwise, it is calculated w.r.t. the premise.
    :return:
        overlap_percent: The percentage of overlapping words (types) in the hypothesis the are also in
        the premise.
    """
    prem_words, hyp_words = _prem_hypothesis_to_words(premise, hypothesis, lowercase=lowercase)
    num_overlapping = len(list(set(hyp_words) & set(prem_words)))
    overlap_percent = num_overlapping / len(set(hyp_words)) #if len(set(prem_words)) > 0 else 0
    return overlap_percent

def indexing(p) :
    if p == 1.0 : return 0
    elif 0.8 <= p < np.around([1.0],2) : return 1
    elif 0.6 <= p < np.around([0.8],2) : return 2
    elif 0.4 <= p < np.around([0.6],2) : return 3
    elif 0.2 <= p < np.around([0.4],2) : return 4
    elif 0.0 < p < np.around([0.2],2) : return 5
    else : return 6

def is_neg(sentence: str, lowercase=True) :
    words = sentence_to_words(sentence, lowercase=lowercase)
    return int(len(set(words) & NEG_WORDS) > 0)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed for randomized elements in the training")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--n_processes", type=int, default=4,
                        help="Processes to use for pre-processing")
    parser.add_argument("--sorted", action="store_true",
                        help='Sort the data so most batches have the same input length,'
                             ' makes things about 2x faster.')
    parser.add_argument("--fusion_mode", type=str, default='hm', choices=['rubi', 'hm', 'sum', 'fc'],
                        help='Fusion function')
    parser.add_argument("--train_data", type=str, required=True, choices=['MNLI', 'FEVER', 'SNLI'],
                        help='Training data')   
    parser.add_argument("--do_eval_on_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--postfix", type=str, default="",
                        help='postfix to be added to file name')
    parser.add_argument("--save", action='store_true',
                        help='Save or not')
    args = parser.parse_args()
    utils_cf.add_stdout_logger()

    # Evaluation datasets
    if args.train_data == 'MNLI' :
        eval_datasets = [("mnli_dev_m", load_mnli(False))]
        #eval_datasets = [("mnli_dev_m", load_mnli(False)),
                         #("mnli_dev_mm", load_mnli(False, custom_path="dev_mismatched.tsv"))]
        #eval_datasets += [("hans", load_hans())]
        #eval_datasets += [("wanli", load_wanli(False, '../dataset/wanli'))]
        #eval_datasets += load_hans_subsets()
        if args.do_eval_on_train:
            eval_datasets = [("mnli_train", load_mnli(True))]
            #eval_datasets = [("mnli_dev_m", load_mnli(False)), ("mnli_train", load_mnli(True))]
    elif args.train_data == 'FEVER' :
        eval_datasets = [("fever_dev", load_fever(False))]
        eval_datasets += [("fever_symmetric_dev_v1", load_fever(False, '../dataset/FEVER-symmetric-generated/nli.dev.jsonl'))]
        eval_datasets += [("fever_symmetric_dev_v2", load_fever(False, '../dataset/FEVER-symmetric-generated/fever_symmetric_dev.jsonl'))]
        eval_datasets += [("fever_symmetric_test_v2", load_fever(False, '../dataset/FEVER-symmetric-generated/fever_symmetric_test.jsonl'))]
        if args.do_eval_on_train:
            eval_datasets = [("fever_train", load_fever(True))]
    elif args.train_data == 'SNLI' :
        eval_datasets = [("snli_dev", load_snli("dev"))]
        eval_datasets += [("snli_test", load_snli("test"))]
        eval_datasets += [("snli_hard", load_jsonl("snli_1.0_test_hard.jsonl", "../dataset/SNLI"))]
        if args.do_eval_on_train :
            eval_datasets = [("snli_pooled", load_snli("train") + load_snli("dev") + load_snli("test"))]

    # Evaluation
    for name, eval_examples in eval_datasets :
        data = []
        for ex in tqdm(eval_examples, desc="computing overlap") :
            idx, text_a, text_b, label = ex
            text_pair = [text_a,text_b]
            p = percent_lexical_overlap(text_a, text_b)
            neg = is_neg(text_b)
            datum = {"id": idx, "label": label, "overlap": p, "Negative": neg}
            data.append(datum)

        file_path = join(args.output_dir, f"overlap_info_{name}.json")
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)
            logging.info("Prediction for non-overlap examples saved!")

if __name__ == "__main__" :
    main()