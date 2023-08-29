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
import math
import argparse
from transformers import BertTokenizer, get_linear_schedule_with_warmup, WEIGHTS_NAME, CONFIG_NAME
import torch
from utils_cf import Processor, process_par
import utils
import json
import re
from tqdm import trange, tqdm
from predictions_analysis import visualize_predictions
from bert_poe import BertForClassification, BertForBinaryClassification, BertForPOE
from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler, SequentialSampler
from nltk import word_tokenize

HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
_IGNORED_TOKENS = [".", "?", "!", "-"]
NEG_WORDS = set(["not", "no", "n't", "none", "nothing", "neither", "never", "cannot"])
NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}
REV_NLI_LABEL_MAP = {i: k for i, k in enumerate(NLI_LABELS)}
NLI_LABEL_MAP["hidden"] = NLI_LABEL_MAP["entailment"]

TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])

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
            tokens_a = tokenizer.tokenize(example.premise)

            tokens_b = None
            if example.hypothesis:
                tokens_b = tokenizer.tokenize(example.hypothesis)
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

def load_mnli(is_train, seed=111, sample=None, custom_path=None) -> List[TextPairExample]:
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
        lines = np.random.RandomState(26096781 + seed).choice(lines, sample, replace=False)
                                                                
    out = []
    for line in lines:
        line = line.split("\t")
        out.append(TextPairExample(line[0], line[8], line[9], NLI_LABEL_MAP[line[-1].rstrip()]))
    return out

def load_easy_hard(prefix="", no_mismatched=False):
    all_datasets = []

    all_datasets.append(("mnli_dev_matched_{}easy".format(prefix),
                         load_mnli(False, custom_path="dev_matched_{}easy.tsv".format(prefix))))
    all_datasets.append(("mnli_dev_matched_{}hard".format(prefix),
                         load_mnli(False, custom_path="dev_matched_{}hard.tsv".format(prefix))))
    if not no_mismatched:
        all_datasets.append(("mnli_dev_mismatched_{}easy".format(prefix),
                             load_mnli(False, custom_path="dev_mismatched_{}easy.tsv".format(prefix))))
        all_datasets.append(("mnli_dev_mismatched_{}hard".format(prefix),
                             load_mnli(False, custom_path="dev_mismatched_{}hard.tsv".format(prefix))))

    return all_datasets

def load_bias(bias_name, custom_path=None) -> Dict[str, np.ndarray]:
    """Load dictionary of example_id->bias where bias is a length 3 array
    of log-probabilities"""

    if custom_path is not None:  # file contains probs
        with open(custom_path, "r") as bias_file:
            all_lines = bias_file.read()
            bias = json.loads(all_lines)
            for k, v in bias.items():
                bias[k] = np.log(np.array(v))
        return bias

    if bias_name == "hans":
        if bias_name == "hans":
            bias_src = config.MNLI_WORD_OVERLAP_BIAS
        if not exists(bias_src):
            raise Exception("lexical overlap bias file is not found")
        bias = utils.load_pickle(bias_src)
        for k, v in bias.items():
            # Convert from entail vs non-entail to 3-way classes by splitting non-entail
            # to neutral and contradict
            bias[k] = np.array([
                v[0] - np.log(2.),
                v[1],
                v[0] - np.log(2.),
            ])
        return bias

    if bias_name in config.BIAS_SOURCES:
        file_path = config.BIAS_SOURCES[bias_name]
        with open(file_path, "r") as hypo_file:
            all_lines = hypo_file.read()
            bias = json.loads(all_lines)
            for k, v in bias.items():
                bias[k] = np.array(v)
        return bias
    else:
        raise Exception("invalid bias name")

def load_teacher_probs(custom_teacher=None):
    if custom_teacher is None:
        file_path = config.TEACHER_SOURCE
    else:
        file_path = custom_teacher

    with open(file_path, "r") as teacher_file:
        all_lines = teacher_file.read()
        all_json = json.loads(all_lines)

    return all_json


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

def load_hans_subsets():
    src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
    if not exists(src):
        logging.info("Downloading source to %s..." % config.HANS_SOURCE)
        utils.download_to_file(HANS_URL, src)

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
        utils.download_to_file(HANS_URL, src)

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
    if batch[0].teacher_probs is None:
        return example_ids, input_ids, mask, segment_ids, label_ids
    teacher_probs = torch.tensor(np.array([x.teacher_probs for x in batch]))
    return example_ids, input_ids, mask, segment_ids, label_ids, teacher_probs

def build_train_dataloader(data: List[InputFeatures], batch_size, seed, sorted):
    if sorted:
        data.sort(key=lambda x: len(x.input_ids))
        ds = InputFeatureDataset(data)
        sampler = SortedBatchSampler(ds, batch_size, seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_input_features)
    else:
        ds = InputFeatureDataset(data)
        return DataLoader(ds, batch_size=batch_size, sampler=RandomSampler(ds),
                          collate_fn=collate_input_features)

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

def is_neg(sentence: str, lowercase=True) :
    words = sentence_to_words(sentence, lowercase=lowercase)
    return int(len(set(words) & NEG_WORDS) > 0)

def compute_acc(eval_dataloader, model, device, labels, hans=False):
    probs = []
    for _, input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():                                                          
            inputs =   {
                        "input_ids": input_ids,
                        "attention_mask": input_mask,
                        "token_type_ids": segment_ids,
                        "labels": None,
                       }
            logits = model(**inputs)
        probs.append(torch.nn.functional.softmax(logits, 1).detach().cpu().numpy())
    probs = np.concatenate(probs, 0)
    preds = np.argmax(probs, axis=1)
    if hans: preds = (preds==1).astype('int')
    acc = simple_accuracy(preds, labels)
    confidence = np.mean(probs[range(probs.shape[0]),labels])
    return acc, confidence

def compute_acc_bin(eval_dataloader, model, device, labels, label_given):
    probs = []
    for _, input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():                                                          
            inputs =   {
                        "input_ids": input_ids,
                        "attention_mask": input_mask,
                        "token_type_ids": segment_ids,
                        "labels": None,
                       }
            logits = model(**inputs)
        probs.append(torch.sigmoid(logits.view(-1)).detach().cpu().numpy())
    probs = np.hstack(probs)
    preds = (probs > 0.5).astype('int')
    all_label_ids = (labels == label_given).astype('int')
    acc = simple_accuracy(preds, all_label_ids)
    confidence = np.mean(probs)
    return acc, confidence

def compute_loss(eval_dataloader, model, device):
    tr_loss = 0
    count = 0
    for _, input_ids, input_mask, segment_ids, label_ids in eval_dataloader:        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():                                                          
            inputs =   {
                        "input_ids": input_ids,
                        "attention_mask": input_mask,
                        "token_type_ids": segment_ids,
                        "labels": None,
                       }
            logits = model(**inputs)
            eval_loss = torch.nn.functional.cross_entropy(logits.view(-1, 3), label_ids.view(-1), reduction='sum')
            tr_loss += eval_loss.item()
            count += logits.size(0)
    return tr_loss / count

def compute_loss_bin(eval_dataloader, model, device, label_given):
    tr_loss = 0,0
    count = 0,0
    for _, input_ids, input_mask, segment_ids, label_ids in eval_dataloader:        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():                                                          
            inputs =   {
                        "input_ids": input_ids,
                        "attention_mask": input_mask,
                        "token_type_ids": segment_ids,
                        "labels": None,
                       }
            logits = model(**inputs)
            label_ids_bin = (label_ids == label_given).float()
            eval_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.view(-1), label_ids_bin.view(-1), reduction='sum')
            tr_loss += eval_loss.item()
            count += logits.size(0)
    return tr_loss / count

def pool_probability_bin2mul(label,p1,p2,p3,mode='overwhelm',tau=1.0):
    if mode=='overwhelm':
        if label==0: #cont
            p = np.array([p1,(1-p1)*p2/(p2+p3+1e-8),(1-p1)*p3/(p2+p3+1e-8)]).astype(np.float32)
        elif label==1: #ent
            p = np.array([(1-p2)*p1/(p1+p3+1e-8),p2,(1-p2)*p3/(p1+p3+1e-8)]).astype(np.float32)
        else: #neut
            p = np.array([(1-p3)*p1/(p1+p2+1e-8),(1-p3)*p2/(p1+p2+1e-8),p3]).astype(np.float32)
    elif mode=='overwhelm_log':
        if label==0: #cont
            p = np.array([p1,(1-p1)*np.log(1-p2+1e-8)/(np.log(1-p2+1e-8)+np.log(1-p3+1e-8)),(1-p1)*np.log(1-p3+1e-8)/(np.log(1-p2+1e-8)+np.log(1-p3+1e-8))]).astype(np.float32)
        elif label==1: #ent
            p = np.array([(1-p2)*np.log(1-p1+1e-8)/(np.log(1-p1+1e-8)+np.log(1-p3+1e-8)),p2,(1-p2)*np.log(1-p3+1e-8)/(np.log(1-p1+1e-8)+np.log(1-p3+1e-8))]).astype(np.float32)
        else: #neut
            p = np.array([(1-p3)*np.log(1-p1+1e-8)/(np.log(1-p1+1e-8)+np.log(1-p2+1e-8)),(1-p3)*np.log(1-p2+1e-8)/(np.log(1-p1+1e-8)+np.log(1-p2+1e-8)),p3]).astype(np.float32)
    elif mode=='softmax':
        z = np.log(np.array([p1,p2,p3])/(1-np.array([p1,p2,p3])+1e-8)+1e-8).astype(np.float32) / tau
        z -= np.max(z)
        p = np.exp(z) / np.sum(np.exp(z))
    elif mode=='simple':
        p = np.array([p1,p2,p3]).astype(np.float32)
    return p

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--custom_teacher", default=None, type=str,
                        help="The directory where teacher prediction is saved.")
    parser.add_argument("--custom_teacher_2", default=None, type=str,
                        help="The 2nd directory where teacher prediction is saved.")
    parser.add_argument("--custom_teacher_3", default=None, type=str,
                        help="The 3rd directory where teacher prediction is saved.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_on_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test and create submission.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed for randomized elements in the training")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--logging_steps", default=100, type=int,
                        help="Step for logging.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Warmup steps")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--n_processes", type=int, default=4,
                        help="Processes to use for pre-processing")
    parser.add_argument("--sorted", action="store_true",
                        help='Sort the data so most batches have the same input length,'
                             ' makes things about 2x faster.')
    parser.add_argument("--mode", type=str, required=True, choices=['vn','vn_bin','poe'],
                        help='vn : vanilla BERT, cf : Counterfactual mode')
    parser.add_argument("--q", type=float, default=0.7,
                        help='Constant used in GCE')
    parser.add_argument("--sample", type=int, default=-1,
                        help='sample or not')
    parser.add_argument("--lamda", type=float, default=0.3,
                        help='lamda for ce')  
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help='weight decay')
    parser.add_argument("--suffix", type=str,
                        help='weight decay')
    parser.add_argument("--index", type=str, required=True, choices=['loss','accuracy','p'],
                        help='which index to measure')
    parser.add_argument("--label_given", type=int, choices=[0,1,2],
                        help='label to give weight')
    parser.add_argument("--label_weight", type=float, default=2.0,
                        help='weight given to labels other than args.label_given')
    parser.add_argument("--joint_mode", type=str, default='overwhelm_log', choices=['overwhelm','overwhelm_log','softmax','simple'],
                        help='how to join three biased probabilities')
    args = parser.parse_args()
    utils.add_stdout_logger()

    output_dir = args.output_dir
    if args.do_train:
        if exists(output_dir):
            if len(os.listdir(output_dir)) > 0:
                logging.warning("Output dir exists and is non-empty")
        else:
            os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(output_dir) and os.listdir(output_dir) and args.do_train:
        logging.warning(
            "Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    num_train_optimization_steps = None
    train_examples = None
    if args.do_train:
        train_examples = load_mnli(True) if args.sample <= 0 else load_mnli(True, seed=args.seed, sample=args.sample)
        eval_examples = load_mnli(False)
        num_train_optimization_steps = int(math.ceil(len(train_examples) / args.train_batch_size) / 
        args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        train_examples_align = []
        train_examples_cflct = []
        if args.mode=='poe':
            pooled_datasets = load_hans_subsets()
            for name, examples in pooled_datasets:
                if 'non' in name:
                    for i, ex in enumerate(examples):
                        train_examples_cflct.append(ex)
                else:
                    for i, ex in enumerate(examples):
                        train_examples_align.append(ex)
        else:
            train_examples_align = []
            train_examples_cflct = []
            for ex in tqdm(train_examples, desc="computing overlap") :
                _, text_a, text_b, label = ex
                p = percent_lexical_overlap(text_a, text_b)
                if p == 1.0 :
                    if label == 1: # ent
                        train_examples_align.append(ex)
                    else: # Non-ent
                        train_examples_cflct.append(ex)
                '''
                if is_neg(text_b):
                    if label == 0: # cont
                        train_examples_align.append(ex)
                    else: # Non-cont
                        train_examples_cflct.append(ex)
                '''
        logging.info("Number of bias-aligned examples: {}".format(len(train_examples_align)))
        logging.info("Number of bias-conflicting examples: {}".format(len(train_examples_cflct)))
        logging.info("Number of dev: {}".format(len(eval_examples)))

    # Model Preparation
    if args.mode == 'vn' : 
        model = BertForClassification.from_pretrained(args.bert_model, num_labels=3).to(device)
    elif args.mode == 'vn_bin' : 
        model = BertForBinaryClassification.from_pretrained(args.bert_model, label_given=args.label_given,
                                                            label_weight=args.label_weight).to(device)
    else : 
        model = BertForPOE.from_pretrained(args.bert_model, num_labels=3, lamda=args.lamda).to(device)

    # Prepare optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    if args.do_train:
        if args.warmup_steps > 0 :
            warmup_steps = args.warmup_steps
        else :
            warmup_steps = int(num_train_optimization_steps * args.warmup_proportion)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        train_features: List[InputFeatures] = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer, args.n_processes)
        train_features_align: List[InputFeatures] = convert_examples_to_features(train_examples_align, args.max_seq_length, tokenizer, args.n_processes)
        train_features_cflct: List[InputFeatures] = convert_examples_to_features(train_examples_cflct, args.max_seq_length, tokenizer, args.n_processes)
        eval_features: List[InputFeatures] = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer, args.n_processes)
        all_label_ids = np.array([x.label_id for x in train_features])
        all_label_ids_ba = np.array([x.label_id for x in train_features_align])
        all_label_ids_bc = np.array([x.label_id for x in train_features_cflct])
        all_label_ids_dev = np.array([x.label_id for x in eval_features])

        if args.mode == 'poe' :
            if args.custom_teacher_2 is not None:
                teacher_probs_map = load_teacher_probs(args.custom_teacher)
                teacher_probs_map_2 = load_teacher_probs(args.custom_teacher_2)
                teacher_probs_map_3 = load_teacher_probs(args.custom_teacher_3)
                logging.info(" Joint mode: {}".format(args.joint_mode))
                for fe in train_features:
                    p1 = teacher_probs_map[fe.example_id][0] #cont
                    p2 = teacher_probs_map_2[fe.example_id][0] #ent
                    p3 = teacher_probs_map_3[fe.example_id][0] #neut
                    p = pool_probability_bin2mul(fe.label_id,p1,p2,p3,mode=args.joint_mode)
                    fe.teacher_probs = p
            else:
                logging.info(" Joint mode is not used")
                teacher_probs_map = load_teacher_probs(args.custom_teacher)
                for fe in train_features:
                    fe.teacher_probs = np.array(teacher_probs_map[fe.example_id]).astype(np.float32)
        else :
            for fe in train_features:
                fe.teacher_probs = None

        for fe in train_features_align:
            fe.teacher_probs = None
        for fe in train_features_cflct:
            fe.teacher_probs = None
        for fe in eval_features:
            fe.teacher_probs = None

        example_map = {}
        for ex in train_examples:
            example_map[ex.id] = ex

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = build_train_dataloader(train_features, args.train_batch_size, args.seed, args.sorted)
        eval_dataloader_ba = build_eval_dataloader(train_features_align, args.eval_batch_size)
        eval_dataloader_bc = build_eval_dataloader(train_features_cflct, args.eval_batch_size)
        eval_dataloader = build_eval_dataloader(eval_features, args.eval_batch_size)

        model.train()
        loss_ema = 0
        total_steps = 0
        decay = 0.99
        losses_ba, losses_bc = [],[]
        accs_ba, accs_bc, accs_dev = [],[],[]
        pbas, pbcs = [],[]

        for _ in trange(int(args.num_train_epochs), desc="Epoch", ncols=100):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            pbar = tqdm(train_dataloader, desc="loss", ncols=100)
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)
                if args.mode == 'poe' :
                    example_ids, input_ids, mask, segment_ids, label_ids, teacher_probs = batch
                else :
                    teacher_probs = None
                    example_ids, input_ids, mask, segment_ids, label_ids = batch
                inputs = {
                        "input_ids": input_ids,
                        "attention_mask": mask,
                        "token_type_ids": segment_ids,
                        "labels": label_ids,
                        "teacher_probs": teacher_probs,
                        }
                loss,_ = model(**inputs)
                total_steps += 1
                loss_ema = loss_ema * decay + loss.cpu().detach().numpy() * (1 - decay)
                descript = "loss=%.4f" % (loss_ema / (1 - decay ** total_steps))
                pbar.set_description(descript, refresh=False)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0 :
                        tr_loss = 0
                        model.eval()
                        if args.index=='loss':
                            loss_ba = compute_loss(eval_dataloader_ba, model, device)
                            loss_bc = compute_loss(eval_dataloader_bc, model, device)
                            losses_ba.append(loss_ba)
                            losses_bc.append(loss_bc)
                            print("step: %d, ba/bc : %.4f/%.4f" % (global_step, loss_ba, loss_bc))
                        elif args.index=='accuracy':
                            acc_ba,_ = compute_acc(eval_dataloader_ba, model, device, all_label_ids_ba, hans=True)
                            acc_bc,_ = compute_acc(eval_dataloader_bc, model, device, all_label_ids_bc, hans=True)
                            acc_dev,_ = compute_acc(eval_dataloader, model, device, all_label_ids_dev)
                            accs_ba.append(acc_ba)
                            accs_bc.append(acc_bc)
                            accs_dev.append(acc_dev)
                            print("step: %d, ba/bc/dev : %.4f/%.4f/%.4f" % (global_step, acc_ba, acc_bc, acc_dev))
                        elif args.index=='p':
                            if 'bin' in args.mode:
                                _,pba = compute_acc_bin(eval_dataloader_ba, model, device, all_label_ids_ba, args.label_given)
                                _,pbc = compute_acc_bin(eval_dataloader_bc, model, device, all_label_ids_bc, args.label_given)
                                pbas.append(pba)
                                pbcs.append(1-pbc)
                                print("step: %d, ba/bc: %.4f/%.4f" % (global_step, pba, 1-pbc))
                            else:
                                _,pba = compute_acc(eval_dataloader_ba, model, device, all_label_ids_ba)
                                _,pbc = compute_acc(eval_dataloader_bc, model, device, all_label_ids_bc)
                                pbas.append(pba)
                                pbcs.append(pbc)
                                print("step: %d, ba/bc : %.4f/%.4f" % (global_step, pba, pbc))
                        model.train()

        if args.index=='loss':
            print(losses_ba)
            print(losses_bc)
        elif args.index=='accuracy':
            print(accs_ba)
            print(accs_bc)
            print(accs_dev)
        elif args.index=='p':
            print(pbas)
            print(pbcs)

        # Save model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Record the args as well
        arg_dict = {}
        for arg in vars(args):
            arg_dict[arg] = getattr(args, arg)
        with open(join(output_dir, "args.json"), 'w') as out_fh:
            json.dump(arg_dict, out_fh)

        # Load a trained model and config that you have fine-tuned
        if args.mode == 'vn' : 
            model = BertForClassification.from_pretrained(args.bert_model, num_labels=3).to(device)
        elif args.mode == 'vn_bin' : 
            model = BertForBinaryClassification.from_pretrained(args.bert_model).to(device)
        else : 
            model = BertForPOE.from_pretrained(args.bert_model, num_labels=3).to(device)
        model.load_state_dict(torch.load(output_model_file))

    else:
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        if args.mode == 'vn' : 
            model = BertForClassification.from_pretrained(args.bert_model, num_labels=3).to(device)
        elif args.mode == 'vn_bin' : 
            model = BertForBinaryClassification.from_pretrained(args.bert_model).to(device)
        else : 
            model = BertForPOE.from_pretrained(args.bert_model, num_labels=3).to(device)
        model.load_state_dict(torch.load(output_model_file))
    model.eval()

    if args.do_eval:
        eval_datasets = [("mnli_dev_m", load_mnli(False)),
                         ("mnli_dev_mm", load_mnli(False, custom_path="dev_mismatched.tsv"))]
        #eval_datasets += load_easy_hard(prefix="overlap_", no_mismatched=True)
        #eval_datasets += load_easy_hard()
        if args.mode == 'poe':
            eval_datasets += [("hans", load_hans())]
            #eval_datasets += [("wanli", load_wanli(False, '../dataset/wanli'))]
            eval_datasets += load_hans_subsets()
        if args.do_eval_on_train:
            eval_datasets = [("mnli_train", load_mnli(True))]
    else:
        eval_datasets = []

    for ix, (name, eval_examples) in enumerate(eval_datasets):
        logging.info("***** Running evaluation on %s *****" % name)
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", args.eval_batch_size)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, tokenizer)
        eval_features.sort(key=lambda x: len(x.input_ids))
        all_label_ids = np.array([x.label_id for x in eval_features])

        for fe in eval_features:
            fe.teacher_probs = None
        eval_dataloader = build_eval_dataloader(eval_features, args.eval_batch_size)

        probs = []
        if 'bin' in args.mode:
            acc,_ = compute_acc_bin(eval_dataloader, model, device, all_label_ids, args.label_given)
        else:
            acc,_ = compute_acc(eval_dataloader, model, device, all_label_ids)

        result = {"acc": acc}

        logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))

        if args.do_eval_on_train:
            output_answer_file = os.path.join(output_dir, "eval_%s_answers.json" % name)
            if args.mode in ['vn_bin']:
                answers = {ex.example_id: [float(p)] for ex, p in zip(eval_features, probs)}
            else:
                answers = {ex.example_id: [float(x) for x in p] for ex, p in zip(eval_features, probs)}
            with open(output_answer_file, "w") as f:
                json.dump(answers, f)

if __name__ == "__main__" :
    main()