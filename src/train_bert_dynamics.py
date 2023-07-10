"""
Script to train BERT on MNLI with our loss function

Modified from the old "run_classifier" script from
https://github.com/huggingface/pytorch-transformer
"""

from collections import namedtuple
import os
from os.path import join, exists
import random
from typing import List, Dict, Iterable
import config
import numpy as np
import logging
import math
import argparse
from transformers import BertTokenizer, get_linear_schedule_with_warmup, WEIGHTS_NAME, CONFIG_NAME
import torch
from utils import Processor, process_par
import utils
import csv
import json
from tqdm import trange, tqdm
from predictions_analysis import visualize_predictions
from bert_poe import BertForClassification
from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler, SequentialSampler

HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"

NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}
REV_NLI_LABEL_MAP = {i: k for i, k in enumerate(NLI_LABELS)}
NLI_LABEL_MAP["hidden"] = NLI_LABEL_MAP["entailment"]
FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
FEVER_LABEL_MAP = {k: i for i, k in enumerate(FEVER_LABELS)}
QQP_SOURCE = '../dataset/QQP'

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

def load_fever(is_train, seed=111, custom_path=None, sample=None):
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
            id = str(i)
            text_a = example["claim"]
            text_b = example["evidence"] if "evidence" in example.keys() else example["evidence_sentence"]
            label = example["gold_label"] if "gold_label" in example.keys() else example["label"]
            out.append(TextPairExample(id, text_a, text_b, FEVER_LABEL_MAP[label]))
    if sample:
        random.seed(seed)
        random.shuffle(out)
        out = out[:sample]
    return out

def load_qqp(is_train, seed=111, sample=None, custom_path=None) -> List[TextPairExample]:
    fieldnames = ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"]
    if is_train:
        filename = os.path.join(QQP_SOURCE, "qqp_train.csv")
    else:
        if custom_path is None:
            filename = os.path.join(QQP_SOURCE, "qqp_val.csv")
        else:
            filename = os.path.join(QQP_SOURCE, custom_path)

    out = []
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            if line_count == 1: continue
            out.append(TextPairExample(row['id'], row['question1'], row['question2'], int(row['is_duplicate'])))

    if sample:
        random.seed(seed)
        random.shuffle(out)
        out = out[:sample]
    return out

def load_qqp_paws(is_train, custom_path=None) -> List[TextPairExample]:
    if is_train:
        filename = '../dataset/PAWS/train.tsv'
    else:
        if custom_path is None:
            filename = '../dataset/PAWS/dev_and_test.tsv'
        else:
            filename = custom_path

    with open(filename) as f:
        f.readline()
        lines = f.readlines()

    out = []
    for line in lines:
        line = line.split("\t")
        out.append(TextPairExample(line[0], eval(line[1]).decode('utf-8'), eval(line[2]).decode('utf-8'), int(line[3])))
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

def trace_bert_dynamics(model, features, dataloader, device, answers):
    probs = []
    all_label_ids = np.array([x.label_id for x in features])
    num_labels = len(set(list(all_label_ids)))
    for _, input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Evaluating Training Dynamics", ncols=100):                                                                
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

        predictions = torch.nn.functional.softmax(logits.view(-1,num_labels), 1)
        #probs.append(predictions.detach().cpu().numpy())
        probs.append(torch.gather(predictions, dim=1, index=label_ids.unsqueeze(1)).view(-1).detach().cpu().numpy())

    probs = np.hstack(probs)
    #probs = np.concatenate(probs, 0)
    #preds = np.argmax(probs, axis=1)
    answers_new = {ex.example_id: answers[ex.example_id] + [float(p)] for ex, p in zip(features, probs)}
    #answers_new = {ex.example_id: answers[ex.example_id] + [(float(p[label]),int(pred==label))] for ex,p,pred,label in zip(features,probs,preds,all_label_ids)}
    return answers_new

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
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
    parser.add_argument("--mode", type=str, required=True, choices=['vn', 'cf'],
                        help='vn : vanilla BERT, cf : Counterfactual mode') 
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help='weight decay')
    parser.add_argument("--logging_steps", default=-1, type=int,
                        help="Step for logging.")
    parser.add_argument("--train_data", type=str, required=True, choices=['MNLI','FEVER','QQP','SNLI'],
                        help='data to train on')  
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
        if args.train_data == 'MNLI' :
            train_examples, eval_examples_4_dynamics = load_mnli(True), load_mnli(True)
        elif args.train_data == 'FEVER' :
            train_examples, eval_examples_4_dynamics = load_fever(True), load_fever(True)
        elif args.train_data == 'QQP':
            train_examples, eval_examples_4_dynamics = load_qqp(True), load_qqp(True)
        elif args.train_data == 'SNLI':
            train_examples, eval_examples_4_dynamics = load_snli("train"), load_snli("train")
        num_train_optimization_steps = int(math.ceil(len(train_examples) / args.train_batch_size) / 
        args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Model Preparation
    num_labels = 2 if args.train_data == 'QQP' else 3
    if args.mode == 'vn' : 
        model = BertForClassification.from_pretrained(args.bert_model, num_labels=num_labels).to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    if args.do_train:
        warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(num_train_optimization_steps * args.warmup_proportion)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        train_features: List[InputFeatures] = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer, args.n_processes)
        eval_features_4_dynamics: List[InputFeatures] = convert_examples_to_features(eval_examples_4_dynamics, args.max_seq_length, tokenizer,
                                                                                     args.n_processes)

        example_map = {}
        for ex in train_examples:
            example_map[ex.id] = ex

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = build_train_dataloader(train_features, args.train_batch_size, args.seed, args.sorted)
        eval_dataloader_4_dynamics = build_eval_dataloader(eval_features_4_dynamics, args.eval_batch_size)

        model.train()
        loss_ema = 0
        total_steps = 0
        decay = 0.99
        answers_dynamics = {ex.example_id: [] for ex in eval_features_4_dynamics}
        dynamics_dir = './dynamics'
        output_answer_dynamics_file = os.path.join(dynamics_dir, f"training_dynamics_{int(args.num_train_epochs)}_{args.train_data}.json")

        for _ in trange(int(args.num_train_epochs), desc="Epoch", ncols=100):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            pbar = tqdm(train_dataloader, desc="loss", ncols=100)
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)
                example_ids, input_ids, mask, segment_ids, label_ids = batch
                inputs = {
                        "input_ids": input_ids,
                        "attention_mask": mask,
                        "token_type_ids": segment_ids,
                        "labels": label_ids,
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
                        model.eval()
                        answers_dynamics = trace_bert_dynamics(model, eval_features_4_dynamics, eval_dataloader_4_dynamics, device, answers_dynamics)
                        with open(output_answer_dynamics_file, "w") as f:
                            json.dump(answers_dynamics, f)
                        model.train()
                        tr_loss = 0

        # Save model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Save training dynamics
        with open(output_answer_dynamics_file, "w") as f:
            json.dump(answers_dynamics, f)

        # Record the args as well
        arg_dict = {}
        for arg in vars(args):
            arg_dict[arg] = getattr(args, arg)
        with open(join(output_dir, "args.json"), 'w') as out_fh:
            json.dump(arg_dict, out_fh)

        # Load a trained model and config that you have fine-tuned
        if args.mode == 'vn' : 
            model = BertForClassification.from_pretrained(args.bert_model, num_labels=num_labels).to(device)
        model.load_state_dict(torch.load(output_model_file))

    else:
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        if args.mode == 'vn' : 
            model = BertForClassification.from_pretrained(args.bert_model, num_labels=num_labels).to(device)
        model.load_state_dict(torch.load(output_model_file))
                                                
    if not args.do_eval and not args.do_test:
        return
    if not (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        return

    model.eval()

    if args.do_eval:
        if args.train_data == 'MNLI' :
            eval_datasets = [("mnli_dev_m", load_mnli(False))]
            #eval_datasets += load_easy_hard()
            eval_datasets += [("hans", load_hans())]
            #eval_datasets += [("wanli", load_wanli(False, '../dataset/wanli'))]
            eval_datasets += load_hans_subsets()
            if args.do_eval_on_train:
                eval_datasets = [("mnli_train", load_mnli(True))]
        elif args.train_data == 'FEVER' :
            eval_datasets = [("fever_dev", load_fever(False))]
            eval_datasets += [("fever_symmetric_dev_v1", load_fever(False,custom_path='../dataset/FEVER-symmetric-generated/nli.dev.jsonl'))]
            eval_datasets += [("fever_symmetric_dev_v2", load_fever(False,custom_path='../dataset/FEVER-symmetric-generated/fever_symmetric_dev.jsonl'))]
            eval_datasets += [("fever_symmetric_test_v2", load_fever(False,custom_path='../dataset/FEVER-symmetric-generated/fever_symmetric_test.jsonl'))]
            if args.do_eval_on_train:
                eval_datasets = [("fever_train", load_fever(True))]
        elif args.train_data == 'QQP':
            eval_datasets = [("qqp_dev", load_qqp(False))]
            eval_datasets += [("qqp_paws", load_qqp_paws(False))]
            if args.do_eval_on_train:
                eval_datasets = [("qqp_train", load_qqp(True))]
        elif args.train_data == 'SNLI':
            eval_datasets = [("snli_dev", load_snli("dev"))]
            eval_datasets += [("snli_test", load_snli("test"))]
            eval_datasets += [("snli_test_hard", load_jsonl("snli_1.0_test_hard.jsonl", "../dataset/SNLI"))]
            if args.do_eval_on_train:
                eval_datasets = [("snli_train", load_snli("train"))]
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
        eval_dataloader = build_eval_dataloader(eval_features, args.eval_batch_size)

        eval_loss = 0
        nb_eval_steps = 0
        probs = []
        test_subm_ids = []

        for example_ids, input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                               desc="Evaluating",
                                                                               ncols=100):
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

            # create eval loss and other metric required by the task
            loss_fct = torch.nn.CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            probs.append(torch.nn.functional.softmax(logits, 1).detach().cpu().numpy())
            test_subm_ids.append(example_ids.cpu().numpy())

        probs = np.concatenate(probs, 0)
        test_subm_ids = np.concatenate(test_subm_ids, 0)
        eval_loss = eval_loss / nb_eval_steps

        if "hans" in name:
            # take max of non-entailment rather than taking their sum
            probs[:, 0] = probs[:, [0, 2]].max(axis=1)
            # probs[:, 0] = probs[:, 0] + probs[:, 2]
            probs = probs[:, :2]

        preds = np.argmax(probs, axis=1)

        result = {"acc": simple_accuracy(preds, all_label_ids)}
        result["loss"] = eval_loss

        '''
        conf_plot_file = os.path.join(output_dir, "eval_%s_confidence.png" % name)
        ECE, bins_acc, bins_conf, bins_num = visualize_predictions(probs, all_label_ids, conf_plot_file=conf_plot_file)
        result["ECE"] = ECE
        result["bins_acc"] = bins_acc
        result["bins_conf"] = bins_conf
        result["bins_num"] = bins_num
        '''

        logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))


if __name__ == "__main__" :
    main()