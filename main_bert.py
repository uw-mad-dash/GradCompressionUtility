import os
import sys
import time
import numpy as np
import argparse
import logging
import json
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from collections import defaultdict
import torchvision.models as models
from torch.autograd import Variable

import gradient_reducers
import s3_utils
from timer import Timer

# pasting from hugging face code
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pandas as pd
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as PowerSGD

import tokenization
from modeling import BertConfig, BertForSequenceClassification

def metric(*args, **kwargs):
    if True == 0:
        log_metric(*args, **kwargs)

timer = Timer(verbosity_level=2, log_fn=metric)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class Sogou_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir, data_num=None):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep="\t", encoding='utf-8-sig').values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep="\t",encoding='utf-8-sig').values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4","5","6"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i > 1199: break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1])+" "+str(line[2]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class AGNewsProcessor(DataProcessor):
    """Processor for the AG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None).values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):	
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1]+" - "+line[2])
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AGNewsProcessor_sep(DataProcessor):
    """Processor for the AG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None).values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):	
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class AGNewsProcessor_sep_aug(DataProcessor):
    """Processor for the AG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None).values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        if set_type=="train":
            scale=5
        else:
            scale=1
        examples = []
        for (i, line) in enumerate(lines):
            s=(line[1]+" - "+line[2]).split()
            l=len(s)
            for j in range(scale):
                r=random.randint(1,l-1)
                guid = "%s-%s" % (set_type, i*scale+j)
                text_a = tokenization.convert_to_unicode(" ".join(s[:r]))
                text_b = tokenization.convert_to_unicode(" ".join(s[r:]))
                label = tokenization.convert_to_unicode(str(line[0]))
                if i%1000==0:
                    print(i)
                    print("guid=",guid)
                    print("text_a=",text_a)
                    print("text_b=",text_b)
                    print("label=",label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class IMDBProcessor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["0","1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i>1000:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class IMDBProcessor_sep(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["0","1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        scale=1
        examples = []
        for (i, line) in enumerate(lines):
            s=(line[1]).split()
            l=len(s)
            for j in range(scale):
                r=random.randint(1,l-1)
                guid = "%s-%s" % (set_type, i*scale+j)
                text_a = tokenization.convert_to_unicode(" ".join(s[:r]))
                text_b = tokenization.convert_to_unicode(" ".join(s[r:]))
                label = tokenization.convert_to_unicode(str(line[0]))
                if i%1000==0:
                    print(i)
                    print("guid=",guid)
                    print("text_a=",text_a)
                    print("text_b=",text_b)
                    print("label=",label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class IMDBProcessor_sep_aug(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["0","1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        if set_type=="train":
            scale=5
        else:
            scale=1
        examples = []
        for (i, line) in enumerate(lines):
            if i==100 and set_type=="train":break
            if i==1009 and set_type=="dev":break
            s=(line[1]).split()
            l=len(s)
            for j in range(scale):
                r=random.randint(1,l-1)
                guid = "%s-%s" % (set_type, i*scale+j)
                text_a = tokenization.convert_to_unicode(" ".join(s[:r]))
                text_b = tokenization.convert_to_unicode(" ".join(s[r:]))
                label = tokenization.convert_to_unicode(str(line[0]))
                if i%1000==0:
                    print(i)
                    print("guid=",guid)
                    print("text_a=",text_a)
                    print("text_b=",text_b)
                    print("label=",label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Yelp_p_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep=",").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep=",").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Yelp_f_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep=",").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep=",").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4","5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

        
class Yahoo_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep=",").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep=",").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1","2","3","4","5","6","7","8","9","10"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1])+" "+str(line[2]))
            text_b = tokenization.convert_to_unicode(str(line[3]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class Trec_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ['LOC', 'NUM', 'HUM', 'ENTY', 'ABBR', 'DESC']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Dbpedia_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"),header=None,sep=",").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"),header=None,sep=",").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1])+" "+str(line[2]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    return features


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

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)


def parse_args(parser):
    # parser.add_argument("--arch", default="resnet50", type=str,
                        # help="network type")
    # parser.add_argument("--master-ip", type=str, help="Ip address of master")
    parser.add_argument("--local_rank", type=int, help="Rank of the experiment")
    parser.add_argument("--batch-size", type=int, help="Batch size to use")
    parser.add_argument("--dataset-location", type=str, help="Data path")
    parser.add_argument("--loader-threads", type=int, default=2, help="Loader threads")
    # parser.add_argument("--device", type=str, default="cuda:0", 
                        # help="GPU to use")
    parser.add_argument("--log-file", type=str, default="Log file")
    parser.add_argument("--num-workers", type=int, 
                        help="Number of total  workers")
    parser.add_argument("--s3-prefix", type=str, default=None, 
                        help="s3-prefix to write")
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--max_seq_length", type=int)
    args = parser.parse_args()
    return args


def _get_compression_param(reducer_name, device, reducer_param):
    if reducer_name == "PowerSGD":
        reducer = gradient_reducers.RankKReducer(random_seed=42,
                                                  device=device,
                                                  timer=timer,
                                                  n_power_iterations=0,
                                                  reuse_query=True,
                                                  rank = reducer_param)

    if reducer_name == "SignSGD":
        reducer = gradient_reducers.SignSGDwithMajorityVoteReducer(random_seed=42,
                                                 device=device,
                                                 timer=timer)
    if reducer_name == "Topk":
        reducer = gradient_reducers.GlobalTopKReducer(random_seed=42,
                                                      device=device,
                                                      timer=timer,
                                                      compression=reducer_param)
    if reducer_name == "MSTopK":
        reducer = gradient_reducers.MsTopKReducer(random_seed=42,
                                                  device=device, timer=timer,
                                                  k=reducer_param)
    if reducer_name == "ExactSerial":
        reducer = gradient_reducers.ExactReducer(random_seed=42, device=device,
                                                 timer=timer)
    
    if reducer_name == "ExactSerial_ps":
        reducer = gradient_reducers.ExactReducerPs(random_seed=42, device=device,
                                                 timer=timer)
    return reducer


def main_bert(args):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 8 + args.local_rank

    data_dir = "/home/ubuntu/bert_data/Sogou_data"
    bert_config_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/bert_config.json"
    task_name = "sogou"
    vocab_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length
    train_batch_size = args.batch_size
    learning_rate = 1e-5
    bert_config = BertConfig.from_json_file(bert_config_file)
    processor = Sogou_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
    train_examples = processor.get_train_examples(data_dir)
    model = BertForSequenceClassification(bert_config,
                                          len(label_list)).to(assigned_device)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


    state = [parameter for parameter in model.parameters()] 

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_dataloader = DataLoader(train_data, train_batch_size)
    model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=45,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()
    for idx, data in enumerate(train_dataloader):
        batch = tuple(t.to(assigned_device) for t in data)
        input_ids, input_mask, segment_ids, label_ids = batch
        (loss, _) = model(input_ids, segment_ids, input_mask, label_ids)
        torch.cuda.synchronize()
        start_time.record()
        loss.backward()
        stop_time.record()
        torch.cuda.synchronize()
        time_list.append(start_time.elapsed_time(stop_time))
        if idx == 60:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "bert_ddp_out_file_{}.json".format(global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name, 
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done bert")
            break

def main_bert_single(args):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 8 + args.local_rank

    data_dir = "/home/ubuntu/bert_data/Sogou_data"
    bert_config_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/bert_config.json"
    task_name = "sogou"
    vocab_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length
    train_batch_size = args.batch_size
    learning_rate = 1e-5
    bert_config = BertConfig.from_json_file(bert_config_file)
    processor = Sogou_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
    train_examples = processor.get_train_examples(data_dir)
    model = BertForSequenceClassification(bert_config,
                                          len(label_list)).to(assigned_device)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


    state = [parameter for parameter in model.parameters()] 

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_dataloader = DataLoader(train_data, train_batch_size)
    # model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=45,
                                                      # device_ids=[args.local_rank],
                                                      # output_device=args.local_rank)
    
    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()
    for idx, data in enumerate(train_dataloader):
        batch = tuple(t.to(assigned_device) for t in data)
        input_ids, input_mask, segment_ids, label_ids = batch
        (loss, _) = model(input_ids, segment_ids, input_mask, label_ids)
        torch.cuda.synchronize()
        start_time.record()
        loss.backward()
        stop_time.record()
        torch.cuda.synchronize()
        time_list.append(start_time.elapsed_time(stop_time))
        print (time_list)
        if idx == 60:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "bert_ddp_out_file_{}.json".format(global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name, 
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done bert")
            break


def powersgd_bert(args, psgd_rank):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 8 + args.local_rank

    data_dir = "/home/ubuntu/bert_data/Sogou_data"
    bert_config_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/bert_config.json"
    task_name = "sogou"
    vocab_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length
    train_batch_size = args.batch_size
    learning_rate = 1e-5
    bert_config = BertConfig.from_json_file(bert_config_file)
    processor = Sogou_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
    train_examples = processor.get_train_examples(data_dir)
    model = BertForSequenceClassification(bert_config,
                                          len(label_list)).to(assigned_device)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


    state = [parameter for parameter in model.parameters()] 

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_dataloader = DataLoader(train_data, train_batch_size)

    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()

    reducer = _get_compression_param("PowerSGD", assigned_device, psgd_rank)

    memories = [torch.zeros_like(p) for p in model.parameters()]
    send_buffers = [torch.zeros_like(p) for p in model.parameters()]

    for idx, data in enumerate(train_dataloader):
        batch = tuple(t.to(assigned_device) for t in data)
        input_ids, input_mask, segment_ids, label_ids = batch
        (loss, _) = model(input_ids, segment_ids, input_mask, label_ids)
        torch.cuda.synchronize()
        start_time.record()
        loss.backward()
        grad_list = [parameter.grad for parameter in model.parameters()]

        for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            send_bfr.data[:] = grad + memory
        reducer.reduce(send_buffers, grad_list, memories)

        stop_time.record()
        torch.cuda.synchronize()

        time_list.append(start_time.elapsed_time(stop_time))
        if idx == 60:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "bert_powersgd_serial_rank_{}_ddp_out_file_{}.json".format(
                psgd_rank,global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name, 
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done bert")
            break

def powersgd_bert_integrated(args, psgd_rank):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 8 + args.local_rank

    data_dir = "/home/ubuntu/bert_data/Sogou_data"
    bert_config_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/bert_config.json"
    task_name = "sogou"
    vocab_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length
    train_batch_size = args.batch_size
    learning_rate = 1e-5
    bert_config = BertConfig.from_json_file(bert_config_file)
    processor = Sogou_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
    train_examples = processor.get_train_examples(data_dir)
    model = BertForSequenceClassification(bert_config,
                                          len(label_list)).to(assigned_device)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


    state = [parameter for parameter in model.parameters()] 

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_dataloader = DataLoader(train_data, train_batch_size)

    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()

    # reducer = _get_compression_param("PowerSGD", assigned_device, psgd_rank)

    # memories = [torch.zeros_like(p) for p in model.parameters()]
    # send_buffers = [torch.zeros_like(p) for p in model.parameters()]

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    state = PowerSGD.PowerSGDState(process_group=None,
                                    matrix_approximation_rank=psgd_rank,
                                    start_powerSGD_iter=3)
    
    model.register_comm_hook(state, PowerSGD.powerSGD_hook) 

    for idx, data in enumerate(train_dataloader):
        batch = tuple(t.to(assigned_device) for t in data)
        input_ids, input_mask, segment_ids, label_ids = batch
        (loss, _) = model(input_ids, segment_ids, input_mask, label_ids)
        torch.cuda.synchronize()
        start_time.record()
        loss.backward()
        # grad_list = [parameter.grad for parameter in model.parameters()]

        # for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            # send_bfr.data[:] = grad + memory
        # reducer.reduce(send_buffers, grad_list, memories)

        stop_time.record()
        torch.cuda.synchronize()

        time_list.append(start_time.elapsed_time(stop_time))
        if idx == 60:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "bert_powersgd_overlap_rank_{}_ddp_out_file_{}.json".format(
                psgd_rank,global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name, 
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done bert")
            break
def topk_bert(args, psgd_rank):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 8 + args.local_rank

    data_dir = "/home/ubuntu/bert_data/Sogou_data"
    bert_config_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/bert_config.json"
    task_name = "sogou"
    vocab_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length
    train_batch_size = args.batch_size
    learning_rate = 1e-5
    bert_config = BertConfig.from_json_file(bert_config_file)
    processor = Sogou_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
    train_examples = processor.get_train_examples(data_dir)
    model = BertForSequenceClassification(bert_config,
                                          len(label_list)).to(assigned_device)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


    state = [parameter for parameter in model.parameters()] 

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_dataloader = DataLoader(train_data, train_batch_size)

    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()

    reducer = _get_compression_param("Topk", assigned_device, psgd_rank)

    memories = [torch.zeros_like(p) for p in model.parameters()]
    send_buffers = [torch.zeros_like(p) for p in model.parameters()]

    for idx, data in enumerate(train_dataloader):
        batch = tuple(t.to(assigned_device) for t in data)
        input_ids, input_mask, segment_ids, label_ids = batch
        (loss, _) = model(input_ids, segment_ids, input_mask, label_ids)
        torch.cuda.synchronize()
        start_time.record()
        loss.backward()
        grad_list = [parameter.grad for parameter in model.parameters()]

        for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            send_bfr.data[:] = grad + memory
        reducer.reduce(send_buffers, grad_list, memories)

        stop_time.record()
        torch.cuda.synchronize()

        time_list.append(start_time.elapsed_time(stop_time))
        if idx == 7:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "bert_topk_{}_out_file_{}.json".format(
                psgd_rank,global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name, 
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done bert")
            break

def encode_decode(state, bucket):
    # tensors = [ t/dist.world_size for t in bucket.get_tensors()]
            
    # print (state)
    tensor = bucket.get_tensors()[0]
    k = int(state['k']*len(tensor))
    N = state['N']
    grad_1d = tensor 
    # grad_1d = grad_in.reshape(-1) #reshaping to 1d
    a = torch.abs(grad_1d)
    a_hat = torch.mean(a)
    u = torch.max(a)
    l = 0
    r = 1
    k1 = 0
    k2 = len(grad_1d)
    thres1 = 0
    thres2 = 0
    for i in range(N):
        ratio = l + (r-l)/2
        thres = a_hat + ratio*(u-a_hat)
        nnz = torch.count_nonzero(a >= thres)
        if nnz <= k:
            r = ratio
            if nnz > k1:
                k1 = nnz
                thres1 = thres
        elif nnz > k:
            l= ratio
            if nnz < k2:
                k2 = nnz
                thres2 = thres
    l1 = torch.nonzero(a>= thres1, as_tuple=True)[0] #since 1d no problem
    l2 = torch.nonzero((a<thres1) & (a >= thres2), as_tuple=True)[0]
    if len(l2)-(k-k1)+1 < 0:
        l = torch.cat((l1, l2[0:k-len(l1)]))
    else:
        rand = random.randint(0, len(l2)-(k-k1)+1)
        l = torch.cat((l1, l2[rand:rand+k-k1]))
    kai = tensor[l]




    # tensor = torch.ones_like(tensor, device=tensor.device, dtype=tensor.dtype)
    group_to_use = dist.group.WORLD
    world_size = group_to_use.size()
    
    out_list = [torch.zeros_like(kai, device=kai.device,
                dtype=kai.dtype) for _ in range(world_size)]

    idx_list = [torch.zeros_like(l, device=l.device,
                dtype=l.dtype) for _ in range(world_size)]

    dist.all_gather(idx_list, l, group=group_to_use,
                    async_op=True)

    fut = dist.all_gather(
        out_list, kai, group=group_to_use, async_op=True).get_future()

    def decode(fut):
        agg_tensor = fut.value()[0]
        fut_tensor = bucket.get_tensors()[0]
        out_tensor = torch.zeros_like(fut_tensor, device=tensor.device,
                                      dtype=tensor.dtype)
        for gt in agg_tensor:
            out_tensor[:len(gt)] += gt
        # print (out_tensor) 
        return [out_tensor]
    return fut.then(decode)


def topk_bert_single_call(args, topk_k):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 8 + args.local_rank

    data_dir = "/home/ubuntu/bert_data/Sogou_data"
    bert_config_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/bert_config.json"
    task_name = "sogou"
    vocab_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length
    train_batch_size = args.batch_size
    learning_rate = 1e-5
    bert_config = BertConfig.from_json_file(bert_config_file)
    processor = Sogou_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
    train_examples = processor.get_train_examples(data_dir)
    model = BertForSequenceClassification(bert_config,
                                          len(label_list)).to(assigned_device)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


    state = [parameter for parameter in model.parameters()] 

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_dataloader = DataLoader(train_data, train_batch_size)

    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    model.register_comm_hook(state={'N':100, 'k':topk_k}, hook=encode_decode)
    # reducer = _get_compression_param("Topk", assigned_device, psgd_rank)

    # memories = [torch.zeros_like(p) for p in model.parameters()]
    # send_buffers = [torch.zeros_like(p) for p in model.parameters()]

    for idx, data in enumerate(train_dataloader):
        batch = tuple(t.to(assigned_device) for t in data)
        input_ids, input_mask, segment_ids, label_ids = batch
        (loss, _) = model(input_ids, segment_ids, input_mask, label_ids)
        torch.cuda.synchronize()
        start_time.record()
        loss.backward()
        # grad_list = [parameter.grad for parameter in model.parameters()]

        # for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            # send_bfr.data[:] = grad + memory
        # reducer.reduce(send_buffers, grad_list, memories)

        stop_time.record()
        torch.cuda.synchronize()

        time_list.append(start_time.elapsed_time(stop_time))
        if idx == 60:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "bert_mstopk_overlap_{}_out_file_{}.json".format(
                topk_k,global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name, 
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done bert")
            break

def mstopk_serial(args, topk_k):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 8 + args.local_rank

    data_dir = "/home/ubuntu/bert_data/Sogou_data"
    bert_config_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/bert_config.json"
    task_name = "sogou"
    vocab_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length
    train_batch_size = args.batch_size
    learning_rate = 1e-5
    bert_config = BertConfig.from_json_file(bert_config_file)
    processor = Sogou_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
    train_examples = processor.get_train_examples(data_dir)
    model = BertForSequenceClassification(bert_config,
                                          len(label_list)).to(assigned_device)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


    state = [parameter for parameter in model.parameters()] 

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_dataloader = DataLoader(train_data, train_batch_size)

    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()

    # model = torch.nn.parallel.DistributedDataParallel(model,
                                                      # device_ids=[args.local_rank],
                                                      # output_device=args.local_rank)
    # model.register_comm_hook(state={'N':100, 'k':topk_k}, hook=encode_decode)
    # reducer = _get_compression_param("Topk", assigned_device, psgd_rank)

    # memories = [torch.zeros_like(p) for p in model.parameters()]
    send_buffers = [torch.zeros_like(p) for p in model.parameters()]
    reducer = _get_compression_param("MSTopK", assigned_device, topk_k) 
    model.train()

    for idx, data in enumerate(train_dataloader):
        batch = tuple(t.to(assigned_device) for t in data)
        input_ids, input_mask, segment_ids, label_ids = batch
        (loss, _) = model(input_ids, segment_ids, input_mask, label_ids)
        torch.cuda.synchronize()
        start_time.record()
        loss.backward()
        grad_list = [parameter.grad for parameter in model.parameters()]

        # for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            # send_bfr.data[:] = grad + memory
        reducer.reduce(send_buffers, grad_list)

        stop_time.record()
        torch.cuda.synchronize()

        time_list.append(start_time.elapsed_time(stop_time))
        if idx == 60:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "bert_mstopk_serial_{}_out_file_{}.json".format(
                topk_k,global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name, 
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done bert")
            break


def encode_decode_signsgd(state, bucket):
    """
    signsgd in parallel
    """
    sign_compressor = gradient_reducers.SignCompressor()
    # tensor_flat = TensorBuffer(bucket)
    bits, sign_size = sign_compressor.compress(bucket.get_tensors()[0])
    copy_bits = [torch.empty_like(bits) for i in range(dist.get_world_size())]

    fut = dist.all_gather(copy_bits, bits, group=dist.group.WORLD,
                          async_op=True).get_future()
    def decode(fut):
        sum_of_signs = None
        agg_tensor = fut.value()[0]
        for their_bits in agg_tensor:
            uncompressed = sign_compressor.uncompress(their_bits, sign_size)
            if sum_of_signs is None:
                sum_of_signs = uncompressed
            else:
                sum_of_signs += uncompressed
        total_sign = sum_of_signs.sign()
        return [total_sign]
    return fut.then(decode)


def signsgd_bert_single_call(args):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 8 + args.local_rank

    data_dir = "/home/ubuntu/bert_data/Sogou_data"
    bert_config_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/bert_config.json"
    task_name = "sogou"
    vocab_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length
    train_batch_size = args.batch_size
    learning_rate = 1e-5
    bert_config = BertConfig.from_json_file(bert_config_file)
    processor = Sogou_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
    train_examples = processor.get_train_examples(data_dir)
    model = BertForSequenceClassification(bert_config,
                                          len(label_list)).to(assigned_device)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


    state = [parameter for parameter in model.parameters()] 

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_dataloader = DataLoader(train_data, train_batch_size)

    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    model.register_comm_hook(state=None, hook=encode_decode_signsgd)
    # reducer = _get_compression_param("Topk", assigned_device, psgd_rank)

    # memories = [torch.zeros_like(p) for p in model.parameters()]
    # send_buffers = [torch.zeros_like(p) for p in model.parameters()]

    for idx, data in enumerate(train_dataloader):
        batch = tuple(t.to(assigned_device) for t in data)
        input_ids, input_mask, segment_ids, label_ids = batch
        (loss, _) = model(input_ids, segment_ids, input_mask, label_ids)
        torch.cuda.synchronize()
        start_time.record()
        loss.backward()
        # grad_list = [parameter.grad for parameter in model.parameters()]

        # for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            # send_bfr.data[:] = grad + memory
        # reducer.reduce(send_buffers, grad_list, memories)

        stop_time.record()
        torch.cuda.synchronize()

        time_list.append(start_time.elapsed_time(stop_time))
        if idx == 60:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "bert_signsgd_overlap_out_file_{}.json".format(global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name, 
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done bert")
            break


def signsgd_bert(args):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 8 + args.local_rank

    data_dir = "/home/ubuntu/bert_data/Sogou_data"
    bert_config_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/bert_config.json"
    task_name = "sogou"
    vocab_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length
    train_batch_size = args.batch_size
    learning_rate = 1e-5
    bert_config = BertConfig.from_json_file(bert_config_file)
    processor = Sogou_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
    train_examples = processor.get_train_examples(data_dir)
    model = BertForSequenceClassification(bert_config,
                                          len(label_list)).to(assigned_device)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


    state = [parameter for parameter in model.parameters()] 

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_dataloader = DataLoader(train_data, train_batch_size)

    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()

    reducer = _get_compression_param("SignSGD", assigned_device, None)

    memories = [torch.zeros_like(p) for p in model.parameters()]
    send_buffers = [torch.zeros_like(p) for p in model.parameters()]

    for idx, data in enumerate(train_dataloader):
        batch = tuple(t.to(assigned_device) for t in data)
        input_ids, input_mask, segment_ids, label_ids = batch
        (loss, _) = model(input_ids, segment_ids, input_mask, label_ids)
        torch.cuda.synchronize()
        start_time.record()
        loss.backward()
        grad_list = [parameter.grad for parameter in model.parameters()]

        for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            send_bfr.data[:] = grad + memory
        reducer.reduce(send_buffers, grad_list, memories)

        stop_time.record()
        torch.cuda.synchronize()

        time_list.append(start_time.elapsed_time(stop_time))
        if idx == 60:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['timing_log'] = time_list
            file_name = "bert_signsgd_serial_out_file_{}.json".format(global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name, 
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done bert")
            break

def fullcomm_serial(args):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 8 + args.local_rank

    data_dir = "/home/ubuntu/bert_data/Sogou_data"
    bert_config_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/bert_config.json"
    task_name = "sogou"
    vocab_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length
    train_batch_size = args.batch_size
    learning_rate = 1e-5
    bert_config = BertConfig.from_json_file(bert_config_file)
    processor = Sogou_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
    train_examples = processor.get_train_examples(data_dir)
    model = BertForSequenceClassification(bert_config,
                                          len(label_list)).to(assigned_device)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


    state = [parameter for parameter in model.parameters()] 

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_dataloader = DataLoader(train_data, train_batch_size)

    start_time_backward = torch.cuda.Event(enable_timing=True)
    stop_time_backward = torch.cuda.Event(enable_timing=True)

    start_time_comm = torch.cuda.Event(enable_timing=True)
    stop_time_comm = torch.cuda.Event(enable_timing=True)
    
    time_comm_list = list()
    time_backward_list = list() 

    reducer = _get_compression_param("ExactSerial", assigned_device, None)

    memories = [torch.zeros_like(p) for p in model.parameters()]
    send_buffers = [torch.zeros_like(p) for p in model.parameters()]

    for idx, data in enumerate(train_dataloader):
        batch = tuple(t.to(assigned_device) for t in data)
        input_ids, input_mask, segment_ids, label_ids = batch
        (loss, _) = model(input_ids, segment_ids, input_mask, label_ids)
        torch.cuda.synchronize()
        start_time_backward.record()
        loss.backward()
        stop_time_backward.record()
        
        start_time_comm.record()
        grad_list = [parameter.grad for parameter in model.parameters()]

        for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            send_bfr.data[:] = grad + memory
        reducer.reduce(send_buffers, grad_list, memories)

        stop_time_comm.record()
        torch.cuda.synchronize()

        time_comm_list.append(start_time_comm.elapsed_time(stop_time_comm))

        time_backward_list.append(start_time_backward.elapsed_time(stop_time_backward))

        if idx == 60:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['comm_timing_log'] = time_comm_list
            data_dict['backward_timing_log'] = time_backward_list
            file_name = "bert_serial_full_out_file_{}.json".format(global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name, 
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done bert")
            break

def fullcomm_serial_ps(args):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 8 + args.local_rank

    data_dir = "/home/ubuntu/bert_data/Sogou_data"
    bert_config_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/bert_config.json"
    task_name = "sogou"
    vocab_file = "/home/ubuntu/bert_data/chinese_L-12_H-768_A-12/vocab.txt"
    do_lower_case = True
    max_seq_length = args.max_seq_length
    train_batch_size = args.batch_size
    learning_rate = 1e-5
    bert_config = BertConfig.from_json_file(bert_config_file)
    processor = Sogou_Processor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
    train_examples = processor.get_train_examples(data_dir)
    model = BertForSequenceClassification(bert_config,
                                          len(label_list)).to(assigned_device)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


    state = [parameter for parameter in model.parameters()] 

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_dataloader = DataLoader(train_data, train_batch_size)

    start_time_backward = torch.cuda.Event(enable_timing=True)
    stop_time_backward = torch.cuda.Event(enable_timing=True)

    start_time_comm = torch.cuda.Event(enable_timing=True)
    stop_time_comm = torch.cuda.Event(enable_timing=True)
    
    time_comm_list = list()
    time_backward_list = list() 

    reducer = _get_compression_param("ExactSerial_ps", assigned_device, None)

    memories = [torch.zeros_like(p) for p in model.parameters()]
    send_buffers = [torch.zeros_like(p) for p in model.parameters()]

    for idx, data in enumerate(train_dataloader):
        batch = tuple(t.to(assigned_device) for t in data)
        input_ids, input_mask, segment_ids, label_ids = batch
        (loss, _) = model(input_ids, segment_ids, input_mask, label_ids)
        torch.cuda.synchronize()
        start_time_backward.record()
        loss.backward()
        stop_time_backward.record()
        
        start_time_comm.record()
        grad_list = [parameter.grad for parameter in model.parameters()]

        for grad, memory, send_bfr in zip(grad_list, memories, send_buffers):
            send_bfr.data[:] = grad + memory
        reducer.reduce(send_buffers, grad_list, memories)

        stop_time_comm.record()
        torch.cuda.synchronize()

        time_comm_list.append(start_time_comm.elapsed_time(stop_time_comm))

        time_backward_list.append(start_time_backward.elapsed_time(stop_time_backward))

        if idx == 60:
            file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict['args'] = args.__str__()
            data_dict['comm_timing_log'] = time_comm_list
            data_dict['backward_timing_log'] = time_backward_list
            file_name = "bert_serial_full_out_ps_file_{}.json".format(global_rank)
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            file_uploader.push_file(file_name, 
                                    "{}/{}".format(args.s3_prefix, file_name))
            print ("Done bert")
            break

if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Large Scale Verification"))
    log_file_name = os.path.basename(args.log_file).split(".")[0]+"_args_logged_{}.log".format(9)
    # timing_logging = os.path.basename(args.log_file).split(".")[0]+"_time_logged_{}.json".format(args.device)
    logging.basicConfig(filename=log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Arguments: {}".format(args))
    print ("In If")
    print (args)
    dist.init_process_group(backend="NCCL", init_method="env://")
    print ("Dist connected")
    # main_bert(args)
    # main_bert_single(args)
    # powersgd_bert(args, 4)
    # powersgd_bert(args, 8)
    # powersgd_bert(args, 16)

    # powersgd_bert_integrated(args, 4)
    # powersgd_bert_integrated(args, 8)
    # powersgd_bert_integrated(args, 16)
    
    # topk_bert_single_call(args, 0.001)
    # mstopk_serial(args, 0.001)
    # topk_bert_single_call(args, 0.01)
    # mstopk_serial(args, 0.01)
    
    # topk_bert_single_call(args, 0.01)
    # topk_bert(args, 0.2)
    # topk_bert(args, 0.1)
    # topk_bert(args, 0.01)
    # signsgd_bert(args)
    # signsgd_bert(args)
    # signsgd_bert_single_call(args)

    fullcomm_serial_ps(args)

