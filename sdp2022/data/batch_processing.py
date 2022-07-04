import zipfile
import pandas as pd
from os.path import join as join_path
from typing import Dict, List
import random
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch import tensor, LongTensor, FloatTensor
from sklearn.utils.class_weight import compute_class_weight as c_weights
import numpy as np


class BatchProcessing:
    def __init__(
            self,
            train_f_name: str = 'task1_train_dataset.csv.zip',
            test_f_name: str = 'task1_test_no_label.csv.zip',
            mode: str = 'trainig',
            splits: Dict = {'train': .60, 'val': .10, 'test': .30},
            r_seed: int = 42,
            tokenizer_name: str = "bert-base-uncased",
            train_batch_size: int = 16,
            n_val_samples: int = None,
            n_test_samples: int = None
    ):
        self.train_batch_size = train_batch_size
        self.tokenizer_name = tokenizer_name

        random.seed(r_seed)
        path = '../../data/raw/'

        if mode == 'trainig':
            data = zipfile.ZipFile(join_path(path, train_f_name), 'r')
            data = pd.read_csv(data.open(data.filelist[0].filename))
            self.data = data

            n_samples = len(data)
            val_size = int(n_samples * splits["val"])

            classes = data["theme"].unique()
            map_classes = {}
            for i, class_ in enumerate(classes):
                map_classes[class_] = i
            self.classes = list(map_classes.values())

            data["label"] = data.replace({"theme": map_classes})["theme"]

            train, self.test, _, _ = train_test_split(
                data,
                data.label,
                test_size=splits["test"],
                random_state=r_seed
            )

            val_size = val_size / len(train)

            self.train, self.val, _, _ = train_test_split(
                train,
                train.label,
                test_size=val_size,
                random_state=r_seed
            )

            if n_val_samples is not None:
                self.val = self.val[:n_val_samples]

        elif mode == 'testing':
            data = zipfile.ZipFile(join_path(path, train_f_name), 'r')
            data = pd.read_csv(data.open(data.filelist[0].filename))
            classes = data["theme"].unique()
            map_classes = {}
            for i, class_ in enumerate(classes):
                map_classes[i] = class_
            self.map_classes = map_classes
            data = zipfile.ZipFile(join_path(path, test_f_name), 'r')
            self.test = pd.read_csv(data.open(data.filelist[0].filename))

        if n_test_samples is not None:
            self.test = self.test[:n_test_samples]

    def tokenize_samples(self, texts):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name,
                                                  model_max_length=512)

        tokenized_text = tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            return_token_type_ids=True
        )

        return tokenized_text

    def build_train_batch(self, sample_ids: List):

        labels = list(self.train.loc[sample_ids].label)

        n_x_class = self.train_batch_size // len(self.classes) + 1

        batch_, labels_, lengths, sample_classes = [], [], [], []
        for class_ in self.classes:
            samples = [i for i in range(len(labels)) if labels[i] == class_]
            if len(samples) == 0:
                continue
            sample_classes.append(class_)
            random.shuffle(samples)
            samples_chunk = [sample_ids[i] for i in samples[:n_x_class]]
            lengths.append(len(samples_chunk))
            labels_.extend([class_] * lengths[-1])
            batch_.extend(samples_chunk)
            if len(labels_) == self.train_batch_size:
                break
        # TODO complete batch with random sample of the remaining samples
        batch_ = self.train.loc[batch_]
        batch = self.tokenize_samples(list(batch_.title))
        labels = tensor(labels_).type(LongTensor)
        weights = c_weights(
            class_weight='balanced',
            classes=np.array(sample_classes),
            y=np.array(labels_)
        )
        weights = sum([[w] * l for w, l in zip(weights, lengths)], [])
        weights = tensor(weights[:self.train_batch_size]).type(FloatTensor)

        return batch, labels, weights

    def build_val_batch(self, sample_ids: List):
        batch = list(self.val.loc[sample_ids].title)
        batch = self.tokenize_samples(batch)
        labels = list(self.val.loc[sample_ids].label)
        return batch, labels, sample_ids

    def build_test_batch(self, sample_ids: List):
        batch = list(self.test.loc[sample_ids].title)
        batch = self.tokenize_samples(batch)
        labels = list(self.test.loc[sample_ids].label)
        return batch, labels, sample_ids

    def build_pred_batch(self, sample_ids: List):
        batch = list(self.test.loc[sample_ids].title)
        batch = self.tokenize_samples(batch)
        return batch, [-1] * len(sample_ids), sample_ids



