import argparse
import os
import random
import re
import time
from typing import List, Tuple

import fasttext
import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit


nlp = spacy.load(
    "en_core_web_sm",
    disable=[
        "ner",
        "tok2vec",
        "tagger",
        "parser",
        "attribute_ruler",
        "lemmatizer",
    ],
)


def undersample(X: np.array, y: np.array) -> Tuple[np.array, np.array]:
    exclude_label = 0
    include_label = 1
    include_count = len(y[y == include_label])
    exclude_count = len(y[y == exclude_label])

    smaller = include_count if include_count < exclude_count else exclude_count
    print(include_count, exclude_count, smaller)

    incl_indexes = np.where(y == include_label)[0]
    excl_indexes = np.where(y == exclude_label)[0]

    print(incl_indexes.shape, excl_indexes.shape)

    undersampled_indexes = list(random.sample(incl_indexes.tolist(), smaller))
    undersampled_indexes.extend(list(random.sample(excl_indexes.tolist(), smaller)))

    print(len(undersampled_indexes))
    return X[undersampled_indexes], y[undersampled_indexes]


def preprocess_text(text: str) -> str:
    """Preprocesses a custom text field using spacy tokenizer and removing
    stopwords. Preprocessed text is returned as a List of tokenized strings.
    """
    preprocessed = nlp(text)
    return " ".join(token.text for token in preprocessed if not token.is_stop)


def write_temp_fasttext_train_file(X: List[str], y: List[str], outfile: str):
    with open(outfile, "w") as fp:
        for X_row, y_row in zip(X, y):
            fp.write(f"__label__{y_row} {X_row}\n")


def train_fasttext(
    X: List[str],
    y: List[str],
    lr=1.3,
    epoch=100,
    wordNgrams=9,
    dim=200,
    loss="hs",
    undersampling=False,
) -> tuple[fasttext.FastText._FastText, float]:
    VECTORS_FILEPATH: str = (
        "../../data/embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin"
    )
    TRAIN_FILEPATH: str = "../../data/train.data"

    if undersampling:
        print("before", X.shape, y.shape)
        X, y = undersample(X=X, y=y)
        print("after", X.shape, y.shape)

    write_temp_fasttext_train_file(X=X, y=y, outfile=TRAIN_FILEPATH)

    start = time.time()
    model = fasttext.train_supervised(
        input=TRAIN_FILEPATH,
        lr=lr,
        epoch=epoch,
        wordNgrams=wordNgrams,
        dim=dim,
        loss=loss,
        # pretrainedVectors=VECTORS_FILEPATH,
    )
    stop = time.time()
    total_time = stop - start
    print(f"total_time: {total_time}")
    return model, total_time


def train_and_evaluate_fasttext(input_data_file, test_column: str = "theme"):
    """
    Trains and evaluates fasttext

    :param test_column:
    :param input_data_file: a TSV file with the following two columns:
        column 1: abstract of the citation, column 2: classification label
    """
    df = pd.read_csv(input_data_file, delimiter=",")
    df["Title"] = df["title"].fillna(" ")
    df["Abstract"] = df["description"].fillna(" ")

    # get abstracts column into a list
    X = list(df["Title"].str.cat(df["Abstract"], sep=" "))
    X = [preprocess_text(elem) for elem in X]

    X = [re.sub(r"[\W]+", " ", elem) for elem in X]
    X = [re.sub(r"[\n\r\t ]+", " ", elem) for elem in X]
    X = [elem.lower() for elem in X]
    X = np.asarray(X)

    # get labels column into a list
    y = list(df[test_column])
    y = ["_".join(item.split()) for item in y]
    y = np.asarray(y)

    precision = []
    recall = []
    f1_list = []
    train_times = []
    results_dict = {}

    seeds = [60, 55, 98, 27, 36, 44, 72, 67, 3, 42]

    # use same seeds across all baselines
    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)

        for train_indexes, test_indexes in sss.split(X, y):
            # split into training and test subsets
            X_train, X_test = (
                X[train_indexes],
                X[test_indexes],
            )

            # split labels into training and test subsets
            y_train, y_test = y[train_indexes], y[test_indexes]

            model, train_time = train_fasttext(X=X_train, y=y_train)

            y_pred = [model.predict(row) for row in X_test]

            predictions = []
            for row in y_pred:
                predictions.append(row[0][0][9:])

            f1_list.append(f1_score(y_test, y_pred=predictions, average="micro"))
            print(f1_score(y_test, y_pred=predictions, average="macro"))
            print(f1_score(y_test, y_pred=predictions, average="micro"))
            precision.append(
                precision_score(y_test, y_pred=predictions, average="micro")
            )
            recall.append(recall_score(y_test, y_pred=predictions, average="micro"))
            train_times.append(train_time)

            print(f"prec={precision}, recall={recall} f1={f1_list}")

    results_dict["train_time"] = train_times
    results_dict["avg_train_time"] = np.mean(train_times)
    results_dict["f1_raw"] = f1_list
    results_dict["f1"] = np.asarray(f1_list).mean()
    results_dict["precision_raw"] = precision
    results_dict["precision"] = np.asarray(precision).mean()
    results_dict["recall_raw"] = recall
    results_dict["recall"] = np.asarray(recall).mean()

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile", default="../../data/raw/task1_train_dataset.csv", type=str
    )
    parser.add_argument(
        "--results_file",
        default="fasttext-results_summary.tsv",
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        default="../../models/fasttext/",
        type=str,
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    result_dict = train_and_evaluate_fasttext(input_data_file=args.infile)

    if os.path.isfile(f"{args.output_folder}/{args.results_file}"):
        df = pd.read_csv(f"{args.output_folder}/{args.results_file}", sep="\t")
        df = df.append(
            pd.DataFrame.from_dict(result_dict).transpose().reset_index(),
            ignore_index=True,
        )
    else:
        df = pd.DataFrame.from_dict(result_dict).transpose().reset_index()

    df.drop_duplicates().to_csv(args.results_file, sep="\t", index=False)
