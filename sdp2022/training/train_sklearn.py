from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

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

MODELS_RANDOM_SEED = 42


nlp = spacy.load(
    "en_core_sci_lg",
    disable=[
        "ner",
        "tok2vec",
        "tagger",
        "parser",
        "attribute_ruler",
        "lemmatizer",
    ],
)

def preprocess_text(text: str) -> str:
    """Preprocesses a custom text field using spacy tokenizer and removing
    stopwords. Preprocessed text is returned as a List of tokenized strings.
    """
    preprocessed = nlp(text)
    return " ".join(token.text for token in preprocessed if not token.is_stop)


def train_and_evaluate_sklearn(model, input_data_file, test_column: str = "theme"):
    """
    Trains and evaluates fasttext

    :param test_column:
    :param input_data_file: a TSV file with the following two columns:
        column 1: abstract of the citation, column 2: classification label
    """
    scores_average = "micro"

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
        print(f"{seed=}")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)

        for train_indexes, test_indexes in sss.split(X, y):
            # split into training and test subsets
            X_train, X_test = (
                X[train_indexes],
                X[test_indexes],
            )

            # split labels into training and test subsets
            y_train, y_test = y[train_indexes], y[test_indexes]

            start = time.time()
            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 3),
                                    stop_words='english')
            features = tfidf.fit_transform(X_train).toarray()

            # model = model_class()
            # model = model_class()
            model.fit(features, y_train)
            # model, train_time = train_fasttext(X=X_train, y=y_train)
            stop = time.time()
            train_time = stop - start
            print(f"total_time: {train_time}")

            X_test_features = tfidf.transform(X_test)
            y_pred = model.predict(X_test_features)

            predictions = []
            for row in y_pred:
                predictions.append(row)

            f1_list.append(f1_score(y_test, y_pred=predictions, average=scores_average))
            print(f1_score(y_test, y_pred=predictions, average="macro"))
            print(f1_score(y_test, y_pred=predictions, average=scores_average))
            precision.append(
                precision_score(y_test, y_pred=predictions, average=scores_average)
            )
            recall.append(recall_score(y_test, y_pred=predictions, average=scores_average))
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
        "--output_folder",
        default="../../models/sklearn/",
        type=str,
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for model in [MultinomialNB(), KNeighborsClassifier(), SVC()]:
        results_file = f"{str(model)}-results_summary.tsv"
        print(model, results_file)
        result_dict = train_and_evaluate_sklearn(model=model, input_data_file=args.infile)

        if os.path.isfile(f"{args.output_folder}/{results_file}"):
            df = pd.read_csv(f"{args.output_folder}/{results_file}", sep="\t")
            df = df.append(
                pd.DataFrame.from_dict(result_dict).transpose().reset_index(),
                ignore_index=True,
            )
        else:
            df = pd.DataFrame.from_dict(result_dict).transpose().reset_index()

        df.drop_duplicates().to_csv(f"{args.output_folder}/{results_file}", sep="\t", index=False)
