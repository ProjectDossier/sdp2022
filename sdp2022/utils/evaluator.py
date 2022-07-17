from os.path import join, isfile
import csv
from typing import List, Dict

import pandas as pd
from sklearn.metrics import f1_score
import numpy as np


def agg_preds(x):

    x['agg_preds'] = 0
    fields = list(x['mode'].values)
    rel_fields = ["description", "title"]

    if "description" in fields and len(x[~x['mode'].isin(rel_fields)]) > 0:
        # title + description + add fields
        x.loc[x['mode'] == "description", 'agg_preds'] = np.multiply(x[x['mode'] == "description"].predictions, .25)
        x.loc[x['mode'] == "title", 'agg_preds'] = np.multiply(x[x['mode'] == "title"].predictions, .25)
        weight = .5 / len(x[~x['mode'].isin(rel_fields)])
        x.loc[~x['mode'].isin(rel_fields), 'agg_preds'] = np.multiply(x[~x['mode'].isin(rel_fields)].predictions, weight)

    elif len(x[~x['mode'].isin(rel_fields)]) > 0:
        # title + add fields
        weight = .5 / len(x[~x['mode'].isin(rel_fields)])
        x.loc[~x['mode'].isin(rel_fields), 'agg_preds'] = np.multiply(x[~x['mode'].isin(rel_fields)].predictions, weight)
        x.loc[x['mode'] == "title", 'agg_preds'] = np.multiply(x[x['mode'] == "title"].predictions, .5)

    elif "description" in fields:
        # title + abstract
        x.loc[x['mode'] == "description", 'agg_preds'] = np.multiply(x[x['mode'] == "description"].predictions, .5)
        x.loc[x['mode'] == "title", 'agg_preds'] = np.multiply(x[x['mode'] == "title"].predictions, .5)

    else:
        # only title
        x.loc[x['mode'] == "title", 'agg_preds'] = x[x['mode'] == "title"].predictions

    x['agg_preds'] = [x.agg_preds.sum()] * len(x)

    return x


class Evaluator:
    """
    Class responsible for performing evaluation of the model.
    """
    def __init__(self,
                 metric,
                 write_csv: bool = True,
                 output_path: str = "../../reports/",
                 pred_samples=None,
                 map_classes=None,
                 run_id: str = None
                 ):
        self.metric = metric
        self.output_path = output_path
        self.write_csv = write_csv
        self.eval = {"f1_macro": "macro",
                     "f1_micro": "micro",
                     "f1_weighted": "weighted",
                     }
        self.csv_headers = ["epoch"] + [i for i in self.eval.keys()][:-1]
        self.pred_samples = pred_samples
        self.map_classes = map_classes
        self.run_id = run_id

    def __call__(self,
                 model=None,
                 examples: List[List[str]] = None,
                 ids: List = None,
                 labels: List[int] = None,
                 pred_scores=None,
                 epoch: int = -1,
                 out_f_name: str = "") -> Dict[str, float]:
        """

        :param model:
        :param examples:
        :param ids:
        :param labels:
        :param pred_scores:
        :param epoch:
        :param out_f_name:
        :return: returns a dict with all metrics from self.eval
        """

        output_path = self.output_path
        pred_scores = np.argmax(pred_scores, axis=1).tolist()

        np.save(
            file=f"{output_path}Evaluator_{out_f_name}_results.pkl",
            arr=pred_scores,
            allow_pickle=True,
            fix_imports=True
        )

        # in case the original set of examples is given
        if self.pred_samples is not None:

            pred_samples = self.pred_samples.copy()
            pred_samples['predictions'] = [i for i in pred_scores]

            if 'theme' in pred_samples.columns:
                pred_samples.drop(columns='theme', inplace=True)

            # in case the samples are augmented, the core_ids would be redundant
            # scores from different sources are aggregated
            if len(pred_samples.core_id.unique()) != len(pred_samples):
                pred_samples = pred_samples.groupby("core_id").apply(agg_preds)
                pred_samples.drop_duplicates(
                    subset='core_id',
                    inplace=False,
                    ignore_index=True
                )
                pred_field = 'agg_preds'
            else:
                pred_field = 'predictions'

            # map to original classes
            preds = {'index': [], 'pred': []}
            pred_classes = np.argmax(np.vstack(tuple(pred_samples[pred_field])), axis=1).tolist()
            for id, label in zip(ids, pred_classes):
                preds['index'].append(id)
                preds['pred'].append(label)

            p_copy = pd.merge(pred_samples, pd.DataFrame.from_dict(preds),
                              left_index=True, right_index=True)

            core_ids = self.pred_samples.core_id.unique()
            labels = []
            pred_scores = []
            preds = {'core_id': core_ids, 'theme': []}
            for id in core_ids:
                try:
                    labels.append(p_copy[p_copy.core_id == id].label.values[0])
                except AttributeError:
                    pass
                t_preds = p_copy[p_copy.core_id == id].pred.tolist()
                preds['theme'].append(max(set(t_preds), key=t_preds.count))

            pred_classes = preds['theme']

            pred_samples = pred_samples[pred_samples['mode'] == 'title']
            pred_samples = pd.merge(pred_samples, pd.DataFrame.from_dict(preds),
                                    on='core_id')
            pred_samples.reset_index(drop=True, inplace=True)

            # map classes to actual labels
            pred_samples["theme"] = pred_samples.replace({"theme": self.map_classes})["theme"]

            # save report
            for i in ['label', 'text', 'mode', 'predictions', 'agg_preds']:
                try:
                    pred_samples.drop(columns=i, inplace=True)
                except KeyError:
                    pass

            pred_samples.to_csv(f'{output_path}{self.run_id}.csv')

        else:
            pred_classes = np.argmax(pred_scores, axis=1).tolist()

        if labels is not None and -1 not in labels and len(labels) > 0:
            # if model is not None:
            #     pred_scores = model.predict(examples)
            #     pred_scores = pred_scores
            eval = {}
            for name, metric in self.eval.items():
                eval[name] = f1_score(labels, pred_classes, average=metric)

            acc = eval[self.metric]

            csv_file = f"Evaluator_{out_f_name}_results.csv"
            if output_path is not None and self.write_csv:
                csv_path = join(output_path, csv_file)
                out_file_exists = isfile(csv_path)
                with open(csv_path, mode="a" if out_file_exists else 'w', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not out_file_exists:
                        writer.writerow(self.csv_headers)

                    writer.writerow([epoch] + list(eval.values())[:-1])
            return eval
        else:
            return 0
