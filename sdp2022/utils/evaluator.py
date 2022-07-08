from os.path import join, isfile
import csv
from typing import List, Dict

import pandas as pd
from sklearn.metrics import f1_score
import numpy as np


class Evaluator:
    """
    Class responsible for performing evaluation of the model.
    """
    def __init__(self,
                 metric,
                 write_csv: bool = True,
                 output_path: str = "../../reports/",
                 pred_samples=None,
                 map_classes=None
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

        if self.pred_samples is not None:
            preds = {'index': [], 'theme': []}
            for id, label in zip(ids, pred_scores):
                preds['index'].append(id)
                preds['theme'].append(self.map_classes[label])
            self.pred_samples = self.pred_samples.merge(pd.DataFrame.from_dict(preds))
            self.pred_samples.to_csv(f'{output_path}DoSSIER_run.csv')
            return 0
        else:
            if model is not None:
                pred_scores = model.predict(examples)
                pred_scores = pred_scores

            eval = {}
            for name, metric in self.eval.items():
                eval[name] = f1_score(labels, pred_scores, average=metric)

            csv_file = "Evaluator_" + out_f_name + "_results.csv"
            if output_path is not None and self.write_csv:
                csv_path = join(output_path, csv_file)
                out_file_exists = isfile(csv_path)
                with open(csv_path, mode="a" if out_file_exists else 'w', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not out_file_exists:
                        writer.writerow(self.csv_headers)

                    writer.writerow([epoch] + list(eval.values())[:-1])
            return eval
