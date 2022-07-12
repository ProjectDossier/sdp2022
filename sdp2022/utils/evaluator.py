from os.path import join, isfile
import csv
from typing import List

import pandas as pd
from sklearn.metrics import f1_score
import numpy as np


class Evaluator:
    """
    """
    def __init__(self,
                 write_csv: bool = True,
                 metric="f1_weighted",
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
                     "f1": None}
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
                 out_f_name: str = "") -> float:

        output_path = self.output_path

        if self.pred_samples is not None:
            p_copy = self.pred_samples.copy()
            c_pred_scores = pred_scores
            np.save(f"{output_path}Evaluator_{out_f_name}_results.pkl", c_pred_scores, allow_pickle=True, fix_imports=True)
            preds = {'index': [], 'pred': []}
            pred_classes = np.argmax(pred_scores, axis=1).tolist()
            for id, label in zip(ids, pred_classes):
                preds['index'].append(id)
                preds['pred'].append(label)

            p_copy = pd.merge(p_copy, pd.DataFrame.from_dict(preds),
                              left_index=True, right_index=True)

            core_ids = self.pred_samples.core_id.unique()
            labels = []
            pred_scores = []
            preds = {'core_id': core_ids, 'theme': []}
            for id in core_ids:
                labels.append(p_copy[p_copy.core_id == id].label.values[0])
                t_preds = p_copy[p_copy.core_id == id].pred.tolist()
                preds['theme'].append(max(set(t_preds), key=t_preds.count))

            pred_scores = preds['theme']

            self.pred_samples = self.pred_samples[self.pred_samples['mode'] == 'title']
            self.pred_samples.drop(['mode', 'text'], inplace=False, axis=1)
            self.pred_samples = pd.merge(self.pred_samples, pd.DataFrame.from_dict(preds),
                                         on='core_id')
            self.pred_samples.reset_index(drop=True, inplace=True)

            self.pred_samples.to_csv(f'{output_path}DoSSIER_run.csv')

        if labels is not None:
            if model is not None:
                pred_scores = model.predict(examples)
                pred_scores = pred_scores

            eval = {}
            for name, metric in self.eval.items():
                eval[name] = f1_score(labels, pred_scores, average=metric)

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
            return acc
