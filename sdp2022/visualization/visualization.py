import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sdp2022.data.batch_processing import BatchProcessing
from sdp2022.utils.evaluator import Evaluator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


if __name__ == "__main__":
    path = "../../reports/"

    def converter(instr):
        return np.fromstring(instr[1:-1], sep=' ')

    # download summary from https://drive.google.com/file/d/1jar2Zq-14XvCteLkCvN9uBiX6w14LLEH/view?usp=sharing
    pred_samples = pd.read_csv(f"{path}summary.csv", converters={'predictions': converter})

    # fields = ["title", 'az_Claim_Abs', 'az_Method_Abs',
    #           'az_Conclusion_Abs', 'citation', 'reference',
    #           'recommendation'
    #           ]
    # ori_len = len(pred_samples)
    pred_samples = pred_samples[pred_samples["mode"] != 'recommendation']
    # pred_samples = pred_samples[pred_samples["mode"].isin(fields[:])]
    # pred_samples["sample_count"] = pred_samples.groupby("core_id").label.transform('count')
    # pred_samples = pred_samples[pred_samples.sample_count > 1]
    pred_samples.reset_index(drop=True, inplace=True)

    data = BatchProcessing(augment=None)

    eval = Evaluator(
        metric="f1_weighted",
        pred_samples=pred_samples,
        map_classes=data.map_classes,
        weighting_scheme="free"
    )

    _, preds = eval(
        save_report=False,
        return_report=True
    )

    preds["class_count"] = preds.groupby("label").label.transform('count')

    # update labels which count is less than a threshold and to a different label 36
    group = preds[preds['class_count'] < 200].label.unique()

    # update predictions for the same items in group
    label_correction = []
    for i in preds.label:
        item = 36 if i in group else i
        label_correction.append(item)
    preds["corrected_label"] = label_correction

    # if the predictions is a class inside the group,
    # the predictions should become the same as label
    pred_correction = []
    for i in preds.theme:
        item = 36 if i in group else i
        pred_correction.append(item)
    preds["corrected_pred"] = pred_correction

    map_classes = data.map_classes
    back_map_classes = {}
    for i, item in map_classes.items():
        if item not in group:
            back_map_classes[item] = i
    back_map_classes[36] = "others"

    preds.replace({"corrected_label": map_classes}, inplace=True)

    # By definition a confusion matrix C is such that  C(i, j)
    # is equal to the number of observations known to be in group
    # i  and predicted to be in group j.
    CM = confusion_matrix(preds.corrected_label, preds.corrected_pred)

    cmd = ConfusionMatrixDisplay(CM, display_labels=list(back_map_classes.values()))
    _, ax = plt.subplots(figsize=(50, 50))
    cmd.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=90)

    fig = plt.gcf()

    font = {
        'family': 'serif',
        'weight': 'bold',
        'size': 30
    }

    plt.rc('font', **font)

    plt.tight_layout()
    plt.show()
    plt.draw()
    fig.savefig(f"{path}conf.pdf", pad_inches=1)
