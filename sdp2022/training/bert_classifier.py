from abc import ABC
import numpy as np
import pytorch_lightning as pl
from sdp2022.utils.evaluator import Evaluator
from torch import nn, mean
from transformers import AdamW, AutoConfig, AutoModel, get_linear_schedule_with_warmup


class BertClassifier(pl.LightningModule, ABC):
    def __init__(
        self,
        model_name: str,
        num_labels: int = None,
        n_training_steps=None,
        n_warmup_steps=None,
        batch_size=16,
        pred_samples=None,
        map_classes=None,
        run_id=None,
        metric="f1_weighted",
        weighting_scheme="free",
    ):
        super().__init__()
        self.n_training_steps = n_training_steps
        self.batch_size = batch_size
        self.n_warmup_steps = n_warmup_steps
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)

        if num_labels is not None:
            self.config.num_labels = num_labels - 1

        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        try:
            self.out_size = self.bert.pooler.dense.out_features
        except:
            self.out_size = self.bert.config.dim

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.out_size, num_labels - 1)

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.softmax = nn.Softmax(dim=1)

        self.evaluator = Evaluator(
            metric=metric,
            pred_samples=pred_samples,
            map_classes=map_classes,
            run_id=run_id,
            weighting_scheme=weighting_scheme,
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        sequence_output = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        ).last_hidden_state

        linear_output = self.linear(
            self.dropout(sequence_output[:, 0, :].view(-1, self.out_size))
        )
        return linear_output

    def training_step(self, batch, batch_idx):
        batch, labels, weights = batch
        model_predictions = self(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )

        loss_value = self.criterion(model_predictions, labels)
        loss_value = mean(loss_value * weights)
        self.log("train_loss", loss_value, prog_bar=True, logger=True)
        return loss_value

    def eval_batch(self, batch):
        batch, labels, ids = batch

        preds = self(
            batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
        )

        return {"id": ids, "prediction": preds, "labels": labels}

    def eval_epoch(self, outputs, name, epoch=-1):
        ids, labels, preds = [], [], []
        if name == "pred":
            outputs = outputs[0]
        for output in outputs:
            ids.extend(output["id"])
            labels.extend(output["labels"])
            preds.append(output["prediction"].cpu().detach().numpy())

        preds = np.concatenate(preds, 0)

        eval = self.evaluator(
            ids=ids, labels=labels, pred_scores=preds, epoch=epoch, out_f_name=name
        )

        for metric, value in eval.items():
            self.log(metric, value, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        return self.eval_batch(batch)

    def validation_epoch_end(self, outputs):
        self.eval_epoch(outputs, "during_training", self.current_epoch)

    def test_step(self, batch, batch_idx):
        return self.eval_batch(batch)

    def test_epoch_end(self, outputs):
        self.eval_epoch(outputs, "dev")

    def predict_step(self, batch, batch_idx):
        return self.eval_batch(batch)

    def on_predict_epoch_end(self, outputs):
        self.eval_epoch(outputs, "pred")

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        optimizer_class = AdamW
        optimizer_params = {"lr": 2e-5}
        linear = ["linear.weight", "linear.bias"]
        params = list(
            map(
                lambda x: x[1],
                list(filter(lambda kv: kv[0] in linear, param_optimizer)),
            )
        )
        base_params = list(
            map(
                lambda x: x[1],
                list(filter(lambda kv: kv[0] not in linear, param_optimizer)),
            )
        )
        optimizer = optimizer_class(
            [{"params": base_params}, {"params": params, "lr": 1e-3}],
            **optimizer_params
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )
