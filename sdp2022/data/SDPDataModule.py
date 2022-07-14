from abc import ABC
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .batch_processing import BatchProcessing


class SDPDataModule(pl.LightningDataModule, ABC):
    def __init__(
            self,
            train_batch_size: Optional[int] = None,
            test_batch_size: int = 16,
            n_train_samples: int = 1024,
            n_val_samples: Optional[int] = None,
            n_test_samples: Optional[int] = None,
            mode: str = 'training'
    ):
        super().__init__()
        if mode == 'training':
            self.expected_batches = n_train_samples / train_batch_size
            batch_processing = BatchProcessing(
                train_batch_size=train_batch_size,
                n_val_samples=n_val_samples,
                n_test_samples=n_test_samples
            )
            train_sample_size = int(len(batch_processing.train) // self.expected_batches)
            self.n_labels = len(batch_processing.classes)

            self.train_data = list(batch_processing.train.index.values)
            self.val_data = list(batch_processing.val.index.values)
            self.test_data = list(batch_processing.test.index.values)

            self.train_batch_size = train_sample_size
            self.test_batch_size = test_batch_size

            self.train_batch_processing = batch_processing.build_train_batch
            self.val_batch_processing = batch_processing.build_val_batch
            self.eval_batch_processing = batch_processing.build_test_batch
        elif mode == 'testing':
            # TODO discriminate testing and validations
            batch_processing = BatchProcessing(
                mode='validation',
                n_test_samples=n_test_samples)
            self.pred_data = list(batch_processing.test.index.values)
            self.pred_batch_size = test_batch_size
            self.pred_batch_processing = batch_processing.build_pred_batch
            self.pred_samples = batch_processing.test
            self.map_classes = batch_processing.map_classes

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=False,
            collate_fn=self.train_batch_processing
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.test_batch_size,
            collate_fn=self.val_batch_processing
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            collate_fn=self.eval_batch_processing
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_data,
            batch_size=self.pred_batch_size,
            collate_fn=self.pred_batch_processing
        )
