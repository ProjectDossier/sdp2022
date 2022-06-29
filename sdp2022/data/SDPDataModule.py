from abc import ABC

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .batch_processing import BatchProcessing


class SDPDataModule(pl.LightningDataModule, ABC):
    def __init__(self, train_batch_size: int, test_batch_size: int, n_train_samples: int):
        super().__init__()
        self.expected_batches = n_train_samples / train_batch_size
        batch_processing = BatchProcessing(train_batch_size=train_batch_size)
        train_sample_size = int(len(batch_processing.train) // self.expected_batches)
        self.n_labels = len(batch_processing.classes)

        self.train_data = list(batch_processing.train.index.values)
        self.val_data = list(batch_processing.val.index.values)
        self.test_data = list(batch_processing.test.index.values)

        self.train_batch_size = train_sample_size
        self.test_batch_size = test_batch_size

        self.train_batch_processing = batch_processing.build_train_batch
        self.val_batch_processing = batch_processing.build_test_batch
        self.eval_batch_processing = batch_processing.build_test_batch

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
