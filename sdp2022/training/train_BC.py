import pytorch_lightning as pl
import yaml
from dotmap import DotMap
from sdp2022.data.SDPDataModule import SDPDataModule
from bert_classifier import BertClassifier
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == '__main__':

    with open('../../config/train_config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["xtremedistil_classifier"]
        config = DotMap(config)

    data_module = SDPDataModule(
        train_batch_size=config.TRAIN_BATCH_SIZE,
        test_batch_size=config.TEST_BATCH_SIZE,
        n_train_samples=config.N_TRAIN_SAMPLES,
        n_val_samples=config.N_VAL_SAMPLES,
        n_test_samples=config.N_TEST_SAMPLES,
    )

    n_training_steps = (
            data_module.expected_batches *
            config.TRAIN_BATCH_SIZE *
            config.N_EPOCHS
    )

    model = BertClassifier(
        model_name=config.MODEL_NAME,
        num_labels=data_module.n_labels + 1,
        n_warmup_steps=config.WARMUP_STEPS,
        n_training_steps=n_training_steps,
        batch_size=config.TRAIN_BATCH_SIZE,
        metric=config.TRACK_METRIC
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="../../models/bert_classifier/checkpoints",
        filename="best-checkpoint_",
        save_top_k=1,
        verbose=True,
        monitor=config.TRACK_METRIC,
        mode="max"
    )

    logger = TensorBoardLogger(
        save_dir="../../reports/SDP_logs",
        name=config.LOGGER_NAME
    )

    early_stopping_callback = EarlyStopping(
        monitor=config.TRACK_METRIC,
        patience=config.PATIENCE,
        mode='max'
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=config.N_EPOCHS,
        gpus=[0, 1],
        progress_bar_refresh_rate=1,
        accumulate_grad_batches=config.ACCUM_ITER,
        check_val_every_n_epoch=config.EVAL_EVERY_N_EPOCH,
        log_every_n_steps=10
    )

    trainer.fit(
        model=model,
        datamodule=data_module
    )

    trainer.test(
        dataloaders=data_module.test_dataloader()
    )
