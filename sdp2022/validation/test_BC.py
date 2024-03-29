from dotmap import DotMap
import pytorch_lightning as pl
from sdp2022.training.bert_classifier import BertClassifier
from sdp2022.data.SDPDataModule import SDPDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import yaml


with open("../../config/prediction_config.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)[10]  # name of the configuration
    config = DotMap(config)


data_module = SDPDataModule(
    test_batch_size=config.TEST_BATCH_SIZE,
    n_test_samples=config.N_TEST_SAMPLES,
    mode=config.MODE,
    augment=config.AUGMENT,
)

model = BertClassifier.load_from_checkpoint(
    checkpoint_path=f"../../models/bert_classifier/checkpoints/{config.CHECKPOINT}",
    model_name=config.MODEL_NAME,
    num_labels=len(data_module.map_classes.keys()) + 1,
    pred_samples=data_module.pred_samples,
    map_classes=data_module.map_classes,
    run_id=config.RUN_ID,
    weighting_scheme=config.WEIGHTING_SCHEME,
)

logger = TensorBoardLogger(save_dir="../../reports/SDP_logs", name="BC_preds")

trainer = pl.Trainer(logger=logger, gpus=1)

trainer.predict(model=model, dataloaders=data_module.predict_dataloader())
