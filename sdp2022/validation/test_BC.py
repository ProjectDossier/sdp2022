from dotmap import DotMap
import pytorch_lightning as pl
from sdp2022.training.bert_classifier import BertClassifier
from sdp2022.data.SDPDataModule import SDPDataModule
import yaml


with open('./config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)["bert_classifier"]
    config = DotMap(config)


data_module = SDPDataModule(
    test_batch_size=config.TEST_BATCH_SIZE,
    n_test_samples=config.N_TEST_SAMPLES,
    mode='testing'
)
checkpoint_path = "../../models/bert_classifier/checkpoints/" + config.CHECKPOINT
model = BertClassifier.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    model_name=config.MODEL_NAME,
    num_labels=len(data_module.map_classes.keys()) + 1,
    pred_samples=data_module.pred_samples,
    map_classes=data_module.map_classes,
)

trainer = pl.Trainer(gpus=1)
trainer.predict(model=model, dataloaders=data_module.predict_dataloader())
