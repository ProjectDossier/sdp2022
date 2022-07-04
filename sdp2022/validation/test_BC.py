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

model = BertClassifier.load_from_checkpoint(
    checkpoint_path=f"../checkpoints/{config.CHECKPOINT}",
    model_name=config.MODEL_NAME,
    num_labels=data_module.n_labels + 1
)

trainer = pl.Trainer(gpus=1)
trainer.predict(model=model, dataloaders=data_module.predict_dataloader())
