from lightning_modules.model import DistilBERTClass
from lightning_modules.dataset import lmaoDataset
from lightning_modules import config
import lightning as L
from transformers import DistilBertTokenizer
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.loggers import WandbLogger
import wandb

if __name__ == '__main__':
#     wandb_logger = WandbLogger(
#     project="CSS",
#     log_model="all",
#     save_dir=config.LOGS_DIR,
#     name="distilBERT_run_1",
# ) 
    # strategy = DeepSpeedStrategy()
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
    dm = lmaoDataset(datadir='jigsaw-toxic-comment-classification-challenge/clean_train_data.csv',
                          batch_size=config.BATCH_SIZE,
                          tokenizer=tokenizer,
                          max_len=config.MAX_LEN,
                          num_workers=config.NUM_WORKERS
                          )
    dm.prepare_data()
    dm.setup()
    
    model = DistilBERTClass(learning_rate=config.LEARNING_RATE)
    
    trainer = L.Trainer(
        strategy='auto',
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
        max_epochs=config.EPOCHS,
        # logger=wandb_logger
    )
    
    trainer.fit(model, dm)
    
    wandb.finish()