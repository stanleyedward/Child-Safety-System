from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from transformers import DistilBertModel
import torch
import lightning as L
from torchmetrics import Accuracy

class DistilBERTClass(L.LightningModule):
    
    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 1)
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task='binary')
        # self.prepare_data_per_node = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        ids = batch['ids']
        mask = batch['mask']
        token_type_ids = batch['token_type_ids']
        targets = batch['targets']
        
        outputs = self.forward(ids, mask, token_type_ids)
        loss = self.loss_fn(torch.squeeze(outputs, dim=1), targets)
        
        # if batch_idx % 1000 == 0:
            
        #     self.log("train_loss",
        #              loss,
        #              prog_bar=True)
        #     self.log("train_acc",
        #              self.accuracy(torch.squeeze(outputs, dim=1), targets),
        #              prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        ids = batch['ids']
        mask = batch['mask']
        token_type_ids = batch['token_type_ids']
        targets = batch['targets']
        
        outputs = self.forward(ids, mask, token_type_ids)
        loss = self.loss_fn(torch.squeeze(outputs, dim=1), targets)
        
        # if batch_idx % 1000 == 0:
        #     self.log("val_loss",
        #              loss,
        #              prog_bar=True,
        #              )
        #     self.log("val_acc",
        #              self.accuracy(torch.squeeze(outputs, dim=1), targets),
        #              prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(params =  self.parameters(), lr=self.learning_rate)
        
        