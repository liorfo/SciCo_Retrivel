from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Any, Optional
import torchmetrics as tm
from deepspeed.ops.adam import FusedAdam


os.environ['CUDA_HOME'] = '/usr/local/nvidia/cuda/11.7'
os.environ['RDMAV_FORK_SAFE'] = '1'


# tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K",
#                                           cache_dir='/cs/labs/tomhope/forer11/cache/')
# model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K",
#                                              trust_remote_code=True,
#                                              torch_dtype=torch.float16,
#                                              cache_dir='/cs/labs/tomhope/forer11/cache/')
#
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(device)
# input_context = "When did the Soviet Union end?"
# input_ids = tokenizer.encode(input_context, return_tensors="pt").to(device)
# model = model.to(device)
# output = model.generate(input_ids, max_length=128, temperature=0.7)
# output_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(output_text)


class LlamaMulticlassCrossEncoder(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related
    1 coref
    2. hypernym
    3. neutral (hyponym)
    '''

    def __init__(self, config, num_classes=4):
        super(LlamaMulticlassCrossEncoder, self).__init__()
        self.cdlm = 'cdlm' in config["model"]["bert_model"].lower()
        self.long = True if 'longformer' in config["model"]["bert_model"] or self.cdlm else False
        self.config = config

        # self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K",
        #                                                cache_dir='/cs/labs/tomhope/forer11/cache/')
        # self.model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K",
        #                                                   trust_remote_code=False,
        #                                                   torch_dtype=torch.float16,
        #                                                   cache_dir='/cs/labs/tomhope/forer11/cache/')

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        self.tokenizer.add_tokens('<def>', special_tokens=True)
        self.tokenizer.add_tokens('</def>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')
        self.start_def = self.tokenizer.convert_tokens_to_ids('<def>')
        self.end_def = self.tokenizer.convert_tokens_to_ids('</def>')
        self.sep = self.tokenizer.convert_tokens_to_ids('</s>')
        self.doc_start = self.tokenizer.convert_tokens_to_ids('<doc-s>') if self.cdlm else None
        self.doc_end = self.tokenizer.convert_tokens_to_ids('</doc-s>') if self.cdlm else None

        self.model = AutoModel.from_pretrained(config["model"]["bert_model"], add_pooling_layer=False,
                                               cache_dir='/cs/labs/tomhope/forer11/cache', attention_window=512)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, num_classes)
        # self.linear = nn.Linear(32004, num_classes)
        # self.linear.weight.data = self.linear.weight.data.to(torch.float16)
        # self.linear.bias.data = self.linear.bias.data.to(torch.float16)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.acc = tm.Accuracy(top_k=1, task="multiclass", num_classes=num_classes)
        self.f1 = tm.F1Score(task="multiclass", num_classes=num_classes, average='none')
        self.recall = tm.Recall(task="multiclass", num_classes=num_classes, average='none')
        self.val_precision = tm.Precision(task="multiclass", num_classes=num_classes, average='none')

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)

        # cls_vector = output.logits[:, 0, :]
        # scores = self.linear(cls_vector.to(torch.float16))

        cls_vector = output.last_hidden_state[:, 0, :]
        scores = self.linear(cls_vector)
        return scores

    def training_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask = x
        y_hat = self(input_ids, attention_mask)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask = x
        y_hat = self(input_ids, attention_mask)
        loss = self.criterion(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=1)
        self.compute_metrics(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        return loss

    def on_validation_epoch_end(self):
        self.log_metrics()

    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask = x
        y_hat = self(input_ids, attention_mask)
        loss = self.criterion(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=1)

        return {
            'loss': loss,
            'preds': y_hat,
            'label': y
        }

    def test_step_end(self, outputs):
        y_hat, y = outputs['preds'], outputs['label']
        self.compute_metrics(y_hat, y)
        return outputs

    def test_epoch_end(self, outputs):
        self.log_metrics()
        self.results = outputs

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        x, y = batch
        input_ids, attention_mask = x
        y_hat = self(input_ids, attention_mask)
        y_hat = torch.softmax(y_hat, dim=1)
        return y_hat

    def compute_metrics(self, y_hat, y):
        self.acc(y_hat, y)
        self.f1(y_hat, y)
        self.recall(y_hat, y)
        self.val_precision(y_hat, y)

    def log_metrics(self):
        self.log('acc', self.acc.compute())
        f1_negative, f1_coref, f1_hypernym, f1_hyponym = self.f1.compute()
        recall_negative, recall_coref, recall_hypernym, recall_hyponym = self.recall.compute()
        precision_negative, precision_coref, precision_hypernym, precision_hyponym = self.val_precision.compute()
        self.log('f1_coref', f1_coref)
        self.log('recall_coref', recall_coref)
        self.log('precision_coref', precision_coref)
        self.log('f1_hypernym', f1_hypernym)
        self.log('recall_hypernym', recall_hypernym)
        self.log('precision_hypernym', precision_hypernym)
        self.log('f1_hyponym', f1_hyponym)
        self.log('recall_hyponym', recall_hyponym)
        self.log('precision_hyponym', precision_hyponym)

    def configure_optimizers(self):
        return FusedAdam(self.parameters(), lr=self.config['model']['lr'])
        # return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])


    def get_global_attention(self, input_ids):
        global_attention_mask = torch.zeros(input_ids.shape)
        global_attention_mask[:, 0] = 1  # global attention to the CLS token
        start = torch.nonzero(input_ids == self.start)
        end = torch.nonzero(input_ids == self.end)
        start_def = torch.nonzero(input_ids == self.start_def)
        end_def = torch.nonzero(input_ids == self.end_def)
        if self.cdlm:
            doc_start = torch.nonzero(input_ids == self.doc_start)
            doc_end = torch.nonzero(input_ids == self.doc_end)
            globs = torch.cat((start, end, doc_start, doc_end))
        else:
            globs = torch.cat((start, end, start_def, end_def))
            # globs = torch.cat((start, end))

        value = torch.ones(globs.shape[0])
        global_attention_mask.index_put_(tuple(globs.t()), value)
        return global_attention_mask

    def tokenize_batch(self, batch):
        inputs, labels = zip(*batch)
        tokens = self.tokenizer(list(inputs), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        labels = torch.stack(labels)

        return (input_ids, attention_mask), labels