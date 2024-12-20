# from transformers import *
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Any, Optional
import numpy as np
import torchmetrics as tm
import pickle
from SciCo_Retrivel.MistralLite.MistralLite_model import MistarlLightCrossEncoder
from SciCo_Retrivel.MistralInstructV2Model.MistralInstruct2_model import MistralInstruct2CrossEncoder

def get_gpt_score(inputs, labels, sentences_to_score):
    try:
        preds = [int(sentences_to_score[sent]) for sent in inputs]
    except:
        preds = [0 for _ in range(len(labels))]
    # chance = np.random.uniform(0, 1)
    # ind = np.random.randint(0, 4) if chance < 0.1 else label
    l = [[0, 0, 0, 0] for _ in range(len(inputs))]
    for i, pred in enumerate(preds):
        if pred < 0 or pred > 3:
            pred = 0
        l[i][pred] = 100
    return l

def get_classification_score(inputs, labels, sentences_to_score):
    # only works with batch size of 1 for now
    scores = sentences_to_score[inputs[0]]
    # batch_response = torch.cat([tensor.unsqueeze(0) for tensor in scores], dim=0)
    return scores


def get_global_attention(input_ids, start_token, end_token):
    global_attention_mask = torch.zeros(input_ids.shape)
    global_attention_mask[:, 0] = 1  # global attention to the CLS token
    start = torch.nonzero(input_ids == start_token)
    end = torch.nonzero(input_ids == end_token)
    globs = torch.cat((start, end))
    value = torch.ones(globs.shape[0])
    global_attention_mask.index_put_(tuple(globs.t()), value)
    return global_attention_mask


class MulticlassModel:
    def __init__(self):
        super(MulticlassModel, self).__init__()

    @classmethod
    def get_model(cls, name, config, is_gpt=False):
        if name == 'multiclass':
            if is_gpt:
                return MulticlassCrossEncoderGPT(config, num_classes=4)
            # return LlamaMulticlassCrossEncoder(config, num_classes=4)
            # return MistarlLightCrossEncoder(config, num_classes=4)
            # return MistralInstruct2CrossEncoder(config, num_classes=4)
            return MulticlassCrossEncoder(config, num_classes=4)
        elif name == 'coref':
            return BinaryCorefCrossEncoder(config)
        elif name == 'hypernym':
            return HypernymCrossEncoder(config)


class MulticlassCrossEncoderGPT(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related
    1 coref
    2. hypernym
    3. neutral (hyponym)
    '''

    def __init__(self, config, num_classes=4):
        super(MulticlassCrossEncoderGPT, self).__init__()
        self.acc = tm.Accuracy(top_k=1, task="multiclass", num_classes=num_classes)
        self.f1 = tm.F1Score(task="multiclass", num_classes=num_classes, average='none')
        self.recall = tm.Recall(task="multiclass", num_classes=num_classes, average='none')
        self.val_precision = tm.Precision(task="multiclass", num_classes=num_classes, average='none')

        with open("/cs/labs/tomhope/forer11/SciCo_Retrivel/mistral_1_classification/with_gpt_4_def/results/merged_final_results.pickle","rb") as file:
            self.sentences_to_score = pickle.load(file)

    def forward(self, inputs, labels):
        # scores = get_gpt_score(inputs, labels, self.sentences_to_score)
        scores = get_classification_score(inputs, labels, self.sentences_to_score)
        # scores = torch.tensor(scores, dtype=torch.float)
        return scores

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        self.log_metrics()

    def test_step(self, batch, batch_idx):
        pass

    def test_step_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        x, y = batch
        y_hat = self(x, y)
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
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])

    def tokenize_batch(self, batch):
        inputs, labels = zip(*batch)
        labels = np.array(labels)

        return np.array(inputs), labels


class MulticlassCrossEncoder(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related
    1 coref
    2. hypernym
    3. neutral (hyponym)
    '''

    def __init__(self, config, num_classes=4):
        super(MulticlassCrossEncoder, self).__init__()
        self.cdlm = 'cdlm' in config["model"]["bert_model"].lower()
        self.long = True if 'longformer' in config["model"]["bert_model"] or self.cdlm else False
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        # self.tokenizer.add_tokens('<def>', special_tokens=True)
        # self.tokenizer.add_tokens('</def>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')
        # self.start_def = self.tokenizer.convert_tokens_to_ids('<def>')
        # self.end_def = self.tokenizer.convert_tokens_to_ids('</def>')
        self.sep = self.tokenizer.convert_tokens_to_ids('</s>')
        self.doc_start = self.tokenizer.convert_tokens_to_ids('<doc-s>') if self.cdlm else None
        self.doc_end = self.tokenizer.convert_tokens_to_ids('</doc-s>') if self.cdlm else None

        self.model = AutoModel.from_pretrained(config["model"]["bert_model"], add_pooling_layer=False,
                                               # attention_window=768,
                                               cache_dir='/cs/labs/tomhope/forer11/cache')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.acc = tm.Accuracy(top_k=1, task="multiclass", num_classes=num_classes)
        self.f1 = tm.F1Score(task="multiclass", num_classes=num_classes, average='none')
        self.recall = tm.Recall(task="multiclass", num_classes=num_classes, average='none')
        self.val_precision = tm.Precision(task="multiclass", num_classes=num_classes, average='none')

    def forward(self, input_ids, attention_mask, global_attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask,
                            global_attention_mask=global_attention_mask)
        cls_vector = output.last_hidden_state[:, 0, :]
        scores = self.linear(cls_vector)
        return scores

    def training_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        # self.log('loss', loss, on_step=True, prog_bar=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=1)
        self.compute_metrics(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        return loss

    def on_validation_epoch_end(self):
        self.log_metrics()

    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
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
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
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
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])

    def get_global_attention(self, input_ids):
        global_attention_mask = torch.zeros(input_ids.shape)
        global_attention_mask[:, 0] = 1  # global attention to the CLS token
        start = torch.nonzero(input_ids == self.start)
        end = torch.nonzero(input_ids == self.end)
        # start_def = torch.nonzero(input_ids == self.start_def)
        # end_def = torch.nonzero(input_ids == self.end_def)
        if self.cdlm:
            doc_start = torch.nonzero(input_ids == self.doc_start)
            doc_end = torch.nonzero(input_ids == self.doc_end)
            globs = torch.cat((start, end, doc_start, doc_end))
        else:
            # globs = torch.cat((start, end, start_def, end_def))
            globs = torch.cat((start, end))

        value = torch.ones(globs.shape[0])
        global_attention_mask.index_put_(tuple(globs.t()), value)
        return global_attention_mask

    def tokenize_batch(self, batch):
        inputs, labels = zip(*batch)
        tokens = self.tokenizer(list(inputs), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        global_attention_mask = self.get_global_attention(input_ids)
        labels = torch.stack(labels)

        return (input_ids, attention_mask, global_attention_mask), labels


class BinaryCorefCrossEncoder(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related, hypernyn, or hyponym
    1 coref
    '''

    def __init__(self, config):
        super(BinaryCorefCrossEncoder, self).__init__()
        self.long = True if 'longformer' in config["model"]["bert_model"] else False
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')

        self.model = AutoModel.from_pretrained(config["model"]["bert_model"], add_pooling_layer=False)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.acc = pl.metrics.Accuracy()
        self.f1 = pl.metrics.F1(num_classes=1)
        self.recall = pl.metrics.Recall(num_classes=1)
        self.val_precision = pl.metrics.Precision(num_classes=1)

    def forward(self, input_ids, attention_mask, global_attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask,
                            global_attention_mask=global_attention_mask)
        cls_vector = output.last_hidden_state[:, 0, :]
        scores = self.linear(cls_vector)
        return scores.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        y_hat = torch.sigmoid(y_hat)
        self.compute_metrics(y_hat, y.to(torch.int))
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def validation_epoch_end(self, outputs):
        self.log_metrics()

    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)

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
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        y_hat = torch.sigmoid(y_hat)
        return y_hat

    def compute_metrics(self, y_hat, y):
        self.acc(y_hat, y)
        self.f1(y_hat, y)
        self.recall(y_hat, y)
        self.val_precision(y_hat, y)

    def log_metrics(self):
        self.log('acc', self.acc.compute())
        self.log('f1', self.f1.compute())
        self.log('recall', self.recall.compute())
        self.log('precision', self.val_precision.compute())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])

    def tokenize_batch(self, batch):
        inputs, labels = zip(*batch)
        tokens = self.tokenizer(list(inputs), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        global_attention_mask = get_global_attention(input_ids, self.start, self.end)
        labels = torch.stack(labels)

        return (input_ids, attention_mask, global_attention_mask), labels


class HypernymCrossEncoder(pl.LightningModule):
    '''
        multiclass classification with labels:
        0 not related or coref
        1. hypernym
        2. hyponym
        '''

    def __init__(self, config, num_classes=3):
        super(HypernymCrossEncoder, self).__init__()
        self.long = True if 'longformer' in config["model"]["bert_model"] else False
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')

        self.model = AutoModel.from_pretrained(config["model"]["bert_model"], add_pooling_layer=False)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.acc = pl.metrics.Accuracy(top_k=1)
        self.f1 = pl.metrics.F1(num_classes=num_classes, average='none')
        self.recall = pl.metrics.Recall(num_classes=num_classes, average='none')
        self.val_precision = pl.metrics.Precision(num_classes=num_classes, average='none')

    def forward(self, input_ids, attention_mask, global_attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask,
                            global_attention_mask=global_attention_mask)
        cls_vector = output.last_hidden_state[:, 0, :]
        scores = self.linear(cls_vector)
        return scores

    def training_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        loss = self.criterion(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=1)
        self.compute_metrics(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_epoch_end(self, outputs):
        self.log_metrics()

    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
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
        input_ids, attention_mask, global_attention_mask = x
        y_hat = self(input_ids, attention_mask, global_attention_mask)
        y_hat = torch.softmax(y_hat, dim=1)
        return y_hat

    def compute_metrics(self, y_hat, y):
        self.acc(y_hat, y)
        self.f1(y_hat, y)
        self.recall(y_hat, y)
        self.val_precision(y_hat, y)

    def log_metrics(self):
        self.log('acc', self.acc.compute())
        f1_negative, f1_hypernym, f1_hyponym = self.f1.compute()
        recall_negative, recall_hypernym, recall_hyponym = self.recall.compute()
        precision_negative, precision_hypernym, precision_hyponym = self.val_precision.compute()
        self.log('f1_hypernym', f1_hypernym)
        self.log('recall_hypernym', recall_hypernym)
        self.log('precision_hypernym', precision_hypernym)
        self.log('f1_hyponym', f1_hyponym)
        self.log('recall_hyponym', recall_hyponym)
        self.log('precision_hyponym', precision_hyponym)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])

    def tokenize_batch(self, batch):
        inputs, labels = zip(*batch)
        tokens = self.tokenizer(list(inputs), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        global_attention_mask = get_global_attention(input_ids, self.start, self.end)
        labels = torch.stack(labels)

        return (input_ids, attention_mask, global_attention_mask), labels


class MulticlassBiEncoder(pl.LightningModule):
    def __init__(self, config, num_classes=4):
        super(MulticlassBiEncoder, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.long = 'longformer' in config['model']['bert_model']

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')

        self.model = AutoModel.from_pretrained(config["model"]["bert_model"], add_pooling_layer=False)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear = nn.Linear(self.model.config.hidden_size * 2, num_classes)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.acc = pl.metrics.Accuracy(top_k=1)
        self.f1 = pl.metrics.F1(num_classes=num_classes, average='none')
        self.recall = pl.metrics.Recall(num_classes=num_classes, average='none')
        self.val_precision = pl.metrics.Precision(num_classes=num_classes, average='none')

    def get_cls_token(self, mention):
        input_ids, attention_mask, global_attention_mask = mention
        if self.long:
            output = self.model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        else:
            output = self.model(input_ids, attention_mask=attention_mask)

        return output.last_hidden_state[:, 0, :]

    def forward(self, first, second):
        cls_1 = self.get_cls_token(first)
        cls_2 = self.get_cls_token(second)

        input_vec = torch.cat((cls_1, cls_2), dim=1)
        scores = self.linear(input_vec)
        return scores

    def training_step(self, batch, batch_idx):
        m1, m2, y = batch
        y_hat = self(m1, m2)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        m1, m2, y = batch
        y_hat = self(m1, m2)
        loss = self.criterion(y_hat, y)
        self.compute_metrics(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_epoch_end(self, outputs):
        self.log_metrics()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        m1, m2, y = batch
        y_hat = self(m1, m2)
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
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])

    def tokenize_mention(self, mentions):
        tokens = self.tokenizer(list(mentions), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])

        if self.long:
            global_attention_mask = get_global_attention(input_ids, self.start, self.end)
        else:
            global_attention_mask = torch.tensor([])

        return input_ids, attention_mask, global_attention_mask

    def tokenize_batch(self, batch):
        first, second, labels = zip(*batch)
        m1 = self.tokenize_mention(first)
        m2 = self.tokenize_mention(second)
        labels = torch.stack(labels)

        return m1, m2, labels
