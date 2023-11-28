from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, MistralForSequenceClassification
import transformers
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Any, Optional
import torchmetrics as tm
from deepspeed.ops.adam import FusedAdam
import flash_attn
from transformers.utils import (
    is_flash_attn_2_available,
    is_torch_available,
)
import importlib.metadata
from typing import Any, Tuple, Union
from packaging import version
from lion_pytorch import Lion

os.environ['CUDA_HOME'] = '/usr/local/nvidia/cuda/11.7'
os.environ['RDMAV_FORK_SAFE'] = '1'


# def convert_inputs_into_prompts(inputs):
#     pass


# def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
#     # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
#     package_exists = importlib.util.find_spec(pkg_name) is not None
#     package_version = "N/A"
#     if package_exists:
#         try:
#             package_version = importlib.metadata.version(pkg_name)
#             package_exists = True
#         except importlib.metadata.PackageNotFoundError:
#             package_exists = False
#     if return_version:
#         return package_exists, package_version
#     else:
#         return package_exists
#
# print(torch.version.cuda)
# print(is_flash_attn_2_available())
# print(_is_package_available("flash_attn") and version.parse(
#     importlib.metadata.version("flash_attn")))
# print(torch.cuda.is_available())

prompt = (f'<|prompter|>'
          'You are given 2 texts below seperated by </s></s>, in each text there is a scientific term inside <m> </m> and a context for said term. please read them carefully and answer the follow up question.\n'
          '=== BEGIN ===\n'
          '{sentences}'
          '\n=== END OF SENTENCES ===\n'
          'Please define the hierarchy between Term A and Term B using the following levels: '
          '0 - No relation, no hierarchical connection for example: "Systems Network Architecture" and "AI Network Architecture"'
          '1 - Same level, co-referring terms (for example: "self-driving cars" and "autonomous vehicles")'
          '2 - Term A is a parent concept of term B for example: "Information Extraction" is a parent concept of "Definition extraction"'
          '3 - Term A is a child concept of Term B for example: "image synthesis task" is a child concept of "computer vision"'
          'answer shortly with only the number of the correct hierarchy level\n'
          '</s><|assistant|>')


# model_id = "amazon/MistralLite"
#
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id,
#                                              torch_dtype=torch.bfloat16,
#                                              use_flash_attention_2=True,
#                                              device_map="auto",
#                                              cache_dir='/cs/labs/tomhope/forer11/cache')
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
# )
# sequences = pipeline(
#     prompt.format(sentences='We consider the problem of developing a <m> linear transformation process </m> to compensate for range-dependent bistatic clutter spectral dispersion . </s></s>The simple <m> linear projection </m> makes the method easy to interpret , while the visualization task is made well-defined by the novel information retrieval criterion . </s>'),
#     max_new_tokens=400,
#     do_sample=False,
#     return_full_text=False,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
# )
# for seq in sequences:
#     print(f"{seq['generated_text']}")


class MistarlLightCrossEncoder(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related
    1 coref
    2. hypernym
    3. neutral (hyponym)
    '''

    def __init__(self, config, num_classes=4):
        super(MistarlLightCrossEncoder, self).__init__()
        self.cdlm = 'cdlm' in config["model"]["bert_model"].lower()
        self.long = True if 'longformer' in config["model"]["bert_model"] or self.cdlm else False
        self.config = config

        model_id = "amazon/MistralLite"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = MistralForSequenceClassification.from_pretrained(model_id,
                                                                      torch_dtype=torch.bfloat16,
                                                                      use_flash_attention_2=True,
                                                                      # device_map="auto",
                                                                      cache_dir='/cs/labs/tomhope/forer11/cache',
                                                                      output_hidden_states=True,
                                                                      num_labels=num_classes)

        # self.model = self.model.to('cuda')

        # self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
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

        # self.model = AutoModel.from_pretrained(config["model"]["bert_model"], add_pooling_layer=False,
        #                                        cache_dir='/cs/labs/tomhope/forer11/cache', attention_window=512)
        self.model.resize_token_embeddings(len(self.tokenizer))
        # self.linear = nn.Linear(self.model.config.hidden_size, num_classes)
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

        # cls_vector = output.last_hidden_state[:, 0, :]
        # scores = self.linear(cls_vector)
        return output.logits

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
        # return torch.optim.SGD(self.parameters(), lr=self.config['model']['lr'])
        # return Lion(self.parameters(), lr=1e-4, weight_decay=1e-2)

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
        inputs = tuple(prompt.format(sentences=s) for s in inputs)
        tokens = self.tokenizer(list(inputs), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        labels = torch.stack(labels)

        return (input_ids, attention_mask), labels
