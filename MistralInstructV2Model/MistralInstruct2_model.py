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

# os.environ['CONSUL_HTTP_ADDR'] = ''
# os.environ['BNB_CUDA_VERSION'] = '117'
# os.environ['CUDA_HOME'] = '/cs/labs/tomhope/forer11/cuda117'
# os.environ['LD_LIBRARY_PATH'] = '/cs/labs/tomhope/forer11/cuda117/lib64'
# os.environ['PATH'] = '/cs/labs/tomhope/forer11/cuda117/lib64/bin'
os.environ['RDMAV_FORK_SAFE'] = '1'
from peft import get_peft_model, LoraConfig, TaskType
import re

sys_msg = """You are a helpful AI assistant, you are an agent capable of reading and understanding scientific text with terms and defining the hierarchy between them. this is the process you should follow:

- Read texts: for each of the two mentions there will be a text with some context for the term. the context is from a scientific paper the term was used in.

- Understand the hierarchy: after reading the texts, looks at those possible hierarchies between the two terms:
0 - No relation, no hierarchical connection (for example the terms: "Systems Network Architecture" and "AI Network Architecture" has no connection)
1 - Same level, co-referring terms (for example the terms: "self-driving cars" and "autonomous vehicles" are co-referring terms)
2 - Term A is a parent concept of term B (for example the term "Information Extraction" is a parent concept of the term "Definition extraction")
3 - Term A is a child concept of Term B (for example the term "image synthesis task" is a child concept of the term "computer vision")

- Classify the hierarchy: after understanding the hierarchy, classify the hierarchy between the two terms using the following classes: {0, 1, 2, 3} like in the examples that follows:

here is an EXAMPLE for a query and a required generated definition:

### START EXAMPLE ###

User: Classify the hierarchy between the terms term A and term B after reading the following texts for each term:

term 1 <m>term A</m> text: example text where term A is mentioned

term 2 <m>term B</m> text: example text where term B is mentioned

Assistant: hierarchy: 0

### END EXAMPLE ###

Let's get started. The users query is as follows:
"""

def get_terms_and_texts(sentences):
    matches = re.findall(pattern, sentences)

    if len(matches) >= 2:
        term1, term2 = matches[:2]
        term1_text = re.search(f'<m>\s*{re.escape(term1)}\s*</m>(.*?)<m>\s*{re.escape(term2)}\s*</m>', sentences,
                               re.DOTALL).group(1).strip()
        term2_text = re.search(f'<m>\s*{re.escape(term2)}\s*</m>(.*?)$', sentences, re.DOTALL).group(1).strip()
        return term1, term2, term1_text, term2_text


def instructions_query_format(term_a, term_b, term_a_text, term_b_text):
    query = f'Classify the hierarchy between the terms <m>{term_a}</m> and <m>{term_b}</m> after reading the following texts for each term:\n\nterm 1 <m>{term_a}</m> text: {term_a_text}\n\nterm 2 <m>{term_b}</m> text: {term_b_text}\n\n'
    return query


def instruction_format(sys_message, query):
    # note, don't "</s>" to the end
    return f'<s> [INST] {sys_message} [/INST]\nUser: {query}\nAssistant: hierarchy: '


def get_prompt(sentences):
    term1_text, term2_text, _ = sentences.split('</s>')
    term1 = re.search(r'<m>(.*?)</m>', term1_text).group(1)
    term2 = re.search(r'<m>(.*?)</m>', term2_text).group(1)
    query = instructions_query_format(term1.strip(), term2.strip(), term1_text, term2_text)
    prompt = instruction_format(sys_msg, query)
    return prompt


# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# model = MistralForSequenceClassification.from_pretrained(model_id,
#                                              cache_dir='/cs/labs/tomhope/forer11/Retrieval-augmented-defenition-extractor/cache',
#                                              attn_implementation="flash_attention_2",
#                                              trust_remote_code=True,
#                                              device_map="auto",
#                                              # quantization_config=bnb_config,
#                                              torch_dtype=torch.bfloat16)
# print(model)

class MistralInstruct2CrossEncoder(pl.LightningModule):
    '''
    multiclass classification with labels:
    0 not related
    1 coref
    2. hypernym
    3. neutral (hyponym)
    '''

    def __init__(self, config, num_classes=4):
        super(MistralInstruct2CrossEncoder, self).__init__()
        self.config = config
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        self.model = MistralForSequenceClassification.from_pretrained(model_id,
                                                                      torch_dtype=torch.bfloat16,
                                                                      attn_implementation="flash_attention_2",
                                                                      # quantization_config=bnb_config,
                                                                      cache_dir='/cs/labs/tomhope/forer11/cache',
                                                                      num_labels=num_classes)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,  # dimension of the updated matrices
            lora_alpha=16,  # parameter for scaling
            target_modules=[
                "q_proj",
                "up_proj",
                "o_proj",
                "k_proj",
                "down_proj",
                "gate_proj",
                "v_proj"
            ],
            modules_to_save=["score"],
            inference_mode=False,
            lora_dropout=0.05,  # dropout probability for layers
            bias="lora_only",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')

        self.tokenizer.padding_side = "left"
        self.model.config.pad_token_id = self.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.criterion = torch.nn.CrossEntropyLoss()

        self.acc = tm.Accuracy(top_k=1, task="multiclass", num_classes=num_classes)
        self.f1 = tm.F1Score(task="multiclass", num_classes=num_classes, average='none')
        self.recall = tm.Recall(task="multiclass", num_classes=num_classes, average='none')
        self.val_precision = tm.Precision(task="multiclass", num_classes=num_classes, average='none')

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        return output.logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        input_ids, attention_mask = x
        y_hat = self(input_ids, attention_mask)
        loss = self.criterion(y_hat, y)
        self.log('loss', loss, on_step=True, prog_bar=True, rank_zero_only=True)
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
        # return FusedAdam(self.parameters(), lr=self.config['model']['lr'])
        return torch.optim.AdamW(self.parameters(), lr=self.config['model']['lr'])
        # return Lion(self.parameters(), lr=1e-4, weight_decay=1e-2)

    def tokenize_batch(self, batch):
        inputs, labels = zip(*batch)
        inputs = tuple(get_prompt(s) for s in inputs)
        tokens = self.tokenizer(list(inputs), padding=True)
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        labels = torch.stack(labels)

        return (input_ids, attention_mask), labels
