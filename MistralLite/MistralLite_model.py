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

# prompt = (f'<|prompter|>'
#           'You are given 2 texts below seperated by </sent>, in each text there is a scientific term inside <m> </m> and a context for said term\n'
#           'those are the possible hierarchies between Term A and Term B: '
#           '0 - No relation, no hierarchical connection for example: "Systems Network Architecture" and "AI Network Architecture"'
#           '1 - Same level, co-referring terms (for example: "self-driving cars" and "autonomous vehicles")'
#           '2 - Term A is a parent concept of term B for example: "Information Extraction" is a parent concept of "Definition extraction"'
#           '3 - Term A is a child concept of Term B for example: "image synthesis task" is a child concept of "computer vision\n"'
#           # 'with only a 4 index confidence score vector where each index represents a hierarchy level,'
#           # 'answer shortly like the next example: if 0 is the index of the hierarchy level the model has the most confidence in then the output will be: [0.8,0.1,0.05,0.05]\n'
#           "here are examples of sentences and the model logits representing the confidence with each hierarchy level:\n"
#           '=== SENTENCES INPUT ===\n'
#           "We use ResNet - 50 [ reference ] to implement the <m> CNN regressor </m> . </sent> <m> Artificial "
#           "neural network Artificial Neural Networks </m> ( ANN ) has the characteristics of adaptive , "
#           "self-organization and self-learning . </sent>\n"
#           "===POSSIBLE MODEL OUTPUT ===\n"
#           "[3.0,0.5,−1.0,−2.0]\n"
#           '=== SENTENCES INPUT ===\n'
#           "The selected MFCC features were then used to train several <m> ANN Multi-Layer Perceptron ( MLP </m> ) . "
#           "</sent> As introduced in section [ reference ] , BaseModel follows the Embedding & <m> MLP architecture "
#           "</m> and is the base of most of subsequently developed deep networks for CTR modeling . </sent>\n"
#           "===POSSIBLE MODEL OUTPUT ===\n"
#           "[−0.5,3.0,0.5,−1.0]\n"
#           '=== SENTENCES INPUT ===\n'
#           "A CNN consists of two layers : a convolutional layer , followed by a <m> subsampling layer </m> . </sent> "
#           "subsection : Results of Hierarchical Subsampling We first demonstrate the results of the <m> hierarchical "
#           "subsampling recurrent network </m> , which is the key to speed up our experiments . </sent>\n"
#           "===POSSIBLE MODEL OUTPUT ===\n"
#           "[−1.5,2.7,3.1,−0.8]\n"
#           '=== SENTENCES INPUT ===\n'
#           "Although this approach is the traditional architecture choice for <m> text classification CNNs </m> , "
#           "it introduces a significant number of parameter in the network . </sent> "
#           "Authorship attribution may be considered as a <m> text categorization problem </m> . </sent>\n"
#           "===POSSIBLE MODEL OUTPUT ===\n"
#           "[−2.0,1.5,−0.7,3.2]\n"
#           "\n\nnow read the following sentences and generate a confidence logit:\n"
#           '=== SENTENCES INPUT ===\n'
#           '{sentences}'
#           "===POSSIBLE MODEL OUTPUT ===\n"
#
#           '</s><|assistant|>')


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
# model = MistralForSequenceClassification.from_pretrained(model_id,
#                                                          torch_dtype=torch.bfloat16,
#                                                          use_flash_attention_2=True,
#                                                          # device_map="auto",
#                                                          cache_dir='/cs/labs/tomhope/forer11/cache',
#                                                          output_hidden_states=True,
#                                                          num_labels=4)
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,
#     r=16,  # dimension of the updated matrices
#     lora_alpha=64,  # parameter for scaling
#     target_modules=[
#         "q_proj",
#         "up_proj",
#         "o_proj",
#         "k_proj",
#         "down_proj",
#         "gate_proj",
#         "v_proj",
#         "score"],
#     lora_dropout=0.1,  # dropout probability for layers
#     bias="none",
# )
# model = get_peft_model(model.to("cuda"), peft_config)
# model.print_trainable_parameters()
#
#
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
        self.config = config
        model_id = "amazon/MistralLite"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = MistralForSequenceClassification.from_pretrained(model_id,
                                                                      torch_dtype=torch.bfloat16,
                                                                      use_flash_attention_2=True,
                                                                      # device_map="auto",
                                                                      cache_dir='/cs/labs/tomhope/forer11/cache',
                                                                      num_labels=num_classes)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,  # dimension of the updated matrices
            lora_alpha=64,  # parameter for scaling
            target_modules=[
                "q_proj",
                # "up_proj",
                "o_proj",
                "k_proj",
                # "down_proj",
                # "gate_proj",
                "v_proj"
            ],
            modules_to_save=["score"],
            inference_mode=False,
            lora_dropout=0.1,  # dropout probability for layers
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["bert_model"])
        self.tokenizer.add_tokens('<m>', special_tokens=True)
        self.tokenizer.add_tokens('</m>', special_tokens=True)
        # self.tokenizer.add_tokens('<def>', special_tokens=True)
        # self.tokenizer.add_tokens('</def>', special_tokens=True)
        self.start = self.tokenizer.convert_tokens_to_ids('<m>')
        self.end = self.tokenizer.convert_tokens_to_ids('</m>')
        # self.start_def = self.tokenizer.convert_tokens_to_ids('<def>')
        # self.end_def = self.tokenizer.convert_tokens_to_ids('</def>')
        self.sep = self.tokenizer.convert_tokens_to_ids('</s>')
        self.tokenizer.padding_side = "left"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
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
