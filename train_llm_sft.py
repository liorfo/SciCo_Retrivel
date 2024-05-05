import wandb
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
from definition_handler.process_data import DatasetsHandler
import pandas as pd
import re
from datasets import Dataset, load_dataset

################################################################################
# LoRA parameters
################################################################################
# LoRA attention dimension
# lora_r = 64
lora_r = 16
# Alpha parameter for LoRA scaling
lora_alpha = 32
# Dropout probability for LoRA layers
lora_dropout = 0.05

################################################################################
# bitsandbytes parameters
################################################################################
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################
# Output directory where the model predictions and checkpoints will be stored
output_dir = '/cs/labs/tomhope/forer11/SciCo_Retrivel/phi3_4k_sft/no_def/full_docs_model'
# Number of training epochs
num_train_epochs = 1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True
# Batch size per GPU for training
per_device_train_batch_size = 2
# Batch size per GPU for evaluation
per_device_eval_batch_size = 4
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
# learning_rate = 2e-4
learning_rate = 2e-4
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = 'paged_adamw_8bit'
# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "cosine"
# Number of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
# Save checkpoint every X updates steps
save_steps = 200
# Log every X updates steps
logging_steps = 10

################################################################################
# SFT parameters
################################################################################
# Maximum sequence length to use
max_seq_length = 1024  # None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = True  # False
# Load the entire model on the GPU 0
# device_map = {"": 0}
# device_map = "auto"
device_map = "cuda"

################################################################################
# start of training
################################################################################

task_msg = """### Task: 
Each of the following scientific texts in the ### Input section has a term surrounded by a relevant context. Read the terms with their context and define the correct relationship between the two terms as follows:
1 - Term A and term B are co-referring terms
2 - Term A is a parent concept of term B
3 - Term A is a child concept of term B
0 - None of the above relations are appropriate
"""

input_msg = """### Input: 
first term: {term1} 
first term context: {term1_text}

second term: {term2}
second term context: {term2_text}
"""

out_prompt = """### Output:
{label}"""


phi3_instruct_prompt = """<|user|>
You are a helpful AI assistant. you will get two scientific texts that has a term surrounded by a relevant context. Read the terms with their context and define the correct relationship between the two terms as follows:
1 - Co-referring terms: Both term1 and term2 refer to the same underlying concept or entity.
2 - Parent concept: Term1 represents a broader category or concept that encompasses term2, such that mentioning term1 implicitly invokes term2.
3 - Child concept: The inverse of a parent concept relation. Term1 is a specific instance or subset of the broader concept represented by term2, such that mentioning term2 implicitly invokes term1.
0 - None of the above: Term1 and term2 are not co-referring, and do not have a parent-child or child-parent relation.

here are the terms and their context:
first term: {term1} 
first term context: {term1_text}

second term: {term2}
second term context: {term2_text}

please select the correct relationship between the two terms from the options above.<|end|>
<|assistant|>
{label}"""

def get_phi3_instruct_prompt(pair, label):
    term1_text, term2_text, _ = pair.split('</s>')
    term1 = re.search(r'<m>(.*?)</m>', term1_text).group(1).strip()
    term2 = re.search(r'<m>(.*?)</m>', term2_text).group(1).strip()
    term1_text, term2_text = term1_text.replace('<m> ', '').replace(' </m>', ''), term2_text.replace('<m> ','').replace(' </m>', '')
    return phi3_instruct_prompt.format(term1=term1, term1_text=term1_text, term2=term2, term2_text=term2_text, label=label)



def get_task_prompt(with_def=False):
    if with_def:
        # TODO return def appropriate task prompt
        return task_msg
    return task_msg


def get_input_prompt(pair, with_def=False):
    term1_text, term2_text, _ = pair.split('</s>')
    term1 = re.search(r'<m>(.*?)</m>', term1_text).group(1).strip()
    term2 = re.search(r'<m>(.*?)</m>', term2_text).group(1).strip()
    term1_text, term2_text = term1_text.replace('<m> ', '').replace(' </m>', ''), term2_text.replace('<m> ',
                                                                                                     '').replace(
        ' </m>', '')
    if with_def:
        # TODO return def appropriate input prompt
        return input_msg
    return input_msg.format(term1=term1, term1_text=term1_text, term2=term2, term2_text=term2_text)


def get_output_prompt(label):
    return out_prompt.format(label=label)


def get_format_prompt(pair, label, with_def=False):
    return get_task_prompt(with_def) + '\n' + get_input_prompt(pair, with_def) + '\n' + get_output_prompt(label)


def format_prompts_fn(example):
    return example['text']


# wandb.login(key='8b5bf778b37dfdd547cbb6f4c1340c3b08ddab75')

base_model = "microsoft/Phi-3-mini-4k-instruct"

# base_model = "alpindale/Mistral-7B-v0.2-hf"

# base_model = "mistralai/Mistral-7B-v0.1"


data = DatasetsHandler(test=False, train=True, dev=True, full_doc=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,  # Activates 4-bit precision loading
    bnb_4bit_quant_type=bnb_4bit_quant_type,  # nf4
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,  # float16
    bnb_4bit_use_double_quant=use_nested_quant,  # False
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # quantization_config=bnb_config,
    cache_dir='/cs/labs/tomhope/forer11/cache',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    device_map=device_map
)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'


# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"]
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="wandb",
)

# train_prompts = [{'text': get_format_prompt(data.train_dataset.pairs[i], data.train_dataset.natural_labels[i])} for i in
#                  range(len(data.train_dataset))]
train_prompts = [{'text': get_phi3_instruct_prompt(data.train_dataset.pairs[i], data.train_dataset.natural_labels[i])} for i in
                 range(len(data.train_dataset))]

# train_data = pd.DataFrame(train_prompts, columns=['text'])
# dataset = Dataset.from_pandas(train_data)

dataset = Dataset.from_list(train_prompts)

# dataset_name = "ai-bites/databricks-mini"
# dataset111 = load_dataset(dataset_name, split="train[0:1000]")

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    # formatting_func=format_prompts_fn,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# trainer.train(resume_from_checkpoint='/cs/labs/tomhope/forer11/SciCo_Retrivel/mistral_v2_sfttrainer/no_def/model/checkpoint-10')
trainer.train()
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
wandb.finish()
