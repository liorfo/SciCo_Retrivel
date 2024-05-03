from peft import LoraConfig, PeftModel
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
from definition_handler.process_data import DatasetsHandler
import re
import transformers

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import pickle

# model_name = "alpindale/Mistral-7B-v0.2-hf"
# model_dir = "/cs/labs/tomhope/forer11/SciCo_Retrivel/mistral_v2_sfttrainer/no_def/merged_model"

model_name = "microsoft/Phi-3-mini-4k-instruct"
model_dir = "/cs/labs/tomhope/forer11/SciCo_Retrivel/phi3_4k_sft/no_def/merged_model"

batch_size = 5

# ===========================================================================================================================================
task_def_msg = """### Task: 
Each of the following scientific texts in the ### Input section has a term surrounded by a relevant context and a definition. Read the terms with their context and their definition and output the correct relationship between the two terms as follows:
1 - Term A and term B are co-referring terms
2 - Term A is a parent concept of term B
3 - Term A is a child concept of term B
0 - None of the above relations are appropriate
"""

task_msg = """### Task: 
Each of the following scientific texts in the ### Input section has a term surrounded by a relevant context. Read the terms with their context and define the correct relationship between the two terms as follows:
1 - Term A and term B are co-referring terms
2 - Term A is a parent concept of term B
3 - Term A is a child concept of term B
0 - None of the above relations are appropriate
"""

input_def_msg = """### Input: 
first term: {term1} 
first term context: {term1_text}
first term definition: {term1_def}

second term: {term2}
second term context: {term2_text}
second term definition: {term2_def}
"""

input_msg = """### Input: 
first term: {term1} 
first term context: {term1_text}

second term: {term2}
second term context: {term2_text}
"""

out_prompt = """### Output: """

def get_task_prompt(with_def = False):
    if with_def:
        return task_def_msg
    return task_msg



def get_input_prompt(pair, with_def = False, def_dict= None):
    term1_text, term2_text, _ = pair.split('</s>')
    term1 = re.search(r'<m>(.*?)</m>', term1_text).group(1).strip()
    term2 = re.search(r'<m>(.*?)</m>', term2_text).group(1).strip()
    term1_text, term2_text = term1_text.replace('<m> ', '').replace(' </m>', ''), term2_text.replace('<m> ', '').replace(' </m>', '')
    if with_def:
        term1_def = def_dict[pair.split('</s>')[0] + '</s>']
        term2_def = def_dict[pair.split('</s>')[1] + '</s>']
        return input_def_msg.format(term1=term1, term1_text=term1_text, term1_def=term1_def, term2=term2, term2_text=term2_text, term2_def=term2_def)
    return input_msg.format(term1=term1, term1_text=term1_text, term2=term2, term2_text=term2_text)

def get_output_prompt():
    return out_prompt


def get_format_prompt(pair, with_def = False, def_dict= None):
    return get_task_prompt(with_def) + '\n' + get_input_prompt(pair, with_def, def_dict) + '\n' + get_output_prompt()

def format_prompts_fn(example):
    return example['text']

phi3_instruct_prompt = """<|user|>
You are a helpful AI assistant. you will get two scientific texts that has a term surrounded by a relevant context. Read the terms with their context and define the correct relationship between the two terms as follows:
1 - Term A and term B are co-referring terms
2 - Term A is a parent concept of term B
3 - Term A is a child concept of term B
0 - None of the above relations are appropriate<|end|>

here are the terms and their context:
first term: {term1} 
first term context: {term1_text}

second term: {term2}
second term context: {term2_text}<|end|>

please select the correct relationship between the two terms from the options above.
<|assistant|>
{label}"""

def get_phi3_instruct_prompt(pair):
    term1_text, term2_text, _ = pair.split('</s>')
    term1 = re.search(r'<m>(.*?)</m>', term1_text).group(1).strip()
    term2 = re.search(r'<m>(.*?)</m>', term2_text).group(1).strip()
    term1_text, term2_text = term1_text.replace('<m> ', '').replace(' </m>', ''), term2_text.replace('<m> ','').replace(' </m>', '')
    return phi3_instruct_prompt.format(term1=term1, term1_text=term1_text, term2=term2, term2_text=term2_text)



# ===========================================================================================================================================


def save_merged_model():
    #Load the base model with default precision
    adapter = "/cs/labs/tomhope/forer11/SciCo_Retrivel/phi3_4k_sft/no_def/model"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir='/cs/labs/tomhope/forer11/cache',
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2")

    #Load and activate the adapter on top of the base model
    model = PeftModel.from_pretrained(model, adapter)

    #Merge the adapter with the base model
    model = model.merge_and_unload()

    #Save the merged model in a directory in the safetensors format
    model.save_pretrained(model_dir, safe_serialization=True)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit= True, # Activates 4-bit precision loading
#     bnb_4bit_quant_type='nf4', # nf4
#     bnb_4bit_compute_dtype=torch.float16, # float16
#     bnb_4bit_use_double_quant=False, # False
# )
#
# model = AutoModelForCausalLM.from_pretrained('/cs/labs/tomhope/forer11/SciCo_Retrivel/mistral_v2_sfttrainer/no_def/merged_model',
#                                              cache_dir='/cs/labs/tomhope/forer11/cache',
#                                              trust_remote_code=True,
#                                              device_map="auto",
#                                              quantization_config=bnb_config,
#                                              torch_dtype=torch.float16)
#
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
# tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
# tokenizer.padding_side = 'right'
# tokenizer.save_pretrained(model_dir, safe_serialization=True)

data = DatasetsHandler(test=True, train=False, dev=False, should_load_definition=False)

# save_merged_model()

# xxx = [x for x in range(len(data.test_dataset)) if data.test_dataset.natural_labels[x] == '3']
# input_text = get_format_prompt(data.test_dataset.pairs[xxx[777]])
# ground_truth = data.test_dataset.natural_labels[xxx[777]]
#
# generate_text = transformers.pipeline(
#     model=model, tokenizer=tokenizer,
#     device_map="auto",
#     return_full_text=False,  # if using langchain set True
#     task="text-generation",
#     # we pass model parameters here too
#     temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
#     top_p=0.15,  # select from top tokens whose probability add up to 15%
#     top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
#     max_new_tokens=100,  # max number of tokens to generate in the output
#     repetition_penalty=1.1  # if output begins repeating increase
# )
# generate_text.tokenizer.pad_token_id = model.config.eos_token_id
#
# print(generate_text(input_text)[0]['generated_text'].strip())
# print('ground truth: ', ground_truth)

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import time

# Initialize model and cache

model_directory =  model_dir
print("Loading model: " + model_directory)

config = ExLlamaV2Config(model_directory)
config.max_batch_size = batch_size  # Model instance needs to allocate temp buffers to fit the max batch size
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy = True, batch_size = batch_size)
model.load_autosplit(cache)
tokenizer = ExLlamaV2Tokenizer(config)

tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.extended_piece_to_id[tokenizer.unk_token]
tokenizer.padding_side = 'right'

# Initialize generator

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# Generate some text

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.3
settings.top_k = 0
settings.top_p = 0.15
settings.token_repetition_penalty = 1.01
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

# prompt = input_text

max_new_tokens = 1

prompts = [(data.test_dataset.pairs[i], get_phi3_instruct_prompt(data.test_dataset.pairs[i])) for i in range(len(data.test_dataset))]
f_prompts = sorted(prompts, key = len)

batches = [f_prompts[i:i + batch_size] for i in range(0, len(f_prompts), batch_size)]



generator.warmup()
time_begin = time.time()

collected_outputs = {}
with tqdm(total=len(batches)) as pbar:
    for b, batch in enumerate(batches):
        # print(f"Batch {b + 1} of {len(batches)}...")

        batch_sentences, batch_prompts = zip(*batch)
        batch_sentences, batch_prompts = list(batch_sentences), list(batch_prompts)

        outputs = generator.generate_simple(batch_prompts, settings, max_new_tokens, seed = 1234, add_bos = True, completion_only=True)

        trimmed_outputs = [s.strip() for s in outputs]
        for i, output in enumerate(trimmed_outputs):
            collected_outputs[batch_sentences[i]] = output
        if b % 5000 == 0:
            print(f'Processed {b} batches')
            with open(
                    f'/cs/labs/tomhope/forer11/SciCo_Retrivel/phi3_4k_sft/no_def/results/results_after_{b}_batches.pickle',
                    'wb') as file:
                pickle.dump(collected_outputs, file)
        pbar.update(1)


print(f'Processed all batches')
with open(f'/cs/labs/tomhope/forer11/SciCo_Retrivel/phi3_4k_sft/no_def/results/final_results.pickle', 'wb') as file:
    pickle.dump(collected_outputs, file)
time_end = time.time()
time_total = time_end - time_begin

# print("Prompt: ", prompt)
# print()
# print('model output: ', output[len(prompt):].strip())
# print('true output: ', ground_truth)
# print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second")