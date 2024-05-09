import torch
from transformers import Phi3ForSequenceClassification, AutoTokenizer, pipeline
from peft import PeftModel
from definition_handler.process_data import DatasetsHandler
import re
from tqdm import tqdm
import pickle

base_model = "microsoft/Phi-3-mini-4k-instruct"
adapter = "/cs/labs/tomhope/forer11/SciCo_Retrivel/phi3_classification/no_def/model"
device_map = 'auto'
max_seq_length = 2048  # None
output_dir = '/cs/labs/tomhope/forer11/SciCo_Retrivel/phi3_classification/no_def/results'

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
"""

phi3_instruct_prompt_with_def = """<|user|>
You are a helpful AI assistant. you will get two scientific texts that has a term surrounded by a relevant context and a definition of those terms that was generated in regard for the context. Read the terms with their context and definitions and define the correct relationship between the two terms as follows:
1 - Co-referring terms: Both term1 and term2 refer to the same underlying concept or entity.
2 - Parent concept: Term1 represents a broader category or concept that encompasses term2, such that mentioning term1 implicitly invokes term2.
3 - Child concept: The inverse of a parent concept relation. Term1 is a specific instance or subset of the broader concept represented by term2, such that mentioning term2 implicitly invokes term1.
0 - None of the above: Term1 and term2 are not co-referring, and do not have a parent-child or child-parent relation.

here are the terms and their context:
first term: {term1}
first term definition: {term1_def}
first term context: {term1_text}

second term: {term2}
second term definition: {term2_def}
second term context: {term2_text}

please select the correct relationship between the two terms from the options above.<|end|>
<|assistant|>
"""


def get_phi3_instruct_prompt(pair, with_def = False, def_dict = None):
    term1_text, term2_text, _ = pair.split('</s>')
    term1 = re.search(r'<m>(.*?)</m>', term1_text).group(1).strip()
    term2 = re.search(r'<m>(.*?)</m>', term2_text).group(1).strip()
    term1_text, term2_text = term1_text.replace('<m> ', '').replace(' </m>', ''), term2_text.replace('<m> ',
                                                                                                     '').replace(
        ' </m>', '')
    if with_def:
        term1_def, term2_def = def_dict[pair[0] + '</s>'], def_dict[pair[1] + '</s>']
        return phi3_instruct_prompt_with_def.format(term1=term1, term1_text=term1_text, term2=term2, term2_text=term2_text, term1_def=term1_def, term2_def=term2_def)

    return phi3_instruct_prompt.format(term1=term1, term1_text=term1_text, term2=term2, term2_text=term2_text)

data = DatasetsHandler(test=True, train=False, dev=False, full_doc=True)

model = Phi3ForSequenceClassification.from_pretrained(
    base_model,
    # quantization_config=bnb_config,
    cache_dir='/cs/labs/tomhope/forer11/cache',
    device_map=device_map,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    num_labels=4,
    torch_dtype=torch.float16,
    # use_cache = False
)
# Load and activate the adapter on top of the base model
model = PeftModel.from_pretrained(model, adapter)

# Merge the adapter with the base model
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, add_prefix_space=True)
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'left'


test_prompts = [{'text': get_phi3_instruct_prompt(data.test_dataset.pairs[i]), 'label': data.test_dataset.labels[i]} for i in range(len(data.test_dataset.pairs))]

results = {}

with torch.no_grad():
    for i, example in enumerate(tqdm(test_prompts)):
        input = tokenizer(example['text'], return_tensors="pt", truncation=True, padding=True, max_length=max_seq_length).to('cuda')
        output = model.forward(**input).logits
        text, label = example['text'], example['label']
        results[data.test_dataset.pairs[i]] = output
        if i % 15000 == 0:
            print(f'Processed {i} examples')
            with open(f'{output_dir}/results_{i}_batches.pickle', 'wb') as file:
                pickle.dump(results, file)

print(f'Processed all examples')
with open(f'{output_dir}/final_results.pickle', 'wb') as file:
    pickle.dump(results, file)