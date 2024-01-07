import yaml
import argparse
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from typing import Any, Dict, List, Optional
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.vectorstores import FAISS, Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
from tqdm import tqdm
import pickle
import pandas as pd
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
import json
import re
import os

from process_data import DatasetsHandler

instructor_persist_directory = '/cs/labs/tomhope/forer11/unarxive_instructor_embeddings/'


sys_msg = """You are a helpful AI assistant, you are an agent capable of reading and understanding scientific papers and defining scientific terms. here are the steps you should take to give a proper definition:

- Read abstracts: given scientific papers abstracts, please read them carefully with the user's query term in mind but do not mention them in the definition itself.
- Understand abstracts: after reading the abstracts, please understand them and try to extract the most important information from them regarding the user's query term, not all the information is relevant to the definition
- Generate definition: after reading the abstracts, please generate a short definition for the user's query term.

here is an EXAMPLE for a query and a required generated definition:

### START EXAMPLE ###

User: Please generate a short and concise definition for the term ML after reading the following abstracts:

ABSTRACT: example abstract 1

ABSTRACT: example abstract 2

ABSTRACT: example abstract 3

...

ABSTRACT: example abstract n

Assistant: definition: ML is a subset of AI where computers learn patterns from data to make predictions or decisions without explicit programming.

### END EXAMPLE ###

Let's get started. The users query is as follows:
"""

def combine_pickle_files_to_terms_definitions(pickle_paths, processed_abstracts):
    terms_definitions = {}
    for path in pickle_paths:
        with open(path, 'rb') as file:
            terms_definitions.update(pickle.load(file))
    return terms_definitions


def save_terms_definitions_from_pickle_to_json(pickle_paths, processed_abstracts):
    terms_definitions = combine_pickle_files_to_terms_definitions(pickle_paths, processed_abstracts)
    with open('/cs/labs/tomhope/forer11/Retrieval-augmented-defenition-extractor/data/definitions_v2/v2_terms_definitions.json', 'w') as file:
        json.dump(terms_definitions, file)


def get_abstracts_texts_formatted(text, retriever):
    docs = retriever.invoke(text)
    abstracts = [text] + [doc.page_content for doc in docs]
    formatted_query = ''.join([f'ABSTRACT:\n{text}\n' for text in abstracts])
    return formatted_query

def instructions_query_format(abstracts_string, text):
    term = text[0]
    query = f'Please generate a short and concise definition for the term {term} after reading the following abstracts:\n{abstracts_string}'
    return query


def instruction_format(sys_message: str, query: str):
    # note, don't "</s>" to the end
    return f'<s> [INST] {sys_message} [/INST]\nUser: {query}\nAssistant: definition: '


def create_mentions_definitions_from_existing_docs_with_mistral_instruct(terms_dict, retriever):
    print('creating terms_definitions with mistral_instruct...')
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 cache_dir='/cs/labs/tomhope/forer11/Retrieval-augmented-defenition-extractor/cache',
                                                 attn_implementation="flash_attention_2",
                                                 trust_remote_code=True,
                                                 device_map="auto",
                                                 # quantization_config=bnb_config,
                                                 torch_dtype=torch.float16)
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        # temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        # top_p=0.15,  # select from top tokens whose probability add up to 15%
        # top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=200,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # if output begins repeating increase
    )
    generate_text.tokenizer.pad_token_id = model.config.eos_token_id
    terms_definitions = {}
    print('Processing Prompts...')
    if os.path.exists('/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/dev_terms_prompt_dict.json'):
        print('Loading terms_prompt_dict from pickle file...')
        with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/dev_terms_prompt_dict.json', 'rb') as file:
            terms_prompt_dict = json.load(file)
    else:
        print('Creating terms_prompt_dict...')
        terms_prompt_dict = {}
        for term in tqdm(terms_dict):
            text = terms_dict[term]
            abstracts = get_abstracts_texts_formatted(text, retriever)
            query = instructions_query_format(abstracts, term)
            prompt = instruction_format(sys_msg, query)
            terms_prompt_dict[term[1]] = prompt

        with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/dev_terms_prompt_dict.json', 'w') as file:
            json.dump(terms_prompt_dict, file)

    data = pd.DataFrame(list(terms_prompt_dict.items()), columns=['Term', 'Prompt'])
    dataset = Dataset.from_pandas(data)

    print('Generating definitions...')

    for i, out in tqdm(enumerate(generate_text(KeyDataset(dataset, 'Prompt'), batch_size=4)), total=len(dataset)):
        term = dataset[i]['Term']
        definition = out[0]['generated_text'].strip()
        terms_definitions[term] = definition
        if i % 100 == 0:
            print(f'Processed {i} terms')
            with open(
                    f'/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/dev_terms_definitions_until_{i}.pickle',
                    'wb') as file:
                # Dump the dictionary into the file using pickle.dump()
                pickle.dump(terms_definitions, file)

    print('Saving terms_definitions to pickle file...')
    with open(
            '/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/dev_terms_definitions_final.pickle',
            'wb') as file:
        # Dump the dictionary into the file using pickle.dump()
        pickle.dump(terms_definitions, file)


def embed_and_store(texts=[], load=True, embedding_type='instructor', persist_directory=instructor_persist_directory):
    print('Created Embeddings')

    embedding = get_embeddings_model('hkunlp/instructor-xl', embedding_type, persist_directory)

    if load:
        print(f'loading Vector embeddings from {persist_directory}...')
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        print(f'creating Vector embeddings to {persist_directory}...')
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embedding,
                                         persist_directory=persist_directory)
    return vectordb


def process_abstracts_to_docs(abstracts):
    formatted_docs = []
    for doi in abstracts:
        # print(f'reading {root + file}...')
        page_content = abstracts[doi]
        # Create an instance of Document with content and metadata
        metadata = {'doi': doi}
        formatted_docs.append(Document(page_content=page_content, metadata=metadata))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(formatted_docs)
    return texts


def get_embeddings_model(embeddings_model_name, embedding_type, cache_folder):
    if embedding_type == 'instructor':
        return get_instructor_embeddings(embeddings_model_name, cache_folder)
    elif embedding_type == 'HF':
        return get_huggingface_embeddings(embeddings_model_name, cache_folder)
    elif embedding_type == 'AnglE':
        return get_angle_embeddings(embeddings_model_name, cache_folder)
    else:
        return None


def get_instructor_embeddings(embeddings_model_name, cache_folder):
    # the default instruction is: 'Represent the document for retrieval:'
    return HuggingFaceInstructEmbeddings(model_name=embeddings_model_name,
                                         model_kwargs={"device": "cuda"},
                                         cache_folder=cache_folder)


def get_huggingface_embeddings(embeddings_model_name, cache_folder):
    # TODO enable multi GPU support
    return HuggingFaceEmbeddings(model_name=embeddings_model_name,
                                 cache_folder=cache_folder,
                                 model_kwargs={"device": "cuda"},
                                 multi_process=True)


def get_angle_embeddings(embeddings_model_name, cache_folder):
    return UniversalAnglEEmbeddings(model_name=embeddings_model_name,
                                    cache_folder=cache_folder,
                                    model_kwargs={"device": "cuda"})

def get_def_dict_from_json(json_path):
    with open(json_path, 'r') as file:
        terms_definitions = json.load(file)
    return terms_definitions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/multiclass.yaml')
    parser.add_argument('--cache_folder', type=str, default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    cache_folder = args.cache_folder if args.cache_folder != '' else config['cache_folder']

    datasets = DatasetsHandler(config)

    vector_store = embed_and_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    create_mentions_definitions_from_existing_docs_with_mistral_instruct(datasets.dev_dataset.term_context_dict, retriever)

    # with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/train_terms_definitions_final.pickle', 'rb') as file:
    #     yay = pickle.load(file)

    # terms_def = get_def_dict_from_json('/cs/labs/tomhope/forer11/Retrieval-augmented-defenition-extractor/data/definitions_v2/v2_terms_definitions.json')
    # print(len(terms_def))

    print('Done!')
