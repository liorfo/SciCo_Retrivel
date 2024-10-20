# SciCo with definitions retrieval


This repository contains the updated model for Scico with definitions retrieval.

the original paper:

[SciCo: Hierarchical Cross-Document Coreference for Scientific Concepts](https://arxiv.org/abs/2104.08809) \
*Arie Cattan, Sophie Johnson, Daniel S. Weld, Ido Dagan, Iz Beltagy, Doug Downey and Tom Hope*

## Project Description

The goal of this project is to improve the [SciCo model](https://github.com/ariecattan/SciCo/tree/main) by adding 
a definitions retrieval step. The definition retrival was tested and implemented only on the Multiclass unified model, trained on longformer.

Note that this project's code was written including hard-coding of specific local paths, and will require adjustments to run. The code is organized in chronological running order.

# Steps to run the project

## 1. Run the original Scico model
First you should run the next part (the original Scico, only with the multiclass part) [here](#original-scico-walkthrough)  with the original multiclass.yaml config: https://github.com/ariecattan/SciCo/blob/main/configs/multiclass.yaml#L1 and be familiar with it 

## 2. Create the definition extractor model

The unarxiv data is a dataset of scientific papers, we will use it to retrieve definitions for the scientific concepts in the Scico dataset.

In this part we will do the following:

1. Save the unarxiv data in a format that is easy to work with
2. Embed each abstract and its metadata with the Instructor text embeddings [instructor](https://instructor-embedding.github.io/)
3. Create a vector store of the embeddings for fast retrieval using [chroma](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/chroma)
4. Create the llm model we use create the definitions (I used [wizard](https://huggingface.co/TheBloke/wizardLM-7B-HF))
5. Then we will have the complete definition extractor model. The model will get a query with a term, and it's context, extract the k most similar abstracts from the unarxiv dataset, and then define the term using the llm model and said abstracts.

![Alt text](definitions/definition-extarctor.png?raw=true "Title")

To create the above model, open the file definition_extractor.py and change the paths to the paths you want to use (look for paths that have /cs/labs/tomhope/forer11 in them).

Then in definition_extractor.py, uncomment the line: run_example_retrieval().

## 3. update the Scico dataset with the definitions

1. In definition_extractor.py, comment the line: run_example_retrieval().
2. Remember to search the project for remaining local paths 
3. In the multiclass.yaml config, change 'definition_extraction' to False and 'should_save_definition' to True
4. Run train 1 time with the new config, this will save the definitions in the Scico dataset

## 4. Run the Scico model with the definitions

1. In the multiclass.yaml config, change 'definition_extraction' to True and 'should_save_definition' to False
2. Run train as usual
3. run run_coref_scorer.py 
4. change the threshold in multiclass.yaml to the best threshold you found
5. run predict as usual
6. run evaluate as usual

## 5. Compare to the original Scico model

Run compare_results.py with the paths to the original Scico model and the new Scico model. This will output 4 files where in each you will have the terms that the second model improved on the first model on each category (same cluster, a->b, b->a, no relation).







## Original Scico walkthrough

## Dataset

You can load SciCo directly from [huggingface.co/datasets/allenai/scico](https://huggingface.co/datasets/allenai/scico) as follows:

```python
from datasets import load_dataset
scico = load_dataset("allenai/scico")
```

To download the raw data, click [here](https://nlp.biu.ac.il/~ariecattan/scico/data.tar).

Each file (train, dev, test) is in the `jsonl` format where each row corresponds a topic.
See below the description of the fields in each topic.

* `flatten_tokens`: a single list of all tokens in the topic
* `flatten_mentions`: array of mentions, each mention is represented by [start, end, cluster_id]
* `tokens`: array of paragraphs 
* `doc_ids`: doc_id of each paragraph in `tokens`
* `metadata`: metadata of each doc_id 
* `sentences`: sentences boundaries for each paragraph in `tokens` [start, end]
* `mentions`: array of mentions, each mention is represented by [paragraph_id, start, end, cluster_id]
* `relations`: array of binary relations between cluster_ids [parent, child]
* `id`: id of the topic 
* `hard_10` and `hard_20` (only in the test set): flag for 10% or 20% hardest topics based on Levenshtein similarity.
* `source`: source of this topic PapersWithCode (pwc), hypernym or curated. 


## Model

Our unified model is available on https://huggingface.co/allenai/longformer-scico.
We provide the following code as an example to set the global attention on the special tokens: `<s>`, `<m>` and `</m>`.

 

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-scico')
model = AutoModelForSequenceClassification.from_pretrained('allenai/longformer-scico')

start_token = tokenizer.convert_tokens_to_ids("<m>")
end_token = tokenizer.convert_tokens_to_ids("</m>")

def get_global_attention(input_ids):
    global_attention_mask = torch.zeros(input_ids.shape)
    global_attention_mask[:, 0] = 1  # global attention to the CLS token
    start = torch.nonzero(input_ids == start_token) # global attention to the <m> token
    end = torch.nonzero(input_ids == end_token) # global attention to the </m> token
    globs = torch.cat((start, end))
    value = torch.ones(globs.shape[0])
    global_attention_mask.index_put_(tuple(globs.t()), value)
    return global_attention_mask
    
m1 = "In this paper we present the results of an experiment in <m> automatic concept and definition extraction </m> from written sources of law using relatively simple natural methods."
m2 = "This task is important since many natural language processing (NLP) problems, such as <m> information extraction </m>, summarization and dialogue."

inputs = m1 + " </s></s> " + m2  

tokens = tokenizer(inputs, return_tensors='pt')
global_attention_mask = get_global_attention(tokens['input_ids'])

with torch.no_grad():
    output = model(tokens['input_ids'], tokens['attention_mask'], global_attention_mask)
    
scores = torch.softmax(output.logits, dim=-1)
# tensor([[0.0818, 0.0023, 0.0019, 0.9139]]) -- m1 is a child of m2
```


**Note:** There is a slight difference between this model and the original model presented in the [paper](https://openreview.net/forum?id=OFLbgUP04nC). 
The original model includes a single linear layer on top of the `<s>` token (equivalent to `[CLS]`) 
while this model includes a two-layers MLP to be in line with `LongformerForSequenceClassification`.   
You can download the original model as follows:
```python
curl -L -o model.tar https://www.dropbox.com/s/cpcnpov4liwuyd4/model.tar?dl=0
tar -xvf model.tar 
rm model.tar 
```


## Training and Evaluation 

### Getting started:

You may wish to create a conda environment:
```
conda create --name scico python=3.8
conda activate scico 
```
 
Install all dependencies using `pip install -r requirements.txt`. \
We provide the code for training the baseline models, Pipeline and Multiclass.



### Baseline

The baseline model uses our recent cross-document coreference model [(Cattan et al., 2021)](https://aclanthology.org/2021.findings-acl.453.pdf), 
the code is in [this](https://github.com/ariecattan/coref) repo.

* __Training__: Set the `configs/config_pairwise.json` file: select any BERT model to run in the field `bert_model` and set the directory to save the model in `model_path`.
This script will save a model at each epoch. 

```
python train_cd_coref_scorer.py --config configs/config_pairwise.json
```

* __Fine-tuning threshold__: (1) Run inference on the dev set using all the saved models and different values of thresholds 
and (2) run the scorer on the above predictions to get the best model with the best threshold. Make sure to set `data_path` 
to the dev set path, `bert_model` to the corresponding BERT model, and `save_path` to 
the corresponding directory to save the conll files.

```
python tune_coref_threshold.py --config configs/config_clustering_cs_roberta
python run_coref_scorer [folder_dev_pred] [gold_dev_conll]
```

* __Inference__: Run inference on the test test, make sure to set the `data_path` 
to the test set path.  You also need to set the name of an `nli_model` in the config 
for predicting the relations between the clusters. 
```
python predict_cd_coref_entailment.py --config configs/config_clustering_cs_roberta
```

* __Inference (cosine similarity)__: We also provide a script for clustering the 
mentions using an agglomerative clustering only on cosine similarity between
the average-pooling of the mentions. Relations between clusters are also predicted using an entailment 
model. 
```
python predict_cosine_similarity.py --gpu 0 \
    --data_path data/test.jsonl \
    --output_dir checkpoints/cosine_roberta \
    --bert_model roberta-large \
    --nli_model roberta-large-mnli \
    --threshold 0.5 
``` 


### Cross-Encoder pipeline and Multiclass

For both training and inference, running the pipeline or the multiclass model
can be done with only modifying the args `--multiclass` to {pipeline, multiclass}.


* __Training__:  Set important config for the model and data path in `configs/multiclass.yaml`,
then run the following script: 
```
python train.py --config configs/multiclass.yaml \
    --multiclass multiclass # (or coref or hypernym) 
```
  

* __Fine tuning threshold__: After training the multiclass model, we need to tune on the dev set 
the threshold for the agglomerative clustering and the stopping criterion for the 
hierarchical relations. 

```
python tune_hp_multiclass.py --config configs/multiclass.yaml 
```


* __Inference__: Set the path to the checkpoints of the models and the best thresholds, run
the following script on the test set.

```
python predict.py --config configs/multiclass.yaml \
    --multiclass multiclass # (or pipeline) 
```


### Evaluation 

Each inference script produces a `jsonl` file with the fields `tokens`, `mentions`, `relations` and `id`.
Models are evaluated using the usual coreference metrics using the [coval](https://github.com/ns-moosavi/coval/) script,
 hierarchy (recall, precision and F1), and directed path ratio. 

```
python evaluate.py [gold_jsonl_path] [sys_jsonl_path] options
```

If you want to evaluate only on the hard topics (based on levenshtein performance, see Section 4.5), 
you can set the `options` to be `hard_10`, `hard_20` or `curated`.
