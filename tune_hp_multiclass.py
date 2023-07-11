import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import socket
import yaml
from torch.utils import data
import logging
from datetime import datetime
import os
import torch
from tqdm import tqdm
import numpy as np
from models.datasets import CrossEncoderDataset
from models.muticlass import MulticlassCrossEncoder
from predict import MulticlassInference

from eval.shortest_path import ShortestPath
import gc

if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/multiclass.yaml')
    parser.add_argument('--full_doc', type=str, default='1')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['full_doc'] = False if args.full_doc == '0' else True

    root_logger = logging.getLogger()
    logger = root_logger.getChild(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=os.path.join(
        config['log'], '{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.info("pid: {}".format(os.getpid()))
    logger.info('Server name: {}'.format(socket.gethostname()))

    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])

    logger.info('loading models')
    model = MulticlassCrossEncoder.load_from_checkpoint(config['checkpoint_multiclass'], config=config)
    dev = CrossEncoderDataset(config["data"]["dev_set"], full_doc=config['full_doc'], multiclass='multiclass',
                              should_load_definition=True, data_label='dev')
    dev_loader = data.DataLoader(dev,
                                 batch_size=config["model"]["batch_size"] * 4,
                                 shuffle=False,
                                 collate_fn=model.tokenize_batch,
                                 num_workers=16,
                                 pin_memory=True)

    pl_logger = CSVLogger(save_dir='logs', name='multiclass_inference')
    trainer = pl.Trainer(gpus=config['gpu_num'], accelerator='dp')
    results = trainer.predict(model, dataloaders=dev_loader)
    results = torch.cat([torch.tensor(x) for x in results])
    # torch.save(results, os.path.join(config['save_path'], 'dev_results.pt'))
    # results = torch.load('checkpoints/multiclass/dev_results.pt')
    coref_threshold = np.arange(0.4, 0.61, 0.1)
    hypernym_threshold = np.arange(0.4, 0.61, 0.1)

    ## run predict for all thresholds
    scores = []
    pairs = [(x, y) for x in coref_threshold for y in hypernym_threshold]
    logger.info(f'Predicting {len(pairs)} configurations...')
    for coref, hypernym in tqdm(pairs):
        inference = MulticlassInference(dev, results, coref, hypernym)
        inference.predict_cluster_relations()

        path_based = ShortestPath(dev.data, inference.predicted_data, directed=True, with_tn=False)
        scores.append(path_based.micro_average)

    best = np.argmax(scores)
    logger.info(f'Highest score: {scores[best]}')
    logger.info(f'coref threshold: {pairs[best][0]} hypernym threshold: f{pairs[best][1]}')
