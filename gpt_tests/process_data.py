from SciCo_Retrivel.models.datasets import CrossEncoderDataset
import yaml


class DatasetsHandler:
    def __init__(self, config):
        self.test_dataset = CrossEncoderDataset('../' + config["data"]["test_set"],
                                                full_doc=config['full_doc'],
                                                is_training=False,
                                                should_load_definition=False,
                                                data_label='test',
                                                should_save_term_context=True)

        self.train_dataset = CrossEncoderDataset('../' + config["data"]["training_set"],
                                                 full_doc=config['full_doc'],
                                                 should_save_term_context=True)

        self.dev_dataset = CrossEncoderDataset('../' + config["data"]["dev_set"],
                                               full_doc=config['full_doc'],
                                               data_label='dev',
                                               should_save_term_context=True)


with open('../configs/multiclass.yaml', 'r') as f:
    config = yaml.safe_load(f)
data = DatasetsHandler(config)
print(data.test_dataset[0])
