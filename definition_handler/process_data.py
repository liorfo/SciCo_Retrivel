from SciCo_Retrivel.models.datasets import CrossEncoderDataset
import yaml


class DatasetsHandler:
    def __init__(self, test=True, train=True, dev=True, only_hard_10=False):
        if test:
            self.test_dataset = CrossEncoderDataset('/cs/labs/tomhope/forer11/SciCo_Retrivel/data/test.jsonl',
                                                    full_doc=False,
                                                    is_training=False,
                                                    data_label='test',
                                                    should_save_term_context=True,
                                                    should_load_definition=False,
                                                    only_hard_10=only_hard_10)

        if train:
            self.train_dataset = CrossEncoderDataset('/cs/labs/tomhope/forer11/SciCo_Retrivel/data/train.jsonl',
                                                     full_doc=False,
                                                     multiclass='multiclass',
                                                     should_save_term_context=True,
                                                     should_load_definition=False,
                                                     only_hard_10=only_hard_10)
        if dev:
            self.dev_dataset = CrossEncoderDataset('/cs/labs/tomhope/forer11/SciCo_Retrivel/data/dev.jsonl',
                                                   full_doc=False,
                                                   multiclass='multiclass',
                                                   data_label='dev',
                                                   is_training=False,
                                                   should_save_term_context=True,
                                                   should_load_definition=False,
                                                   only_hard_10=only_hard_10)


# with open('configs/multiclass.yaml', 'r') as f:
#     config = yaml.safe_load(f)
# data = DatasetsHandler()
# print(data.test_dataset[0])
