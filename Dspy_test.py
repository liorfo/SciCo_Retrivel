import dspy
from definition_handler.process_data import DatasetsHandler
import random

NUM_OF_TRAIN_DATA = 50

OPENAI_API_KEY = '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'

data = DatasetsHandler(test=True, train=False, dev=False)

data_set = data.test_dataset
random_indexes = random.sample(range(len(data_set)), NUM_OF_TRAIN_DATA)

train = [dspy.Example(question=data_set.pairs[i], answer=data_set.natural_labels[i]).with_inputs('question') for i in random_indexes]
print(train[0])
print(f"For this dataset, training examples have input keys {train[0].inputs().keys()} and label keys {train[0].labels().keys()}")

turbo = dspy.OpenAI(model='gpt-3.5-turbo', model_type='chat', api_key=OPENAI_API_KEY)

print(turbo("What is the meaning of life?"))