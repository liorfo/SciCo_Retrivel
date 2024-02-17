import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BayesianSignatureOptimizer
from definition_handler.process_data import DatasetsHandler
import random
import pickle

NUM_OF_TRAIN_DATA = 200
NUM_OF_DEV_DATA = 60
OPENAI_API_KEY = ''


class SCICO(dspy.Signature):
    ("""You are given 2 texts, each one is a context for a scientific concept inside <m></m> tags, for example: <m>example concept</m>."""
    """ You must decide the correct hierarchical relation between the two concepts from the next options
    0 - No relation, no hierarchical connection
    1 - Same level, co-referring concepts
    2 - Term A is a parent concept of concept B
    3 - Term A is a child concept of concept B """)

    text_1 = dspy.InputField()
    text_2 = dspy.InputField()
    answer = dspy.OutputField(desc="The correct hierarchical relation between the two concepts from the next options: 0, 1, 2, 3.")


class ScicoWithDef(dspy.Signature):
    ("""You are given 2 texts, each one is a context for a scientific concept inside <m></m> tags, for example: <m>example concept</m>."""
     """ you also get a definition for each concept."""
    """You must decide the correct hierarchical relation between the two concepts from the next options
    0 - No relation, no hierarchical connection
    1 - Same level, co-referring concepts
    2 - Term A is a parent concept of concept B
    3 - Term A is a child concept of concept B """)

    text_1 = dspy.InputField(desc="The first text with a scientific concept")
    definition_1 = dspy.InputField(desc="The definition of the first concept")
    text_2 = dspy.InputField(desc="The second text with a scientific concept")
    definition_2 = dspy.InputField(desc="The definition of the second concept")
    answer = dspy.OutputField(desc="The correct hierarchical relation between the two concepts from the next options: 0, 1, 2, 3.")


class BaseSCICOModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_hierarchy = dspy.Predict(SCICO)

    def forward(self, text_1, text_2):
        return self.generate_hierarchy(text_1=text_1, text_2=text_2)

class CoTSCICOModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_hierarchy = dspy.ChainOfThought(SCICO)

    def forward(self, text_1, text_2):
        return self.generate_hierarchy(text_1=text_1, text_2=text_2)

class CoTScicoWithDefModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_hierarchy = dspy.ChainOfThought(ScicoWithDef)

    def forward(self, text_1, text_2, definition_1, definition_2):
        return self.generate_hierarchy(text_1=text_1, text_2=text_2, definition_1=definition_1, definition_2=definition_2)


def get_both_sentences(sentence):
    sentences = sentence.split('</s>')
    return sentences[0], sentences[1]

def get_definitions(pair, def_dict):
    sent_1, sent_2 = pair
    return def_dict[sent_1 + '</s>'], def_dict[sent_2 + '</s>']


def get_dspy_example(data_set, num_of_data, shuffle=True, all_data=False, with_def=False):
    if shuffle:
        random.seed(1)
        label_0_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '0'], num_of_data // 4)
        label_1_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '1'], num_of_data // 4)
        label_2_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '2'], num_of_data // 4)
        label_3_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '3'], num_of_data // 4)
        indexes = label_0_indices + label_1_indices + label_2_indices + label_3_indices
        random.shuffle(indexes)

    elif all_data:
        indexes = [i for i in range(len(data_set))]
    else:
        indexes = [i for i in range(0, num_of_data, 200)]

    texts = [(get_both_sentences(data_set.pairs[i])) for i in indexes]
    labels = [data_set.natural_labels[i] for i in indexes]

    if with_def:
        definitions = [get_definitions(sentences, data_set.combined_def_dict) for sentences in texts]
        return [
            dspy.Example(
                text_1=texts[i][0],
                text_2=texts[i][1],
                definition_1=definitions[i][0],
                definition_2=definitions[i][1],
                answer=labels[i])
            .with_inputs('text_1', 'text_2', 'definition_1', 'definition_2') for i in range(len(texts))
        ]

    return [
        dspy.Example(
            text_1=texts[i][0],
            text_2=texts[i][1],
            answer=labels[i])
        .with_inputs('text_1', 'text_2') for i in range(len(texts))
    ]



data = DatasetsHandler(test=True, train=True, dev=True, only_hard_10=True)

train = get_dspy_example(data.train_dataset, NUM_OF_TRAIN_DATA, with_def=True)
dev = get_dspy_example(data.dev_dataset, NUM_OF_DEV_DATA, with_def=True)
test = get_dspy_example(data.test_dataset, len(data.test_dataset), shuffle=False, all_data=True, with_def=True)

print(
    f"For this dataset, training examples have input keys {train[0].inputs().keys()} and label keys {train[0].labels().keys()}")

turbo = dspy.OpenAI(model='gpt-3.5-turbo-0125', model_type='chat', max_tokens=250, api_key=OPENAI_API_KEY)

# GPT-4 will be used only to bootstrap CoT demos:
gpt4T = dspy.OpenAI(model='gpt-4-0125-preview', max_tokens=350, model_type='chat', api_key=OPENAI_API_KEY)

accuracy = dspy.evaluate.metrics.answer_exact_match

dspy.settings.configure(lm=turbo)

bootstrap_optimizer = BootstrapFewShotWithRandomSearch(
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=4,
    num_threads=12,
    # teacher_settings=dict(lm=gpt4T),
    metric=accuracy)

# cot_zeroshot = CoTScicoWithDefModule()
# kwargs = dict(num_threads=8, display_progress=True, display_table=0)
# optuna_trials_num =10 # Use more trials for better results
# teleprompter = BayesianSignatureOptimizer(task_model=turbo, prompt_model=turbo, metric=accuracy, n=5, init_temperature=1.0, verbose=True)
# compiled_prompt_opt = teleprompter.compile(cot_zeroshot, devset=dev, optuna_trials_num=optuna_trials_num, max_bootstrapped_demos=4, max_labeled_demos=4, eval_kwargs=kwargs)
# # compiled_prompt_opt.save("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/BayesianSignatureOptimizer_program_with_def_2.json")

# cot_fewshot(**test[1].inputs())
# print(turbo.inspect_history(n=1))

# cot_fewshot = CoTScicoWithDefModule()
# cot_fewshot.load("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/BayesianSignatureOptimizer_program_with_def_2.json")
# cot_fewshot = bootstrap_optimizer.compile(cot_fewshot, trainset=train, valset=dev)
# cot_fewshot.save("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/cot_def_new_with_sig_opt.json")

# cot_fewshot = CoTScicoWithDefModule()
# cot_fewshot.load("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/BayesianSignatureOptimizer_program_with_def_2.json")
# cot_fewshot(**test[0].inputs())
# print(turbo.inspect_history(n=1))


# evaluator = Evaluate(devset=test, num_threads=1, display_progress=True, display_table=0)
# # basic_module = BaseSCICOModule()
# # basic_module(**test[0].inputs())
# cot_module = CoTSCICOModule()
# cot_module(**test[0].inputs())
# print(turbo.inspect_history(n=1))


# cot_fewshot = CoTScicoWithDefModule()
# cot_fewshot.load("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/BayesianSignatureOptimizer_program_with_def_2.json")
#
# chunk_size = 1000
# with open("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/sorted_results/score_results_until_29000.pkl", "rb") as file:
#     loaded_data = pickle.load(file)
# all_answers = loaded_data['answers']
# # all_answers = []
# for i in range(29000, len(test), chunk_size):
#     chunk = test[i:i+chunk_size]
#     print("Evaluating until: ", i + chunk_size)
#     is_success = False
#     while not is_success:
#         try:
#             evaluator = Evaluate(devset=chunk, num_threads=4, display_progress=True, display_table=0, return_outputs=True)
#             score, results = evaluator(cot_fewshot, metric=accuracy)
#             anwsers = [prediction.answer for _, example, prediction, temp_score in results]
#             all_answers.extend(anwsers)
#             is_success = True
#         except Exception as e:
#             print(e)
#             print("Retrying...")
#     # evaluator = Evaluate(devset=chunk, num_threads=4, display_progress=True, display_table=0, return_outputs=True)
#     # score, results = evaluator(cot_fewshot, metric=accuracy)
#     with open(f'/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/sorted_results/score_results_until_{i + chunk_size}.pkl', "wb") as file:
#         pickle.dump({'score': score, 'answers': all_answers}, file)
#     print("Processed chunk", i//chunk_size)

sentences_to_score_dict = {}

with open("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/sorted_results/score_results_until_70000.pkl", "rb") as file:
    loaded_data3 = pickle.load(file)

for i, sentences in enumerate(data.test_dataset.pairs):
    sentences_to_score_dict[sentences] = loaded_data3['answers'][i]

with open("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/sorted_results/sentences_to_score_dict.pkl", "wb") as file:
    pickle.dump(sentences_to_score_dict, file)
