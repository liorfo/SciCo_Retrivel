import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BayesianSignatureOptimizer
from definition_handler.process_data import DatasetsHandler
import random
import pickle

NUM_OF_TRAIN_DATA = 200
NUM_OF_DEV_DATA = 60
OPENAI_API_KEY = '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'


class SCICO(dspy.Signature):
    ("""You are given 2 texts, each one is a context for a scientific concept inside <m></m> tags, for example: <m>example concept</m>."""
    """You must decide the correct hierarchical relation between the two concepts from the next options
    0 - No relation, no hierarchical connection
    1 - Same level, co-referring concepts
    2 - Term A is a parent concept of concept B
    3 - Term A is a child concept of concept B """)

    text_1 = dspy.InputField()
    text_2 = dspy.InputField()
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


def get_both_sentences(sentence):
    sentences = sentence.split('</s>')
    return sentences[0], sentences[1]

def get_dspy_example(data_set, num_of_data, shuffle=True, all_data=False):
    if shuffle:
        random.seed(1)
        label_0_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '0'], num_of_data // 4)
        label_1_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '1'], num_of_data // 4)
        label_2_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '2'], num_of_data // 4)
        label_3_indices = random.sample([index for index, label in enumerate(data_set.natural_labels) if label == '3'], num_of_data // 4)
        indexes = label_0_indices + label_1_indices + label_2_indices + label_3_indices

    elif all_data:
        indexes = [i for i in range(len(data_set))]
    else:
        indexes = [i for i in range(0, num_of_data, 200)]

    texts = [(get_both_sentences(data_set.pairs[i])) for i in indexes]
    labels = [data_set.natural_labels[i] for i in indexes]

    return [
        dspy.Example(
            text_1=texts[i][0],
            text_2=texts[i][1],
            answer=labels[i])
        .with_inputs('text_1', 'text_2') for i in range(len(texts))
    ]



data = DatasetsHandler(test=True, train=True, dev=True, only_hard_10=True)

train = get_dspy_example(data.train_dataset, NUM_OF_TRAIN_DATA)
dev = get_dspy_example(data.dev_dataset, NUM_OF_DEV_DATA)
test = get_dspy_example(data.test_dataset, len(data.test_dataset), shuffle=False, all_data=True)

print(
    f"For this dataset, training examples have input keys {train[0].inputs().keys()} and label keys {train[0].labels().keys()}")

turbo = dspy.OpenAI(model='gpt-3.5-turbo-0125', model_type='chat', max_tokens=250, api_key=OPENAI_API_KEY)

# GPT-4 will be used only to bootstrap CoT demos:
gpt4T = dspy.OpenAI(model='gpt-4-0125-preview', max_tokens=350, model_type='chat', api_key=OPENAI_API_KEY)

accuracy = dspy.evaluate.metrics.answer_exact_match

evaluator = Evaluate(devset=test, num_threads=1, display_progress=True, display_table=0, return_outputs=True)


dspy.settings.configure(lm=turbo)

bootstrap_optimizer = BootstrapFewShotWithRandomSearch(
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=4,
    num_threads=8,
    # teacher_settings=dict(lm=gpt4T),
    metric=accuracy)

# cot_zeroshot = CoTSCICOModule()
# kwargs = dict(num_threads=8, display_progress=True, display_table=0)
# optuna_trials_num =10 # Use more trials for better results
# teleprompter = BayesianSignatureOptimizer(task_model=turbo, prompt_model=turbo, metric=accuracy, n=5, init_temperature=1.0, verbose=True)
# compiled_prompt_opt = teleprompter.compile(cot_zeroshot, devset=dev, optuna_trials_num=optuna_trials_num, max_bootstrapped_demos=4, max_labeled_demos=4, eval_kwargs=kwargs)
# compiled_prompt_opt.save("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/BayesianSignatureOptimizer_program.json")

# cot_fewshot(**test[1].inputs())
# print(turbo.inspect_history(n=1))


# cot_fewshot = bootstrap_optimizer.compile(cot_fewshot, trainset=train, valset=dev)
# cot_fewshot.save("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/cot_fewshot-turbo-gpt4-equal-separation_3.json")

# cot_fewshot = CoTSCICOModule()
# cot_fewshot.load("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/cot_fewshot-turbo-gpt4-demo.json")
# cot_fewshot(**test[0].inputs())
# print(turbo.inspect_history(n=1))


# evaluator = Evaluate(devset=test, num_threads=1, display_progress=True, display_table=0)
# # basic_module = BaseSCICOModule()
# # basic_module(**test[0].inputs())
# cot_module = CoTSCICOModule()
# cot_module(**test[0].inputs())
# print(turbo.inspect_history(n=1))


cot_fewshot = CoTSCICOModule()
cot_fewshot.load("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/cot_fewshot-turbo-no-gpt4-equal-separation_bayesian.json")
score, results = evaluator(cot_fewshot, metric=accuracy)
anwsers = [result[1].answer for result in results]
with open("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/score_results.pkl", "wb") as file:
    pickle.dump({'score': score, 'answers': anwsers}, file)

# with open("/cs/labs/tomhope/forer11/SciCo_Retrivel/DSPY/score_results.pkl", "rb") as file:
#     loaded_data = pickle.load(file)

# print(loaded_data['score'])