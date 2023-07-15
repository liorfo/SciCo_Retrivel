import jsonlines
import sys
import os

from eval.hypernym import HypernymScore
from eval.hypernym_50 import HypernymScore50
from eval.shortest_path import ShortestPath
from utils.conll import write_output_file
from coval.coval.conll import reader
from coval.coval.eval import evaluator
from itertools import combinations


def generate_mention_couples(mentions):
    mention_couples = list(combinations(mentions, 2))
    return mention_couples


def eval_coref(gold, system):
    allmetrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
                  ('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
                  ('lea', evaluator.lea)]

    NP_only = False
    remove_nested = False
    keep_singletons = False
    min_span = False

    conll = 0

    doc_coref_infos = reader.get_coref_infos(gold, system, NP_only, remove_nested, keep_singletons, min_span)
    scores = {}

    for name, metric in allmetrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos, metric, beta=1)
        scores[name] = [recall, precision, f1]

        if name in ["muc", "bcub", "ceafe"]:
            conll += f1

    scores['conll'] = conll
    return scores


def get_coref_scores(gold, system):
    output_path = 'tmp'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    write_output_file(gold, output_path, 'gold')
    write_output_file(system, output_path, 'system')
    coref_scores = eval_coref('tmp/gold_simple.conll', 'tmp/system_simple.conll')

    return coref_scores


if __name__ == '__main__':
    gold_path = sys.argv[1]
    sys_path = sys.argv[2]
    new_model_path = sys.argv[3]
    hard = sys.argv[4] if len(sys.argv) > 4 else None

    with jsonlines.open(gold_path, 'r') as f:
        gold = [line for line in f]

    with jsonlines.open(sys_path, 'r') as f:
        system = [line for line in f]

    with jsonlines.open(new_model_path, 'r') as f:
        new_model = [line for line in f]

    if hard:
        gold = [topic for topic in gold if topic[hard] == True]
        system = [topic for topic in system if topic['id'] in [x['id'] for x in gold]]
        new_model = [topic for topic in new_model if topic['id'] in [x['id'] for x in gold]]

    for i in range(len(gold)):
        all_gold_couples = generate_mention_couples(gold[i]['mentions'])
        all_system_couples = generate_mention_couples(system[i]['mentions'])
        print(len(all_system_couples))

    print(f'Number of topics to evaluate {len(gold)}')