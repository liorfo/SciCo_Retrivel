import jsonlines
import sys
from itertools import combinations
import pickle
import textwrap
import re
import csv

def_comparison_line_format = """Correct class: {gold_class} 
Mistral no def class: {mistral_no_def_class}
Mistral singleton def class: {mistral_singleton_def_class}
Mistral gpt def class: {mistral_gpt_def_class}
Mistral relational def class: {mistral_relational_def_class}

First mention: {first_mention}

First sentence: {first_sentence}

First singelton def: {first_singleton_def}

First gpt def: {first_gpt_def}

First relational def: {first_relational_def}


Second mention: {second_mention}

Second sentence: {second_sentence}

Second singelton def: {second_singleton_def}

Second gpt def: {second_gpt_def}

Second relational def: {second_relational_def}
"""

def only_relational_right(gold_tag, no_def_tag, singleton_def_tag, relational_def_tag, gpt_def_tag):
    if gold_tag == relational_def_tag and gold_tag != no_def_tag and gold_tag != singleton_def_tag and gold_tag != gpt_def_tag:
        return True
    return False

def singleton_and_relational_right(gold_tag, no_def_tag, singleton_def_tag, relational_def_tag, gpt_def_tag):
    if gold_tag == singleton_def_tag and gold_tag == relational_def_tag and gold_tag != no_def_tag and gold_tag != gpt_def_tag:
        return True
    return False

def relational_right_else_whatever(gold_tag, _no_def_tag, _singleton_def_tag, relational_def_tag, _gpt_def_tag):
    if gold_tag == relational_def_tag:
        return True
    return False

def only_gpt_wrong(gold_tag, no_def_tag, singleton_def_tag, relational_def_tag, gpt_def_tag):
    if gold_tag != gpt_def_tag and gold_tag == no_def_tag and gold_tag == singleton_def_tag and gold_tag == relational_def_tag:
        return True
    return False

def generate_mention_couples(mentions):
    mention_couples = list(combinations(mentions, 2))
    return mention_couples

def only_no_def_right(gold_tag, no_def_tag, singleton_def_tag, relational_def_tag, gpt_def_tag):
    if gold_tag == no_def_tag and gold_tag != singleton_def_tag and gold_tag != relational_def_tag and gold_tag != gpt_def_tag:
        return True
    return False

def no_def_right_singleton_wrong(gold_tag, no_def_tag, singleton_def_tag, relational_def_tag, gpt_def_tag):
    if gold_tag == no_def_tag and gold_tag != singleton_def_tag:
        return True
    return False

def singleton_right_gpt_wrong(gold_tag, singleton_def_tag, gpt_def_tag):
    if gold_tag == singleton_def_tag and gold_tag != gpt_def_tag:
        return True
    return False

def only_no_def_wrong(gold_tag, no_def_tag, singleton_def_tag, relational_def_tag, gpt_def_tag):
    if gold_tag != no_def_tag and gold_tag == singleton_def_tag and gold_tag == relational_def_tag and gold_tag == gpt_def_tag:
        return True
    return False
def no_def_wrong_someone_right(gold_tag, no_def_tag, singleton_def_tag, relational_def_tag, gpt_def_tag):
    if gold_tag != no_def_tag and (gold_tag == singleton_def_tag or gold_tag == relational_def_tag or gold_tag == gpt_def_tag):
        return True
    return False

def no_def_right_someone_wrong(gold_tag, no_def_tag, singleton_def_tag, relational_def_tag, gpt_def_tag):
    if gold_tag == no_def_tag and (gold_tag != singleton_def_tag or gold_tag != relational_def_tag or gold_tag != gpt_def_tag):
        return True
    return False
def singleton_wrong_else_whatever(gold_tag, no_def_tag, singleton_def_tag, relational_def_tag, gpt_def_tag):
    if gold_tag != singleton_def_tag:
        return True
    return False

def relational_wrong_else_whatever(gold_tag, no_def_tag, singleton_def_tag, relational_def_tag, gpt_def_tag):
    if gold_tag != relational_def_tag:
        return True
    return False

def extract_term(text):
    return re.search(r'<m>(.*?)</m>', text).group(1)

def get_full_doc_mention(mention, tokens):
    doc_id, start, end, _ = mention

    mention_rep = tokens[doc_id][:start] + ['<m>']
    mention_rep += tokens[doc_id][start:end + 1] + ['</m>']
    mention_rep += tokens[doc_id][end + 1:] + ['</s>']

    return ' '.join(mention_rep)

def get_couple_class_tag(couple, relations):
    cluster_x, cluster_y = couple[0][-1], couple[1][-1]
    if cluster_x == cluster_y:
        return 'same cluster'
    elif [cluster_x, cluster_y] in relations:
        return 'first -> second'
    elif [cluster_y, cluster_x] in relations:
        return 'second -> first'
    else:
        return 'no relation'


def get_sentences(gold, gold_couple):
    first_sentence = get_full_doc_mention(gold_couple[0], gold[topic_index]['tokens'])
    second_sentence = get_full_doc_mention(gold_couple[1], gold[topic_index]['tokens'])
    return first_sentence, second_sentence


def get_sentences(gold, gold_couple):
    first_sentence = get_full_doc_mention(gold_couple[0], gold[topic_index]['tokens'])
    second_sentence = get_full_doc_mention(gold_couple[1], gold[topic_index]['tokens'])
    return first_sentence, second_sentence

if __name__ == '__main__':
    gold_path = '/cs/labs/tomhope/forer11/SciCo_Retrivel/data/test.jsonl'
    mistral_singleton_def_path = '/cs/labs/tomhope/forer11/SciCo/res_with_def/system_0.6_0.4.jsonl'
    mistral_gpt_def_path = '/cs/labs/tomhope/forer11/SciCo_Retrivel/mistral_1_classification/with_gpt_4_def/results/system_0.6_0.4.jsonl'
    hard = False
    save_as_csv = True  # Add this flag to toggle between text and CSV output

    file_concatenation = ''
    csv_rows = []  # Collect rows for CSV

    with jsonlines.open(gold_path, 'r') as f:
        gold = [line for line in f]

    with jsonlines.open(mistral_singleton_def_path, 'r') as f:
        mistral_singleton_def = [line for line in f]

    with jsonlines.open(mistral_gpt_def_path, 'r') as f:
        mistral_gpt_def = [line for line in f]

    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/full_texts/test_terms_definitions_final.pickle', 'rb') as f:
        singleton_def = pickle.load(f)
    with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/data/gpt_4_definitions/def_files/test_terms_definitions_final.pickle', 'rb') as f:
        gpt_def = pickle.load(f)

    if hard:
        gold = [topic for topic in gold if topic['hard_20'] == True]
        mistral_singleton_def = [topic for topic in mistral_singleton_def if topic['id'] in [x['id'] for x in gold]]

    for topic_index in range(len(gold)):
        all_gold_couples, all_mistral_singleton_def_couples, all_mistral_gpt_def_couples = (
            generate_mention_couples(gold[topic_index]['mentions']),
            generate_mention_couples(mistral_singleton_def[topic_index]['mentions']),
            generate_mention_couples(mistral_gpt_def[topic_index]['mentions']))

        gold_relations, mistral_singleton_def_relations, mistral_gpt_def_relations = (
            gold[topic_index]['relations'],
            mistral_singleton_def[topic_index]['relations'],
            mistral_gpt_def[topic_index]['relations'])

        for couple_index in range(len(all_gold_couples)):
            gold_couple, mistral_singleton_def_couple, mistral_gpt_def_couple = (
                all_gold_couples[couple_index],
                all_mistral_singleton_def_couples[couple_index],
                all_mistral_gpt_def_couples[couple_index])

            gold_couple_class = get_couple_class_tag(gold_couple, gold_relations)
            mistral_singleton_def_couple_class = get_couple_class_tag(mistral_singleton_def_couple, mistral_singleton_def_relations)
            mistral_gpt_def_couple_class = get_couple_class_tag(mistral_gpt_def_couple, mistral_gpt_def_relations)

            first_sent, second_sent = get_sentences(gold, gold_couple)
            first_mention = extract_term(first_sent)
            second_mention = extract_term(second_sent)

            first_singleton_def = singleton_def[first_sent]
            second_singleton_def = singleton_def[second_sent]
            first_gpt_def = gpt_def[first_sent]
            second_gpt_def = gpt_def[second_sent]

            # Call singleton_right_gpt_wrong function
            if singleton_right_gpt_wrong(gold_couple_class, mistral_singleton_def_couple_class, mistral_gpt_def_couple_class):
                if save_as_csv:
                    csv_rows.append([
                        gold_couple_class, mistral_singleton_def_couple_class, mistral_gpt_def_couple_class,
                        first_mention, first_sent, first_singleton_def, first_gpt_def, second_mention, second_sent, second_singleton_def, second_gpt_def
                    ])
                else:
                    comparison_line = def_comparison_line_format.format(
                        gold_class=gold_couple_class,
                        mistral_singleton_def_class=mistral_singleton_def_couple_class,
                        mistral_gpt_def_class=mistral_gpt_def_couple_class,
                        first_sentence=first_sent, first_mention=first_mention,
                        first_singleton_def=first_singleton_def, first_gpt_def=first_gpt_def,
                        second_sentence=second_sent, second_mention=second_mention,
                        second_singleton_def=second_singleton_def, second_gpt_def=second_gpt_def
                    )
                    file_concatenation += comparison_line
                    file_concatenation += '\n\n======================================================================================================================\n\n'

    if save_as_csv:
        with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/definition_comparison/comparisons_with_singleton_and_gpt.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Gold Class', 'Mistral Singleton Def Class', 'Mistral GPT Def Class',
                             'First Mention', 'First Sentence', 'First Singleton Def', 'First GPT Def',
                             'Second Mention', 'Second Sentence', 'Second Singleton Def', 'Second GPT Def'])
            writer.writerows(csv_rows)
    else:
        with open('/cs/labs/tomhope/forer11/SciCo_Retrivel/definition_handler/definition_comparison/comparisons_with_singleton_and_gpt.txt', 'w') as f:
            f.write(file_concatenation)

    print('done!!')